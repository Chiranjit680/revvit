import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function as Function

# We use the standard pytorch multi-head attention module
from torch.nn import MultiheadAttention as MHA
import sys
import numpy as np

from har_blocks import AsymmetricMHPAReversibleBlock, AsymmetricSwinReversibleBlock

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

class AsymmetricRevVit(nn.Module):
    def __init__(
        self,
        block_type=None,
        const_dim=768,
        var_dim=[64, 128, 320, 512],
        sra_R=[8, 4, 2, 1],
        n_head=8,
        stages=[3, 3, 6, 3],
        drop_path_rate=0,
        patch_size=(2, 2),  
        image_size=(32, 32),  # CIFAR-10 image size
        num_classes=10,
        enable_amp=False,
    ):
        super().__init__()

        self.const_dim = const_dim
        self.n_head = n_head
        self.patch_size = patch_size
        self.layers_outputs = []  # Initialize as empty list
        
        self.const_num_patches = (image_size[0] // self.patch_size[0]) * (
            image_size[1] // self.patch_size[1]
        )

        const_patches_shape = (image_size[0] // self.patch_size[0], image_size[1] // self.patch_size[1])

        blk_idx_temp = 0
        block_to_stage_indexing = {}
        for stg_idx_temp, num_blocks in enumerate(stages):
            for _ in range(num_blocks):
                block_to_stage_indexing[blk_idx_temp] = stg_idx_temp
                blk_idx_temp += 1

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(stages))
        ]  # stochastic depth decay rule

        assert const_patches_shape[0] % 2**(len(stages)-1) == 0
        assert const_patches_shape[1] % 2**(len(stages)-1) == 0

        self.layers = []
        for i in range(sum(stages)):
            stage_index = block_to_stage_indexing[i]

            if block_type == "smlp":
                self.layers.append(
                    AsymmetricReversibleBlock(
                        dim_c=self.const_dim,
                        dim_v=var_dim[stage_index],
                        num_heads=self.n_head,
                        enable_amp=enable_amp,
                        sr_ratio=sra_R[stage_index],
                        token_map_pool_size=2**stage_index,
                        drop_path=dpr[i],
                        const_patches_shape=const_patches_shape,
                        block_id=i
                    )
                )
            elif block_type == "mhpa":
                self.layers.append(
                    AsymmetricMHPAReversibleBlock(
                        dim_c=self.const_dim,
                        dim_v=var_dim[stage_index],
                        num_heads=(2**stage_index),
                        enable_amp=enable_amp,
                        kv_pool_size=4,
                        token_map_pool_size=2**stage_index,
                        drop_path=dpr[i],
                        const_patches_shape=const_patches_shape,
                        block_id=i
                    )
                )
            elif block_type in ["swin-attention", "swin-mlp", "swin-dw-mlp"]:
                self.layers.append(
                    AsymmetricSwinReversibleBlock(
                        f_block=block_type,
                        dim_c=self.const_dim,
                        dim_v=var_dim[stage_index],
                        num_heads=3*(2**stage_index),
                        window_size=(8 if block_type=="swin-dw-mlp" else 7),
                        shift_size=(0 if (i % 2 == 0) else (8 if block_type=="swin-dw-mlp" else 7) // 2),
                        enable_amp=enable_amp,
                        token_map_pool_size=2**stage_index,
                        drop_path=dpr[i],
                        const_patches_shape=const_patches_shape,
                        block_id=i
                    )
                )
            else:
                raise ValueError(f"Invalid asymmetric block type: {block_type}")
            
            # Stage transitions
            if (i == np.cumsum(stages)[stage_index] - 1) and (stage_index != len(stages) - 1):
                self.layers.append(
                    VarStreamDownSamplingBlock(
                        input_patches_shape=(const_patches_shape[0] // 2**stage_index, 
                                           const_patches_shape[1] // 2**stage_index),
                        kernel_size=2, 
                        dim_in=var_dim[stage_index], 
                        dim_out=var_dim[stage_index+1]
                    )
                )

        self.layers = nn.ModuleList(self.layers)

        # Boolean to switch between vanilla backprop and rev backprop
        self.no_custom_backward = False

        # Standard Patchification and absolute positional embeddings as in ViT
        self.patch_embed2 = nn.Conv2d(
            3, self.const_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embeddings2 = nn.Parameter(
            torch.zeros(1, self.const_num_patches, self.const_dim)
        )

        self.patch_embed1 = nn.Conv2d(
            3, var_dim[0], kernel_size=patch_size, stride=patch_size
        )

        self.pos_embeddings1 = nn.Parameter(
            torch.zeros(1, self.const_num_patches, var_dim[0])
        )

        # Classification head using constant stream
        self.head = nn.Linear(self.const_dim, num_classes, bias=True)
        self.norm = nn.LayerNorm(self.const_dim)

    @staticmethod
    def vanilla_backward(x1, x2, layers):
        """
        Using rev layers without rev backpropagation. Debugging purposes only.
        Activated with self.no_custom_backward.
        """
        for _, layer in enumerate(layers):
            x1, x2 = layer(x1, x2)
        return x1, x2

    def forward(self, x):
        # Clear previous outputs at the start of each forward pass
        self.layers_outputs = []
        
        # Patchification using conv and flattening + absolute positional embeddings
        x2 = self.patch_embed2(x).flatten(2).transpose(1, 2)
        x2 += self.pos_embeddings2

        x1 = self.patch_embed1(x).flatten(2).transpose(1, 2)
        x1 += self.pos_embeddings1

        # Create segments for reversible computation
        reversible_segments = [[0]]
        
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], VarStreamDownSamplingBlock):
                reversible_segments[-1].append(i)
                reversible_segments.append([i+1])
        reversible_segments[-1].append(len(self.layers))

        for segment in reversible_segments:
            # No need for custom backprop in eval/inference phase
            if not self.training or self.no_custom_backward:
                executing_fn = AsymmetricRevVit.vanilla_backward
            else:
                executing_fn = RevBackProp.apply

            # Switch between vanilla backprop and rev backprop
            x1, x2 = executing_fn(
                x1, x2,
                self.layers[segment[0]:segment[1]],
            )

            # Apply downsampling layer if exists
            if segment[1] != len(self.layers):
                x1, x2 = self.layers[segment[1]](x1, x2)
            
            self.layers_outputs.append(x2.clone())

        # Aggregate across sequence length (global average pooling)
        pred = x2.mean(1)

        # Head pre-norm
        pred = self.norm(pred)

        # Pre-softmax logits
        pred = self.head(pred)

        return pred


class RevBackProp(Function):

    """
    Custom Backpropagation function to allow (A) flusing memory in foward
    and (B) activation recomputation reversibly in backward for gradient
    calculation. Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(
        ctx,
        X_1, X_2,
        layers,
    ):
        """
        Reversible Forward pass.
        Each reversible layer implements its own forward pass pass logic.
        """

        # obtaining X_1 and X_2 from the concatenated input
        # X_1, X_2 = torch.chunk(x, 2, dim=-1)
        # X_1, X_2 = torch.split(x, [x.size(1) - const_num_patches, const_num_patches], dim=1)

        for layer in layers:
            X_1, X_2 = layer(X_1, X_2)
            all_tensors = [X_1.detach(), X_2.detach()]

        # saving only the final activations of the last reversible block
        # for backward pass, no intermediate activations are needed.
        ctx.save_for_backward(*all_tensors)
        ctx.layers = layers

        return X_1, X_2

    @staticmethod
    def backward(ctx, dX_1, dX_2):
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        # obtaining gradients dX_1 and dX_2 from the concatenated input
        # dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve the last saved activations, to start rev recomputation
        X_1, X_2 = ctx.saved_tensors
        # layer weights
        layers = ctx.layers

        for _, layer in enumerate(layers[::-1]):
            # this is recomputing both the activations and the gradients wrt
            # those activations.
            X_1, X_2, dX_1, dX_2 = layer.backward_pass(
                Y_1=X_1,
                Y_2=X_2,
                dY_1=dX_1,
                dY_2=dX_2,
            )
        # final input gradient to be passed backward to the patchification layer
        # dx = torch.cat([dX_1, dX_2], dim=-1)

        # del dX_1, dX_2, X_1, X_2

        return dX_1, dX_2, None, None


class AsymmetricReversibleBlock(nn.Module):
    """
    Reversible Blocks for Reversible Vision Transformer.
    See Section 3.3.2 in paper for details.
    """

    def __init__(self, dim_c, dim_v, num_heads, enable_amp, drop_path, token_map_pool_size, sr_ratio, const_patches_shape, block_id, token_mixer="spatial_mlp"):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.

        self.drop_path_rate = drop_path

        # self.F = SRASubBlockC2V(
        #     dim_c=dim_c,
        #     dim_v=dim_v,
        #     num_heads=num_heads,
        #     patches_shape=const_patches_shape,
        #     token_pool_size=token_map_pool_size,
        #     sr_ratio=sr_ratio, 
        #     enable_amp=enable_amp
        # )

        self.F = TokenMixerFBlockC2V(
            dim_c=dim_c,
            dim_v=dim_v,
            patches_shape=const_patches_shape,
            token_pool_size=token_map_pool_size,
            enable_amp=enable_amp
        )

        self.G = MLPSubblockV2C(
            dim_c=dim_c,
            dim_v=dim_v,
            const_patches_shape=const_patches_shape,
            token_pool_size=token_map_pool_size,
            enable_amp=False,  # standard for ViTs
        )

        # note that since all functions are deterministic, and we are
        # not using any stochastic elements such as dropout, we do
        # not need to control seeds for the random number generator.
        # To see usage with controlled seeds and dropout, see pyslowfast.

        self.seeds = {}
        self.block_id = block_id
        print(f"Block index : {self.block_id} | Dpr : {self.drop_path_rate}")
    
    def seed_cuda(self, key):
        """
        Fix seeds to allow for stochastic elements such as
        dropout to be reproduced exactly in activation
        recomputation in the backward pass.
        """

        # randomize seeds
        # use cuda generator if available
        if (
            hasattr(torch.cuda, "default_generators")
            and len(torch.cuda.default_generators) > 0
        ):
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(self.seeds[key])

    def forward(self, X_1, X_2):
        """
        forward pass equations:
        Y_1 = X_1 + Attention(X_2), F = Attention
        Y_2 = X_2 + MLP(Y_1), G = MLP
        """

        self.seed_cuda("attn")
        # Y_1 : attn_output
        f_X_2 = self.F(X_2)

        self.seed_cuda("droppath")
        f_X_2_dropped = drop_path(
            f_X_2, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_1 = X_1 + f(X_2)
        Y_1 = X_1 + f_X_2_dropped

        # free memory since X_1 is now not needed
        del X_1

        self.seed_cuda("FFN")
        g_Y_1 = self.G(Y_1)

        torch.manual_seed(self.seeds["droppath"])
        g_Y_1_dropped = drop_path(
            g_Y_1, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_2 = X_2 + g(Y_1)
        Y_2 = X_2 + g_Y_1_dropped

        # free memory since X_2 is now not needed
        del X_2

        return Y_1, Y_2

    def backward_pass(
        self,
        Y_1,
        Y_2,
        dY_1,
        dY_2,
    ):
        """
        equation for activation recomputation:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention

        And we use pytorch native logic carefully to
        calculate gradients on F and G.
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            torch.manual_seed(self.seeds["FFN"])
            g_Y_1 = self.G(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = drop_path(
                g_Y_1, drop_prob=self.drop_path_rate, training=self.training
            )

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence we do not
        # need to record it in the computation graph.
        with torch.no_grad():
            # recomputing X_2 from the rev equation
            X_2 = Y_2 - g_Y_1

            # free memory since g_Y_1 is now not needed
            del g_Y_1

            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dX_1 in math.
            # reusing same variable to save memory
            dY_1 = dY_1 + Y_1.grad

            # free memory since Y_1.grad is now not needed
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.F(X_2)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = drop_path(
                f_X_2, drop_prob=self.drop_path_rate, training=self.training
            )

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():
            # recomputing X_1 from the rev equation
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            # the gradients for the previous block
            # note that it is called dY_2 but it in fact dX_2 in math.
            # reusing same variable to save memory
            dY_2 = dY_2 + X_2.grad

            # free memory since X_2.grad is now not needed
            X_2.grad = None

            X_2 = X_2.detach()

        # et voila~
        return X_1, X_2, dY_1, dY_2


class MLPSubblockV2C(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        const_patches_shape,
        token_pool_size,
        enable_amp=False,  # standard for ViTs
    ):
        super().__init__()

        self.patches_shape = (const_patches_shape[0]//token_pool_size, const_patches_shape[1]//token_pool_size)
        self.dim_c = dim_c

        self.norm = nn.LayerNorm(dim_v)

        self.fc1 = nn.Linear(dim_v, dim_c)
        self.act = nn.GELU()
        self.convtranspose = nn.ConvTranspose2d(in_channels=dim_c, out_channels=dim_c, 
                                                kernel_size=token_pool_size, stride=token_pool_size, groups=1)

        self.enable_amp = enable_amp

    def forward(self, x):
        # The reason for implementing autocast inside forward loop instead
        # in the main training logic is the implicit forward pass during
        # memory efficient gradient backpropagation. In backward pass, the
        # activations need to be recomputed, and if the forward has happened
        # with mixed precision, the recomputation must also be so. This cannot
        # be handled with the autocast setup in main training logic.
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x = self.norm(x)
            B, N, d_v = x.shape
            x = self.act(self.fc1(x))
            x = x.transpose(1,2).reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])

            x = self.convtranspose(x)
            x = x.reshape(B, self.dim_c, -1).transpose(1, 2)
            return x
            

class SRASubBlockC2V(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.

    Attention with Spatial-Reduction

    F : (N_c, d_c) --> (N_v, d_v)
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        num_heads,
        patches_shape,
        token_pool_size,
        sr_ratio,
        enable_amp=False,
        qk_scale=None,
        qkv_bias=True,

    ):
        super().__init__()
        
        assert dim_v % num_heads == 0
        assert patches_shape[0] % token_pool_size == 0
        assert patches_shape[1] % token_pool_size == 0

        self.norm = nn.LayerNorm(dim_c, eps=1e-6, elementwise_affine=True)

        self.dim_c, self.dim_v = dim_c, dim_v
        self.num_heads = num_heads
        self.patches_shape = patches_shape
        self.token_pool_size = token_pool_size
        self.N_v = (self.patches_shape[0]//token_pool_size) * (self.patches_shape[1]//token_pool_size)
        head_dim = dim_v // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.pool = nn.AvgPool2d(token_pool_size, stride=token_pool_size)

        self.q = nn.Linear(dim_c, dim_v, bias=qkv_bias)
        self.kv = nn.Linear(dim_c, dim_v * 2, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_v, dim_v)
        # self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio # R --> effective receptive field size patched together
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim_c, dim_c, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim_c)

        self.enable_amp = enable_amp

    def forward(self, x):
        # See MLP fwd pass for explanation.
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x = self.norm(x)

            B, N, d_c = x.shape

            x = x.transpose(1,2)
            x = x.reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])
            x = self.pool(x)
            x = x.reshape(B, self.N_v, self.dim_c)

            q = self.q(x).reshape(B, self.N_v, self.num_heads, self.dim_v // self.num_heads).permute(0, 2, 1, 3) # bs x num_heads x N x C/num_heads 

            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, self.dim_c, self.patches_shape[0]//self.token_pool_size, self.patches_shape[1]//self.token_pool_size)
                x_ = self.sr(x_).reshape(B, self.dim_c, -1).permute(0, 2, 1) # bs x (H/R*W/R) x C
                x_ = self.sr_norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.dim_v // self.num_heads).permute(2, 0, 3, 1, 4) # 2 x bs x num_heads x (H/R*W/R) x C/num_heads
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.dim_v // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1] # bs x num_heads x (H/R*W/R) x C/num_heads

            attn = (q @ k.transpose(-2, -1)) * self.scale # bs x num_heads x N x (H/R*W/R)
            attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, self.N_v, self.dim_v) # bs x num_heads x N x C/num_heads --> bs x N x C
            x = self.proj(x)
            # x = self.proj_drop(x)

            return x


class TokenMixerFBlockC2V(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        patches_shape,
        token_pool_size,
        enable_amp=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim_c, eps=1e-6, elementwise_affine=True)

        self.dim_c, self.dim_v = dim_c, dim_v
        self.patches_shape = patches_shape
        self.token_pool_size = token_pool_size
        self.N_v = (self.patches_shape[0]//token_pool_size) * (self.patches_shape[1]//token_pool_size)

        # self.pool = nn.AvgPool2d(token_pool_size, stride=token_pool_size)
        self.conv = nn.Conv2d(in_channels=dim_c, out_channels=dim_v, kernel_size=token_pool_size, stride=token_pool_size)
        self.token_mixer = nn.Linear(self.N_v, self.N_v)

        self.patches_shape = patches_shape
        self.enable_amp = enable_amp

    def forward(self, x):

        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            B, N, d_c = x.shape
            
            x = self.norm(x)    

            x = x.transpose(1,2)
            x = x.reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])
            x = self.conv(x).reshape(B, self.dim_v, self.N_v)

            x = torch.nn.functional.relu(x) # With this SMLP scores 76.16 ImageNet-100

            x = self.token_mixer(x).transpose(1, 2)

            return x
        

class VarStreamDownSamplingBlock(nn.Module):
    """
    Downsamples the var stream using avg pool
    """

    def __init__(self, input_patches_shape, kernel_size, dim_in, dim_out):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()

        assert input_patches_shape[0] % kernel_size == 0
        assert input_patches_shape[1] % kernel_size == 0


        self.input_patches_shape = input_patches_shape  
        self.dim_in, self.dim_out = dim_in, dim_out
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=kernel_size)    

    def forward(self, X_1, X_2):

        B, _, _ = X_1.shape
        X_1 = X_1.transpose(1, 2).reshape(B, self.dim_in, self.input_patches_shape[0], self.input_patches_shape[1])
        X_1 = self.conv(X_1)
        X_1 = X_1.reshape(B, self.dim_out, -1).transpose(1, 2)

        return X_1, X_2
    
class build_segmentation_model(nn.Module):
    
    def __init__(self, num_classes):
        super(build_segmentation_model, self).__init__()
        self.layers_outputs = [],
        self.num_classes = num_classes
        self.model = AsymmetricRevVit(
            block_type="swin-dw-mlp",
            const_dim=96,
            var_dim=[96, 192, 384, 768],
            sra_R=[8, 4, 2, 1],
            n_head=1,
            stages=[2, 2, 18, 2],
            drop_path_rate=0.1,
            patch_size=(4, 4),
            image_size=(224, 224),
            num_classes=num_classes,
            layers_outputs=self.layers_outputs,
        )

    def forward(self, x):
        output = self.model(x)
        self.layers_outputs.append(output)
        added_output = torch.zeros_like(output)
        for layer_output in self.layers_outputs:
            layer_output.requires_grad = True
            added_output += layer_output
        conv_layer = nn.Conv2d(
            in_channels=added_output.shape[1],
            out_channels=added_output.shape[1],
            kernel_size=1,
            stride=1,
        )
        conv_output = conv_layer(added_output.unsqueeze(-1).unsqueeze(-1))
        class_label_layer= nn.Linear(
            in_features=conv_output.shape[1],
            out_features=self.num_classes,
        )
        segmentation_output = class_label_layer(conv_output.squeeze(-1).squeeze(-1))
        return segmentation_output
        

def main():
    """
    This is a simple test to check if the recomputation is correct
    by computing gradients of the first learnable parameters twice --
    once with the custom backward and once with the vanilla backward.

    The difference should be ~zero.
    """
    layers_outputs = []
    model = build_segmentation_model(
        num_classes=100
    )
     
        
    

    # random input, instaintiate and fixing.
    # no need for GPU for unit test, runs fine on CPU.
    x = torch.rand((1, 3, 224, 224))
    model = model
    
    # model = model.to("cuda")
    # x = x.to("cuda")
    import time
    start_time = time.time()          

    # output of the model under reversible backward logic
    output = model(x)
    # loss is just the norm of the output
    loss = output.norm(dim=1).mean()
    print(loss.shape)

    # computatin gradients with reversible backward logic
    # using retain_graph=True to keep the computation graph.
    loss.backward(retain_graph=True)

    end_time = time.time()

    print(f"Batch time: {(end_time - start_time) * 1000:.3f} ms") 

    # gradient of the patchification layer under custom bwd logic
    rev_grad = model.patch_embed1.weight.grad.clone()

    # resetting the computation graph
    for param in model.parameters():
        param.grad = None

    # switching model mode to use vanilla backward logic
    model.no_custom_backward = True

    # computing forward with the same input and model.
    output = model(x)
    # same loss
    loss = output.norm(dim=1)

    # backward but with vanilla logic, does not need retain_graph=True
    loss.backward()

    # looking at the gradient of the patchification layer again
    vanilla_grad = model.patch_embed1.weight.grad.clone()

    # difference between the two gradients is small enough.
    # assert (rev_grad - vanilla_grad).abs().max() < 1e-6

    print(f"\nNumber of model parameters: {sum(p.numel() for p in model.parameters())}\n")

    try:
        from fvcore.nn import FlopCountAnalysis
        model.training = False
        input = torch.randn(1, 3, 224, 224)
        flops = FlopCountAnalysis(model, input)
        # input = torch.randn(1, 3136, 192)
        # part = model.layers[4].F
        # flops = FlopCountAnalysis(part, input)
        # print(f"\nNumber of model parameters: {sum(p.numel() for p in part.parameters())}\n")
        print(f"Total MACs Estimate (fvcore): {flops.total()}")
    except:
        print("FLOPs estimator failed")
        pass
    
    try:
        from utils import log_model_source
        log_model_source(model)
    except:
        print("No logs created")
        pass

if __name__ == "__main__":
    main()
 