import torch
from torch import nn
import sys
import numpy as np
from typing import List, Tuple

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    output = x.div(keep_prob) * mask
    return output

class MLPSubblockV2C(nn.Module):
    def __init__(
        self,
        dim_c: int,
        dim_v: int,
        const_patches_shape: Tuple[int, int],
        token_pool_size: int,
        enable_amp: bool = False
    ):
        super().__init__()
        self.patches_shape_0 = const_patches_shape[0] // token_pool_size
        self.patches_shape_1 = const_patches_shape[1] // token_pool_size
        self.dim_c = dim_c

        self.norm = nn.LayerNorm(dim_v)
        self.fc1 = nn.Linear(dim_v, dim_c)
        self.act = nn.GELU()
        self.convtranspose = nn.ConvTranspose2d(
            in_channels=dim_c,
            out_channels=dim_c,
            kernel_size=token_pool_size,
            stride=token_pool_size,
            groups=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        B, N, d_v = x.shape
        x = self.act(self.fc1(x))
        x = x.transpose(1, 2).reshape(B, self.dim_c, self.patches_shape_0, self.patches_shape_1)
        x = self.convtranspose(x)
        x = x.reshape(B, self.dim_c, -1).transpose(1, 2)
        return x

class TokenMixerFBlockC2V(nn.Module):
    def __init__(
        self,
        dim_c: int,
        dim_v: int,
        patches_shape: Tuple[int, int],
        token_pool_size: int,
        enable_amp: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim_c, eps=1e-6, elementwise_affine=True)

        self.dim_c = dim_c
        self.dim_v = dim_v
        self.patches_shape_0 = patches_shape[0]
        self.patches_shape_1 = patches_shape[1]
        self.token_pool_size = token_pool_size
        self.N_v = (patches_shape[0] // token_pool_size) * (patches_shape[1] // token_pool_size)

        self.conv = nn.Conv2d(in_channels=dim_c, out_channels=dim_v, kernel_size=token_pool_size, stride=token_pool_size)
        self.token_mixer = nn.Linear(self.N_v, self.N_v)

        self.enable_amp = enable_amp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d_c = x.shape
        
        x = self.norm(x)    

        x = x.transpose(1, 2)
        x = x.reshape(B, self.dim_c, self.patches_shape_0, self.patches_shape_1)
        x = self.conv(x).reshape(B, self.dim_v, self.N_v)

        x = self.token_mixer(x).transpose(1, 2)

        return x

class VarStreamDownSamplingBlock(nn.Module):
    def __init__(self, input_patches_shape: Tuple[int, int], kernel_size: int, dim_in: int, dim_out: int):
        super().__init__()

        self.input_patches_shape_0 = input_patches_shape[0]
        self.input_patches_shape_1 = input_patches_shape[1]
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=kernel_size)    

    def forward(self, X_1: torch.Tensor, X_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, _ = X_1.shape
        X_1 = X_1.transpose(1, 2).reshape(B, self.dim_in, self.input_patches_shape_0, self.input_patches_shape_1)
        X_1 = self.conv(X_1)
        X_1 = X_1.reshape(B, self.dim_out, -1).transpose(1, 2)

        return X_1, X_2

class SimpleReversibleBlock(nn.Module):
    def __init__(self, dim_c: int, dim_v: int, num_heads: int, enable_amp: bool, drop_path: float, 
                 token_map_pool_size: int, sr_ratio: int, const_patches_shape: Tuple[int, int], block_id: int):
        super().__init__()
        
        self.drop_path_rate = drop_path
        self.block_id = block_id

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
            enable_amp=False,
        )

    def forward(self, X_1: torch.Tensor, X_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_X_2 = self.F(X_2)
        f_X_2_dropped = drop_path(f_X_2, drop_prob=self.drop_path_rate, training=self.training)
        Y_1 = X_1 + f_X_2_dropped

        g_Y_1 = self.G(Y_1)
        g_Y_1_dropped = drop_path(g_Y_1, drop_prob=self.drop_path_rate, training=self.training)
        Y_2 = X_2 + g_Y_1_dropped

        return Y_1, Y_2

class AsymmetricRevVitTorchScript(nn.Module):
    def __init__(
        self,
        const_dim: int = 768,
        var_dim: List[int] = [64, 128, 320, 512],
        sra_R: List[int] = [8, 4, 2, 1],
        n_head: int = 8,
        stages: List[int] = [3, 3, 6, 3],
        drop_path_rate: float = 0,
        patch_size: Tuple[int, int] = (2, 2),
        image_size: Tuple[int, int] = (32, 32),
        num_classes: int = 10,
        enable_amp: bool = False,
    ):
        super().__init__()

        self.const_dim = const_dim
        self.n_head = n_head
        self.patch_size_0 = patch_size[0]
        self.patch_size_1 = patch_size[1]

        self.const_num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        
        const_patches_shape_0 = image_size[0] // patch_size[0]
        const_patches_shape_1 = image_size[1] // patch_size[1]

        total_blocks = sum(stages)
        
        # Create drop path rates
        dpr: List[float] = []
        for i in range(total_blocks):
            rate = float(i) * drop_path_rate / float(total_blocks - 1)
            dpr.append(rate)

        # Create layers
        layers = nn.ModuleList()
        block_idx = 0
        
        for stage_idx in range(len(stages)):
            for _ in range(stages[stage_idx]):
                layers.append(
                    SimpleReversibleBlock(
                        dim_c=const_dim,
                        dim_v=var_dim[stage_idx],
                        num_heads=n_head,
                        enable_amp=enable_amp,
                        sr_ratio=sra_R[stage_idx],
                        token_map_pool_size=2**stage_idx,
                        drop_path=dpr[block_idx],
                        const_patches_shape=(const_patches_shape_0, const_patches_shape_1),
                        block_id=block_idx
                    )
                )
                block_idx += 1
            
            # Add downsampling block between stages (except last stage)
            if stage_idx < len(stages) - 1:
                input_shape_0 = const_patches_shape_0 // (2**stage_idx)
                input_shape_1 = const_patches_shape_1 // (2**stage_idx)
                layers.append(
                    VarStreamDownSamplingBlock(
                        input_patches_shape=(input_shape_0, input_shape_1),
                        kernel_size=2, 
                        dim_in=var_dim[stage_idx], 
                        dim_out=var_dim[stage_idx+1]
                    )
                )

        self.layers = layers

        # Patch embedding layers
        self.patch_embed2 = nn.Conv2d(3, const_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embeddings2 = nn.Parameter(torch.zeros(1, self.const_num_patches, const_dim))

        self.patch_embed1 = nn.Conv2d(3, var_dim[0], kernel_size=patch_size, stride=patch_size)
        self.pos_embeddings1 = nn.Parameter(torch.zeros(1, self.const_num_patches, var_dim[0]))

        # Classification head
        self.head = nn.Linear(const_dim, num_classes, bias=True)
        self.norm = nn.LayerNorm(const_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x2 = self.patch_embed2(x).flatten(2).transpose(1, 2)
        x2 = x2 + self.pos_embeddings2

        x1 = self.patch_embed1(x).flatten(2).transpose(1, 2)
        x1 = x1 + self.pos_embeddings1

        # Process through layers
        for layer in self.layers:
            x1, x2 = layer(x1, x2)

        # Classification
        pred = x2.mean(1)
        pred = self.norm(pred)
        pred = self.head(pred)

        return pred

def create_torchscript_model():
    model = AsymmetricRevVitTorchScript(
        const_dim=192,
        var_dim=[64, 128, 320, 512],
        sra_R=[8, 4, 2, 1],
        n_head=8,
        stages=[1, 1, 2, 1],  # Reduced for faster compilation
        drop_path_rate=0.1,
        patch_size=(4, 4),
        image_size=(32, 32),  # CIFAR-10 size
        num_classes=10,
    )
    return model

def convert_to_torchscript():
    # Create model
    model = create_torchscript_model()
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, 32, 32)
    
    # Method 1: torch.jit.trace (recommended for this model)
    try:
        traced_model = torch.jit.trace(model, example_input)
        print("✓ Tracing successful!")
        
        # Test the traced model
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = traced_model(example_input)
            diff = torch.abs(original_output - traced_output).max()
            print(f"Max difference between original and traced: {diff}")
        
        # Save the traced model
        torch.jit.save(traced_model, "asymmetric_revvit_traced.pt")
        print("✓ Traced model saved!")
        
        return traced_model
        
    except Exception as e:
        print(f"✗ Tracing failed: {e}")
    
    # Method 2: torch.jit.script (if tracing fails)
    try:
        scripted_model = torch.jit.script(model)
        print("✓ Scripting successful!")
        
        # Test the scripted model
        with torch.no_grad():
            original_output = model(example_input)
            scripted_output = scripted_model(example_input)
            diff = torch.abs(original_output - scripted_output).max()
            print(f"Max difference between original and scripted: {diff}")
        
        # Save the scripted model
        torch.jit.save(scripted_model, "asymmetric_revvit_scripted.pt")
        print("✓ Scripted model saved!")
        
        return scripted_model
        
    except Exception as e:
        print(f"✗ Scripting failed: {e}")
        return None

if __name__ == "__main__":
    # Convert model to TorchScript
    torchscript_model = convert_to_torchscript()
    
    if torchscript_model is not None:
        # Load and test the saved model
        loaded_model = torch.jit.load("asymmetric_revvit_traced.pt")
        
        # Test with random input
        test_input = torch.randn(2, 3, 32, 32)
        output = loaded_model(test_input)
        print(f"Output shape: {output.shape}")
        print("✓ Model conversion and loading successful!")
    else:
        print("✗ Model conversion failed!")