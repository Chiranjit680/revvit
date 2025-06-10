# create data directory
mkdir -p data/ade
cd data/ade

# download ADE20K (validation + training)
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

# unzip
unzip ADEChallengeData2016.zip

# folder structure becomes: data/ade/ADEChallengeData2016/images/
cd ../../
