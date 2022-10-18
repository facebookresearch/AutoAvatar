#!/bin/bash

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install pytorch-lightning
conda install numpy numba matplotlib tqdm h5py ffmpeg
pip install smplx[all] pyyaml argparse open3d dotmap transforms3d==0.3.1
pip install chumpy scikit-image imageio-ffmpeg wandb hydra-core==1.0.6
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# For human_body_prior
pip install git+https://github.com/nghorbani/body_visualizer
pip install git+https://github.com/MPI-IS/configer.git
# pip install git+https://github.com/MPI-IS/mesh.git
