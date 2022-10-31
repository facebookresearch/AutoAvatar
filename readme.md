<p align="center">
  <p align="center">
    <h1 align="center">AutoAvatar: Autoregressive Neural Fields for Dynamic Avatar Modeling</h1>
  </p>
  <p align="center" style="font-size:16px">
    <a target="_blank" href="https://zqbai-jeremy.github.io"><strong>Ziqian Bai</strong></a>
    ·
    <a target="_blank" href="https://scholar.google.ch/citations?user=oLi7xJ0AAAAJ&hl=en"><strong>Timur Bagautdinov</strong></a>
    ·
    <a target="_blank" href="https://scholar.google.com/citations?user=Wx62iOsAAAAJ&hl=en"><strong>Javier Romero</strong></a>
    .
    <a target="_blank" href="https://zollhoefer.com/"><strong>Michael Zollhöfer</strong></a>
    ·
    <a target="_blank" href="https://www.cs.sfu.ca/~pingtan/"><strong>Ping Tan</strong></a>
    ·
    <a target="_blank" href="http://www-scf.usc.edu/~saitos/"><strong>Shunsuke Saito</strong></a>
  </p>
  <h2 align="center">ECCV 2022</h2>

  <div align="center"></div> <img src="./assets/teaser.gif" alt="Logo" width="100%">

  <p>
  AutoAvatar is an autoregressive approach for modeling dynamically deforming human bodies directly from raw scans without the need of precise surface registration.
  </p>

  <p align="center">
    <!-- <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a> -->
    <br>
    <a href='https://arxiv.org/abs/2203.13817'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <a href='https://zqbai-jeremy.github.io/autoavatar/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/AutoAvatar-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=orange' alt='Project Page'>
    <a href="https://zqbai-jeremy.github.io/autoavatar/static/images/video_arxiv.mp4"><img alt="video views" src="https://img.shields.io/badge/Video-mp4-blue?style=for-the-badge&logo=youtube&logoColor=white"/></a>
  </p>

</p>


## Data Preparation of DFaust

- Create "DFaust" folder under "\<workspace_folder\>".

```bash
cd <workspace_folder>
mkdir DFaust
```

- Download SMPL+H parameters of DFaust from [AMASS dataset](https://amass.is.tue.mpg.de/index.html) to "\<workspace_folder\>/DFaust". Unzip to get the "DFaust_67" folder.

- Download Dfaust scan data from [link](https://dfaust.is.tue.mpg.de/index.html). Here, we take subject 50002 as an example in the following steps. Unzip data to "\<workspace_folder\>/DFaust/scans/50002".

- Download SMPL model from [link](https://smpl.is.tue.mpg.de/). Download SMPL meta data from [link](https://drive.google.com/drive/folders/1ZhS_0FFJ38Mj9pZrkr5HUTurCaofjLSk?usp=sharing). Move SMPL related files "basicmodel_m_lbs_10_207_0_v1.0.0.pkl", "basicModel_f_lbs_10_207_0_v1.0.0.pkl", "uv_info.npz", and "smpl_resample_idxs.npz" into "\<workspace_folder\>/SMPL".

- clone this repo to "\<workspace_folder\>".

```bash
cd <workspace_folder>
git clone https://github.com/facebookresearch/AutoAvatar.git
```

- Now we should have the following folder structure:

```bash
    \<workspace_folder\>
    ├── DFaust
    │   ├── DFaust_67
    │   │   └── 50002
    │   │       └── *.npz
    │   └── scans
    │       └── 50002
    │           └── \<sequences_folders\>
    ├── SMPL
    |   └── \<SMPL_related_files\>
    └── AutoAvatar
```


## Environment Setup

- Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Then run the setup script.

```bash
cd AutoAvatar
conda create -n AutoAvatar python=3.8
conda activate AutoAvatar
bash setup.sh
```

- Create "external" folder and install [human_body_prior](https://github.com/nghorbani/human_body_prior) for DFaust data preprocess.

```bash
mkdir external
cd external
git clone https://github.com/nghorbani/human_body_prior.git
cd human_body_prior
python setup.py develop
```


## Data Preprocess

- Run "DFaust_generate.py" to preprocess data. Note that this may take a long time due to the mesh simplification (the open3d API mesh_o3d.simplify_quadric_decimation() in simplify_scans())! Mesh simplification is to speed up data loading during training.

```bash
cd AutoAvatar
export PYTHONPATH=<workspace_folder>/AutoAvatar
python data/DFaust_generate.py --ws_dir <workspace_folder>
```


## Train

- Run "implicit_train_dfaust.py" to train the model.

```bash
cd AutoAvatar
export PYTHONPATH=<workspace_folder>/AutoAvatar
python exps/PosedDecKNN_dPoses_dHs/implicit_train_dfaust.py --ws_dir <workspace_folder> --configs_path configs/PosedDecKNN_dPoses_dHs/AutoRegr.yaml --configs_path_rollout configs/PosedDecKNN_dPoses_dHs/AutoRegr_Rollout2.yaml
```


## Test

- Run "implicit_eval_dfaust.py" to test the model.

```bash
cd AutoAvatar
export PYTHONPATH=<workspace_folder>/AutoAvatar
python exps/PosedDecKNN_dPoses_dHs/implicit_eval_dfaust.py --ws_dir <workspace_folder> --ckpt_dir <checkpoint_folder>
```

## Pretrained Model

- Download pretrained model for DFaust subject 50002 from [link](https://drive.google.com/file/d/1Z3HmbXFgpTzE55Sxu4T2YkEMPA2j0Iuh/view?usp=sharing).

## Publication
If you find our code or paper useful, please consider citing:
```bibtex
@inproceedings{bai2022autoavatar,
  title={AutoAvatar: Autoregressive Neural Fields for Dynamic Avatar Modeling},
  author={Bai, Ziqian and Bagautdinov, Timur and Romero, Javier and Zollh{\"o}fer, Michael and Tan, Ping and Saito, Shunsuke},
  booktitle={European conference on computer vision},
  year={2022},
}
```

## License
[CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 
