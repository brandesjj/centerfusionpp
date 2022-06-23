# CenterFusion++ <!-- omit in toc --> 
This repository contains the implementation of the Master's Thesis project 
**Camera-Radar Sensor Fusion using Deep Learning** from Johannes Kübel and Julian Brandes.

----------------------------------------
## Contents <!-- omit in toc --> 
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Models](#pretrained-models)
- [Training](#training)
- [Testing](#testing)
- [References](#references)
- [License](#license)
<!-- - [Results](#results) -->

## Introduction
This work is based on the frustum-proposal based radar and camera sensor fusion approach [CenterFusion](https://github.com/mrnabati/CenterFusion) proposed by Nabati et al.
We introduce two major changes to the existing network architecture:
1. **Early Fusion (EF)**  as a projection of the radar point cloud into the image plane. The projected radar point image features (default: depth, velocity components in x and z and RCS value) are then concatenated to the RGB image channels as a new input to the image-based backbone of the network architecture. EF introduces robustness against camera sensor failure and challenging environmental conditions (e.g. rain/night/fog).
2. **Learned Frustum Association (LFANet)**: The Second major change to the architecture regards the frustum-proposal based association between camera and radar point cloud. Instead of selecting the closest radar point to associate it to the detection obtained from the backbone & primary heads, we propose a network termed LFANet that outputs an artifical radar point **r*** representing all the radar points in the frustum. LFANet is trained to output the depth to the center of the bounding box associated with the radar point as well as the corresponding radial velocity. The outputs of LFANet are then used as the new channels in the heatmap introduced by Nabati et al.

We combine these two changes to obtain **CenterFusion++**. <br>
The following figure displays the modified network architecture on a high level:

![CenterFusion++ Overview](./figures/centerfusionpp_github_bright.png#gh-light-mode-only "CenterFusion++ Overview")
![CenterFusion++ Overview](./figures/centerfusionpp_github_dark.png#gh-dark-mode-only "CenterFusion++ Overview")

<!-- ## Results
- #### Overall results: <!-- omit in toc --> 

  <!-- | Dataset      |  NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
  |--------------|------|------|------|------|------|------|------|
  |nuScenes Val  | ? | ? |? | ? | ? | ? | ? | -->

<!-- - #### Per-class mAP: <!-- omit in toc --> 
  
  <!-- |  Dataset    |  Car | Truck | Bus | Trailer | Const. | Pedest. | Motor. | Bicycle | Traff. | Barrier |
  |-------------|------|-------|-----|---------|--------|---------|--------|---------|--------|---------|
  |nuScenes Val | ? | ? |? | ? | ? | ? | ? | ? | ? | ? | -->
<!-- - #### Qualitative results: omit in toc  -->

<!-- <p align="center"> <img src='figures/qualitative_results.jpg' align="center"> </p>  -->


## Installation

The code has been tested on Ubuntu 20.04 with Python 3.7.11, CUDA 11.3.1 and PyTorch 1.10.2. <br>
We used conda for the package management, the conda environment file is provided [here](experiments/centerfusionpp.yml). <br>
For installation, follow these steps:


1. Clone the repository with the `--recursive` option. We'll call the directory that you cloned thesisdlfusion into CFPP_ROOT:
    ```bash
    git clone --recursive https://github.com/brandesjj/centerfusionpp
    ```  
2. Install conda, following the instructions on their [website](https://www.anaconda.com/products/distribution). We use Anaconda, Installation can be done via:
   ```bash
   ./<Anaconda_file.sh>
   ```

2. Create a new conda environment (optional):
    ```bash
    conda env create -f <CFPP_ROOT>/experiments/centerfusionpp.yml   
    ```
   Restarg the shell and activate the conda environment
    ```bash
    conda activate centerfusionpp
    ```

3. Build the deformable convolution library:
    ```bash
    cd <CFPP_ROOT>/src/lib/model/networks/DCNv2
    ./make.sh
    ```
    **Note:** If the DCNv2 folder does not exist in the `networks` directory, it can be downloaded using this command:
    ```bash
    cd <CFPP_ROOT>/src/lib/model/networks
    git clone https://github.com/lbin/DCNv2/
    ```
    Note that this repository uses a slightly different DCNv2 repository than CenterFusion since [this](https://github.com/CharlesShang/DCNv2) caused some problems for our CUDA/pytorch version.

Additionally, the docker file to build a docker container with all the necessary packages is located [here](./experiments/Dockerfile).

## Dataset Preparation

CenterFusion++ was trained and validated using the [nuScenes](https://www.nuscenes.org/nuscenes) dataset only. Previous work (e.g. [CenterTrack](https://github.com/xingyizhou/CenterTrack)) uses other dataset (e.g. KITTI etc.) as well. This is not implemented within CenterFusion++. However, the original files that can be used to convert these datasets into the correct dataformat are not removed from this repository.

To download the dataset to your local machine follow these steps:

1. Download the nuScenes dataset from [nuScenes website](https://www.nuscenes.org/download).

2. Extract the downloaded files in the `<CFPP_ROOT>\data\nuscenes` directory. You should have the following directory structure after extraction:

    ~~~
    <CFPP_ROOT>
    `-- data
        `-- nuscenes
            |-- maps
            |-- samples
            |   |-- CAM_BACK
            |   |   | -- xxx.jpg
            |   |   ` -- ...
            |   |-- CAM_BACK_LEFT
            |   |-- CAM_BACK_RIGHT
            |   |-- CAM_FRONT
            |   |-- CAM_FRONT_LEFT
            |   |-- CAM_FRONT_RIGHT
            |   |-- RADAR_BACK_LEFT
            |   |   | -- xxx.pcd
            |   |   ` -- ...
            |   |-- RADAR_BACK_RIGHT
            |   |-- RADAR_FRON
            |   |-- RADAR_FRONT_LEFT
            |   `-- RADAR_FRONT_RIGHT
            |-- sweeps
            |-- v1.0-mini
            |-- v1.0-test
            `-- v1.0-trainval
        `-- annotations
          
    ~~~
   In this work, not all the data available from nuScenes is required. To save disk space, you can skip the LiDAR data in `/samples` and `/sweeps` and the `CAM_(..)` folders in `/sweeps`.<br>

Now you can create the necessary annotations. 
To create the annotations, run the [convert_nuScenes.py](./src/tools/convert_nuScenes.py) script to convert the nuScenes dataset to the required COCO format:
  ```bash
  cd <CFPP_ROOT>/src/tools
  python convert_nuScenes.py
  ```
The script contains several settings that can be used. They are explained in the first block of the code.


## Pretrained Models
The pre-trained models can be downloaded from the links given in the following table:
  | Model |  GPUs  | Backbone | Val NDS | Val mAP |
  |-|-|-|-|-|
  | [centerfusionpp.pth](https://drive.google.com/file/d/13h1K1MQslzJkcUbK398vaA3EvGNwL1f7/view?usp=sharing) | 2x NVIDIA A100 | EF | 0.4512 | 0.3209 |
  | [centerfusion_lfa.pth](https://drive.google.com/file/d/1MKvqf709wkn2Ejpul5V5my6-pJXM2Rg1/view?usp=sharing) | 2x NVIDIA A100 | CenterNet170 | 0.4407 | 0.3219 |
  | [earlyfusion.pth](https://drive.google.com/file/d/1isyxODucMWO-o7dFxxI5_e66jJraOY3_/view?usp=sharing) | 2x NVIDIA A100 | DLA34 | 0.3954 | 0.3159 |

**Notes**: 

- for the `CenterNet170` backbone, we refer to the [CenterFusion repository](https://github.com/mrnabati/CenterFusion#pretrained-models).

## Training

### Train on local machine

The scripts in `<CFPP_ROOT>/experiments/` can be used to train the network. There is one for the training on 1 GPU and another for the training on 2 GPUs.

  ```bash
  cd <CFPP_ROOT>
  bash experiments/train.sh
  ```

The `--train_split` parameter determines the training set, which could be `mini_train` or `train`. the `--load_model` parameter can be set to continue training from a pretrained model, or removed to start training from scratch. You can modify the parameters in the script as needed, or add more supported parameters from `<CFPP_ROOT>/src/lib/opts.py`.

The script creates a log folder in 
```
<CFPP_ROOT>/exp/ddd/<exp_id>/logs_<time_stamp>
```
where `<time_stamp>` is the time stamp and the default for `<exp_id>` is `centerfusionpp`. <br>
The log folder contains an event file for Tensorboard, a `log.txt` with a brief summary of the training process and a `opt.txt` file containing the specified options. <br>

## Testing

Download the pre-trained model into the `<CFPP_ROOT>/models` directory and use the `<CFPP_ROOT>/experiments/test.sh` script to run the evaluation:

  ```bash
  cd <CFPP_ROOT>
  bash experiments/test.sh
  ```

Make sure the `--load_model` parameter in the script provides the path to the downloaded pre-trained model. The `--val_split` parameter determines the validation set, which could be `mini_val`, `val` or `test`. You can modify the parameters in the script as needed, or add more supported parameters from `<CFPP_ROOT>/src/lib/opts.py`.

## References
The following works have been used by CenterFusion++.

  ~~~

  @INPROCEEDINGS{9423268,
  author={Nabati, Ramin and Qi, Hairong},
  booktitle={2021 IEEE Winter Conference on Applications of Computer Vision (WACV)}, 
  title={CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection}, 
  year={2021},
  volume={},
  number={},
  pages={1526-1535},
  doi={10.1109/WACV48630.2021.00157}}

  @inproceedings{zhou2019objects,
  title={Objects as Points},
  author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={arXiv preprint arXiv:1904.07850},
  year={2019}
  }

  @article{zhou2020tracking,
  title={Tracking Objects as Points},
  author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
  journal={ECCV},
  year={2020}
  }

  @inproceedings{nuscenes2019,
  title={{nuScenes}: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and Giancarlo Baldan and Oscar Beijbom},
  booktitle={CVPR},
  year={2020}
  }
  ~~~

---
---
## License

CenterFusion++ is based on [CenterFusion](https://github.com/mrnabati/CenterFusion) and is released under the MIT License. See [NOTICE](NOTICE) for license information on other libraries used in this project.
