# 2022-PCT-Lightning
This is a Pytorch Lightning implementation of PCT: Point Cloud Transformer.

Paper link: https://arxiv.org/pdf/2012.09688.pdf

### Requirements

This repo is tested with:

#### Software
Ubuntu 20.04 LTS

CUDA 11.6

python 3.9

cudatoolkit 11.3.1

pytorch 1.11.0

pytorch-lightning 1.6.3

#### Hardware
Intel i7-12700

Nvidia GTX 1080

### Install

1. Create a conda environment for this repo.
```shell script
conda create -n pct_lightning python=3.9

conda activate pct_lightning
```

2. Install the requirments.txt
```shell script
pip install requirements.txt
```

3. Download dataset & pre-trained model

You can download dataset and pre-trained model here: [Google Drive](https://drive.google.com/drive/folders/1nJCQBvBV0Xz9AZUYzQ0CvY0lSHdK1uVw?usp=sharing). 

Put `modelnet40_ply_hdf5_2048` under `{Repo_dir}/data/`

Put `model.ckpt` anywhere you want, but please change `configs/test.yaml` line 32 `ckpt_path` to `/your/path/to/model.ckpt`


### Quick Test
Run command:

```shell script
python test.py
```

Due to the hardware limitation, the pre-trained model has accuracy of 83.8% on the [ModelNet40](http://modelnet.cs.princeton.edu/) validation dataset.

### Test with Visualization

1. Install cv2

```shell script
pip install cv2
```

2. Compile
```shell script

cd src/utils

g++ -std=c++11 render_balls_so.cpp -o render_balls_so.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

```

3. Configure

Open `./configs/model/pct.yaml`. Set `visual_pc` to `true`. Set `visual_path` to the path you want the output image to save. (Thousands of images)

4. Run
```shell script
python test.py
```

### Train

```shell script
python train.py
```

You can change the training parameters in `./configs/train.yaml`

### Acknowledgement

Please cite this paper if you found this repo useful for your research.
```latex
@misc{guo2020pct,
      title={PCT: Point Cloud Transformer}, 
      author={Meng-Hao Guo and Jun-Xiong Cai and Zheng-Ning Liu and Tai-Jiang Mu and Ralph R. Martin and Shi-Min Hu},
      year={2020},
      eprint={2012.09688},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

This repo borrows tons of codes from [PCT_Pytorch](https://github.com/Strawberry-Eat-Mango/PCT_Pytorch)

The visualzation tool comes from [Shape2Motion](https://github.com/wangxiaogang866/Shape2Motion)

This repo uses this great template [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)