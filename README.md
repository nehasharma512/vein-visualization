# Hyperspectral Reconstruction from RGB Images for Vein Visualization


## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA CuDNN
- MATLAB

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/nehasharma512/vein-visualization.git
cd vein-visualization
```

- Install [PyTorch](http://pytorch.org) and other dependencies.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

### train/test
- Download hyperspectral dataset:
  - https://nsl.cs.sfu.ca/projects/hyperspectral/hyperspectral_data/dataset.zip
  - move dataset folder to root (vein-visualization/dataset)
  
- Train a model:
  - run generate_paired_rgb_nbands.m file to generate the .h5 dataset
```bash
#!./train/train.py
python train.py
```
- Test the model:
```bash
#!./test/evaluate_model.py
python evaluate_model.py
```
- The test results will be saved to the folder: `./dataset/test_data/inference/`.


