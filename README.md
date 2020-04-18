# Hyperspectral Reconstruction from RGB Images for Vein Visualization
We proposed a data-driven method to reconstruct hyperspectral images from RGB ones. The method is based on a residual learning approach that is effective in capturing the structure of the data manifold, and takes into account the spatial contextual information present in RGB images for the spectral reconstruction process. The proposed RGB-to-hyperspectral conversion method handles images taken in different illuminations, which is an important feature for practical applications. The proposed method is general and can support various applications. To show the value of the proposed conversion method, we designed and evaluated a vein visualization application. We collected one of the first hyperspectral datasets in this domain using a commercial hyperspectral camera; we make this dataset available for other researchers. We used this dataset to train our deep learning model and as ground truth for comparisons. Our experimental results show that the proposed method provides accurate vein visualization and localization results.

## Dataset Structure
- Download link -  https://nsl.cs.sfu.ca/projects/hyperspectral/hyperspectral_data/dataset.zip
- The dataset consists of paired 207 RGB images with their corresponding hypercubes in total.
- The hyperspectral images contain 34 bands in spectral range 820-920nm in matlab (`.mat`) format extracted from raw data.
- The total dataset is having information (images) from 13 participants. 10 participants' data is used for training and remaining 3 participants' data is used for testing/validation.
- Folder contents: The downloaded folder contains a sub-directory named `veins_t34bands`, further having dataset divided into `train_data`, `valid_data` and `test_data` folders. Each dataset folder is further divided into `mat` and `rgb` folders having hyperspectral and RGB images respectively.

## Source Code
### Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA CuDNN
- MATLAB

### Installation
- Clone this repo:
```bash
git clone https://github.com/nehasharma512/vein-visualization.git
cd vein-visualization
```
- Install [PyTorch](http://pytorch.org) and other dependencies.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

### Dataset preparation
- Move downloaded dataset folder to root (vein-visualization/dataset)
- Increase the training data using augmentation techniques (rotation and flipping). The matlab file `./train/augment_data.m` is used to perform augmentaion.
- The dataset is stored in HDF5 (`.h5`) file for training process. The matlab file `./train/generate_paired_rgb_nbands.m` is used to generate `train.h5` and `valid.h5` dataset files.

### Train/Test  
The training and testing codes are present in `./train/` and `./test/` folders respectively. The model architecture is present in `resblock.py` file.
- Train a model:
```bash
#!./train/train.py
python train.py
```
- The trained models will be stored in `./train/models/` folder with log files. 

- Test the model:
```bash
#!./test/evaluate_model.py
python evaluate_model.py
```
- The pre-trained models are present in `./test/models/`. The model can be evaluated on the testing dataset present in `./dataset/test_data/rgb/`. The test results will be saved to the folder: `./dataset/test_data/inference/`.

### Vein enhancement
- The reconstructed and ground truth hyperspectral images can be visualized in MATLAB using commands: `load(‘y.mat’);`,`imshow(rad(:,:,1),[]);`
- The reconstructed band can be enhanced using two enhancement techniques: Contrast Limited Adaptive Histogram Equalization (CLAHE) and Homomorphic Filtering.
- Enhancement can be produced using file `./vein_enhancement/enhance.m`.
