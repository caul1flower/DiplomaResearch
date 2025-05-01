# DiplomaResearch

## SGTNet: Structure Guided Transformer Network for Depth Super Reesolution

SGTNet is a transformer-based model for depth image super-resolution that leverages guidance from high-resolution RGB images to enhance low-resolution depth maps. This repository contains the official implementation of SGTNet along with training and testing code.

## Installation

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

You can download the datasets used in this project from the following links:

- **Training Dataset**: [NYU-v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- **Testing Datasets**: 
  - [Middlebury & Lu](https://web.cecs.pdx.edu/~fliu/project/depth-enhance/)


Or you may also download them from [GoogleDrive](https://drive.google.com/drive/folders/1lkjqi2LH9eKEX7z9JWw3qqYvdQrc403Y?usp=sharing)

Please create a `datasets` folder with the following structure and place files in it:

```
datasets/
├── nyu_data/
├── Middlebury/
└── Lu/
```

## Pretrained Models

You can download pretrained models from the link: [pretrained models](https://drive.google.com/drive/folders/1Zt4aeZPSm3Zy_PIExSRMUc3EXVjMyQaM?usp=sharing)

Please create a `checkpoints` folder and place the downloaded model files inside it.

## Usage

### Training


To train SGTNet on the NYU-v2 dataset:

```bash
python train.py --scale 4 --batch_size 8 --epochs 100
```
(Example of usage, you may change the arguments)

### Tesing

To test a trained model:

```bash
python test.py --scale 4 --pretrained_path ./checkpoints/x16_256.pth --dataset_dir ./datasets/Lu
```
(Example of usage, you may change the arguments)

Be aware, that the scale argument and the scale in the checkpoint name should match. Also, for testing, img_size should be the same as the img_size of the trained model (in our case, 256), but it doesn't affect the actual size of the output image.

### Results

For training, testing scripts and inference, a `results` folder will be created.



## Acknowledgments

- This implementation uses parts of the SGNet and Swin2SR architectures
