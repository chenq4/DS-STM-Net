# DS&STm Net

Official Pytorch Code base for [DS&STM-Net: A Novel Hybrid Network of Feature Mutual Fusion for Medical Image Segmentation]

[project](https://github.com/chenq4/DS-STM-Net)


##Introduction

Medical image segmentation can effectively help doctors diagnose diseases faster in clinical medicine. In recent years, network models based on U-Net have performed well. With the research of Transformer and MLP in computer vision, it can better compensate for some shortcomings in UNet, enabling sufficient information exchange for input elements. In this paper, we are the first to propose the so-called DS&STM-Net, a novel hybrid network of feature mutual fusion via diagonal shifted MLP and Swin Transformer. DS&STM-Net is a parallel encoder to obtain the mutual features, one is the branch serialized by CNN and Diagonal Shifted MLP (DS-MLP), and the other is the Swin Transformer. In addition, a well-designed Feature Mutual Fusion (FMF) Module is proposed to fuse features from the two branches. Finally, we use DS-MLP as a decoder to better utilize its performance in long-distance dependencies. Furthermore, skip connections are used in different layers to provide multi-scale features for enhancing the final performance. We test DS&STM-Net on ISIC 2018 and BUSI datasets, which achieves better performance over state-of-the-art medical image segmentation architectures. 

## 1. Environment

- Please prepare an enviroment with Python 3.6 and PyTorch 1.7.1.

- Clone this repository:

```bash
git clone https://github.com/chenq4/DS-STM-Net
cd DS-STM-Net
```

To intall all the dependencies using pip:
```bash
pip install -r requirements.txt
conda activate DS-STM-Net
```

## 2. Datasets

1) ISIC 2018 - [Link](https://challenge.isic-archive.com/data/)
2) BUSI - [Link](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset)

## 3. Preprocess

View file `preprocess.ipynb`

## 4. Training and Validation

The results of training, testing, and visualization are all in `train_DS_STM_Net.ipynb` file

