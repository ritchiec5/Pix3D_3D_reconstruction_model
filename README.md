# Pix3D_3D_reconstruction_model

## Goal - 3D Voxel Reconstruction from 2D image 
This project is a 3D reconstruction model that takes a 2D image as input and produces a 3D reconstruction of the object in the image.

## Getting Started
download dataset from https://github.com/xingyuansun/pix3d
download magicavoxel to visualize 3D voxel https://ephtracy.github.io/

```console
pip install -r /path/to/requirements.txt
```

```python
python train.py
# train and save checkpoint

python inference.py
# The model will output a 3D reconstruction of the object in the input image as a voxel file saved in the specified output folder.

python mat_to_vox.py
# Utility script to convert voxel.mat to view on magicavoxel
```

## Model Architecture
The model consists of three main parts: an encoder, a decoder, and a reconstruction model that combines the encoder and decoder.

The encoder is a series of four 2D convolutional layers, with the input image being passed through each layer in sequence. The output of the encoder is a latent code, which is a compressed representation of the input image.

The decoder is a series of three fully connected (fc) layers, which take the latent code as input and produce a reconstruction of the input image.

The reconstruction model combines the encoder and decoder, and is used to generate a reconstruction of the input image from the latent code produced by the encoder.

During training, the model is fed an input image, and the output is compared to the ground truth 3D voxel representation of the object in the image. The model is optimized to minimize the reconstruction error between the output and the ground truth.

## Acknowledgement
- Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling
- Chatgpt: Inspiring the architecture