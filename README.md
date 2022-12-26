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

- The encoder is a series of four 2D convolutional layers, with the input image being passed through each layer in sequence. The output of the encoder is a latent code, which is a compressed representation of the input image.

- The decoder is a series of three fully connected (fc) layers, which take the latent code as input and produce a reconstruction of the input image.

- The reconstruction model combines the encoder and decoder, and is used to generate a reconstruction of the input image from the latent code produced by the encoder.

During training, the model is fed an input image, and the output is compared to the ground truth 3D voxel representation of the object in the image. The model is optimized to minimize the reconstruction error between the output and the ground truth.

#### Advantages:
- The model uses convolutional layers, which are well-suited for processing images and can learn local patterns in the input data. This may be useful for reconstructing 3D objects from 2D images, as the model can learn to recognize features in the input images that correspond to different parts of the 3D object.
- The model includes an encoder-decoder structure, which allows it to compress the input data into a latent code, and then reconstruct the original data from that code. This can be useful for 3D reconstruction, as it allows the model to capture the underlying structure of the 3D object in a compact representation, and then reconstruct that object from that representation.

#### Disadvantages:

- The model only has a single fully-connected layer in the decoder, which may not be sufficient to fully capture the complexity of the 3D object being reconstructed.
- The model only uses 4 convolutional layers in the encoder, which may not be enough to fully capture the detail in the input images.
- The model only processes a single image at a time, so it may not be able to take advantage of multiple views of the same object, which could potentially improve the reconstruction.




## Acknowledgement
- Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling
- Chatgpt: Inspiring the architecture