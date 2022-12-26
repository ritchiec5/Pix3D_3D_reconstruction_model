import torch
import torch.nn as nn
import pymesh
from PIL import Image
import numpy as np
from pyvox.models import Vox
from pyvox.writer import VoxWriter
import os

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Define the encoder layers
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        # Apply the encoder layers to the input image
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, batch_size):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define the decoder layers
        self.fc1 = nn.Linear(latent_dim * 4 * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Apply the decoder layers to the latent code
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)

        x = self.fc3(x)
        print(x.shape)
        x = torch.sigmoid(x)
        x = torch.round(x)

        x = x.view(-1, 128, 128, 128)
        return x

class ReconstructionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, batch_size):
        super(ReconstructionModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        
        # Initialize the encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim, batch_size)

    def setbatch_size(self, batch_size):
        self.batch_size = batch_size

    def forward(self, x):
        # Pass the input through the encoder to get the latent code
        latent_code = self.encoder(x)
        print(latent_code.shape)
        latent_code = latent_code.reshape(self.batch_size, -1)
        print(latent_code.shape)

        # Pass the latent code through the decoder to get the reconstruction
        reconstruction = self.decoder(latent_code)
        
        return reconstruction 


def save_voxel_file(file_path, voxel_data):
    # Convert the voxel data to a pymesh.VoxelGrid object
    voxel_grid = pymesh.VoxelGrid(voxel_data)

    # Save the voxel data to a .vox file
    pymesh.save_voxel_to_file(file_path, voxel_grid)


# Define the dimensions of the model
input_dim = 3
hidden_dim = 32
latent_dim = 16
output_dim = 128*128*128

# Initialize the reconstruction model
model = ReconstructionModel(input_dim, hidden_dim, latent_dim, output_dim, batch_size=1)

# Load the trained model weights from the specified checkpoint file
model.load_state_dict(torch.load("./checkpoints/best_weights.pth"))

# Set the model to evaluation mode
model.eval()

# Select an image from the test set
image = Image.open('./pix3d/img/bed/0017.png')

# Resize the image to the dimensions that the model was trained on
image = image.resize((64, 64))
    
# Convert the image to a numpy array and normalize it
image = np.array(image)
image = image / 255.0

# Add a batch dimension to the image
image = image.reshape(1, 3, 64, 64)

# Convert the image to a PyTorch tensor
image = torch.tensor(image, dtype=torch.float).to("cpu")

# Pass the image through the model to get the reconstruction
reconstruction = model(image)

# Remove the batch dimension from the reconstruction
reconstruction = reconstruction.squeeze(0)

# Convert the tensor to a numpy array and cast it to an integer type
reconstruction = reconstruction.detach().numpy().astype(int)

# Create a voxel object from the dense numpy array
vox = Vox.from_dense(reconstruction)

# Write the voxel object to a .vox file
VoxWriter('test.vox', vox).write()
