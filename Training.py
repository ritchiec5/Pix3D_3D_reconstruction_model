import json
from PIL import Image
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import torch
import torch.nn as nn
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
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 128, 128, 128)
        x = torch.round(x)
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


def preprocess_img(file):
    # Load and resize the image
    print(file)
    image = Image.open(file)
    image = image.resize((64, 64))
    
    # Convert the image to a numpy array and normalize it
    image = np.array(image)
    image = image / 255.0
    
    return image


def preprocess_voxel(file):
    mat_data = scipy.io.loadmat(file)
    voxel = np.array(mat_data['voxel'])
    voxel = voxel / np.max(voxel)
    return voxel

def preprocess_pix3d(json_file, img_dir, voxel_dir):
    # Load data from the Pix3D JSON file
    with open(json_file) as f:
        files = json.load(f)

    x = []
    y = []
    for json_obj in tqdm(files[:50]):
        img_file = json_obj["img"]
        voxel_file = json_obj["voxel"]
        
        # Preprocess the image and voxel data
        image = preprocess_img(img_dir + img_file)
        voxel = preprocess_voxel(voxel_dir + voxel_file)
        # Add the image and voxel data to the list
        x.append(image)
        y.append(voxel)
    return x, y


def train_reconstruction_model(model, X_train, y_train, X_test, y_test, num_epochs=50, batch_size=5, save_dir='checkpoints'):
    # Set the device to use for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert the data to tensors and move it to the device
    X_train = torch.tensor(X_train, dtype=torch.float).to(device)
    X_train = X_train.permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train, dtype=torch.float).to(device)


    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Create a directory to save the checkpoints
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize the best test loss to a high value
    best_test_loss = float('inf')

    # Loop over the number of epochs
    for epoch in tqdm(range(num_epochs)):
        # Shuffle the training data
        indices = torch.randperm(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Loop over the batches of data
        for i in range(0, len(X_train), batch_size):
            # Get the current batch of data
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Pass the data through the model
            y_pred = model.forward(X_batch)

            # Calculate the loss
            loss = criterion(y_pred, y_batch)

            # Zero the gradients
            optimizer.zero_grad()

            # Backpropagate the loss and update the model's parameters
            loss.backward()
            optimizer.step()

    # Convert the test data to tensors and move it to the device
    X_test = torch.tensor(X_test, dtype=torch.float).to(device)
    X_test = X_test.permute(0, 3, 1, 2)
    y_test = torch.tensor(y_test, dtype=torch.float).to(device)

    # Pass the test data through the model
    model.setbatch_size(15)
    y_pred = model.forward(X_test)

    # Calculate the test loss
    test_loss = criterion(y_pred, y_test)

    # Print the test loss
    print("Test loss: {:.4f}".format(test_loss))

    # If the test loss is the best so far, save the model's weights
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        filename = os.path.join(save_dir, 'best_weights.pth')
        torch.save(model.state_dict(), filename)


# Define the input, hidden, latent, and output dimensions
x, y = preprocess_pix3d('./pix3d/pix3d.json', './pix3d/', './pix3d/')

# Split the data into training and testing sets with a 70/30 ratio
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Convert list to numpy array
X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

# Define the dimensions of the model
input_dim = 3
hidden_dim = 32
latent_dim = 16
output_dim = 128*128*128

# Initialize the reconstruction model
model = ReconstructionModel(input_dim, hidden_dim, latent_dim, output_dim, 5)

# Train the model
train_reconstruction_model(model, X_train, y_train, X_test, y_test)

