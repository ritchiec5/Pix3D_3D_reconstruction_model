import scipy.io
from pyvox.models import Vox
from pyvox.writer import VoxWriter


# Load the .mat file into a dictionary
mat = scipy.io.loadmat('./pix3d/model/bed/IKEA_BEDDINGE/voxel.mat')

vox = Vox.from_dense(mat['voxel'])

VoxWriter('vox_file.vox', vox).write()