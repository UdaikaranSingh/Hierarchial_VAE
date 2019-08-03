import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import torch.optim as optim
from torchvision.datasets import ImageFolder
import random
import torchvision.transforms as transforms

#######################################
# Importing and setting up networks 
#######################################

from model import Encoder, Decoder

cuda_available = torch.cuda.is_available()
encoded_size = 20
batch_size = 16

if cuda_available:
    print("Using GPU")
else:
    print("Using CPU")


if cuda_available:
	encoder = Encoder(batch_size, encoded_size).cuda()
	decoder = Decoder(batch_size, encoded_size).cuda()

	if os.path.exists("enocder_model.pth"):
		print("Loading in Model")
		encoder.load_state_dict(torch.load("encoder_model.pth"))
		decoder.load_state_dict(torch.load("decoder_model.pth"))
else:
	encoder = Encoder(batch_size, encoded_size)
	decoder = Decoder(batch_size, encoded_size)
	if os.path.exists("enocder_model.pth"):
		print("Loading in Model")
		encoder.load_state_dict(torch.load("encoder_model.pth"))
		decoder.load_state_dict(torch.load("decoder_model.pth"))

##################################################
# Definining Hyperparameters of Training Procedure
##################################################
random.seed(1)
learning_rate = 0.2e-4
beta1 = 0.5
beta2 = 0.999
num_epochs = 10
epsilon = 1e-8

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
	lr=learning_rate, 
	betas=(beta1, beta2))

transform = transforms.Compose([
    transforms.ToTensor()
])
dataset_path = os.path.dirname(os.getcwd()) + "/datasets/celeba"
training_data = ImageFolder(dataset_path, transform = transform)
data_loader = torch.utils.data.DataLoader(training_data,
                                          batch_size= batch_size,
                                          shuffle=True,
                                          num_workers= 1)

##################################################
# Training Procedure
##################################################

loss_tracker = []

for epoch in range(num_epochs):

	print("epoch: " + str(epoch + 1))

	for images, steps in data_loader:

		if (images.shape[0] != batch_size):
			break

		encoder.zero_grad()
		decoder.zero_grad()
		encoded_vector = encoder(images)
		decoded_images = decoder(encoded_vector)

		#calculate loss: loss = ...
		loss.backward()
		optimizer.step()

	#saving models:
	path = "decoder_mdoel.pth"
    torch.save(decoder.state_dict(), path)
    path = "encoder_model.pth"
    torch.save(encoder.state_dict(), path)



