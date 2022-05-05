"""
Grady Landers
Advanced AI Project
Custom GAN Training modified from https://towardsdatascience.com/building-a-gan-with-pytorch-237b4b07ca9a

Datasets:
MNIST - http://yann.lecun.com/exdb/mnist/
KMNIST - https://github.com/rois-codh/kmnist
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from model import Discriminator, Generator


# determine if gpu is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
device = torch.device(device)

# hyperparameter settings
epochs = 150
lr = 2e-4
batch_size = 64
loss = nn.BCELoss()

# Model
G = Generator().to(device)
D = Discriminator().to(device)

"""
Image transformation and dataloader creation
Note that we are training generation and not classification, and hence
only the train_loader is loaded
"""
# Transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# Load data
train_set = datasets.MNIST('mnist/', train=True, download=True, transform=transform) #datasets.KMNIST('kmnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

trainingOutput = open("mnist_training", 'w') #open("kmnist_training", 'w')

for epoch in range(epochs):
    for idx, (imgs, _) in enumerate(train_loader):
        idx += 1

        # Training the discriminator
        # Real inputs are actual images of the MNIST dataset
        # Fake inputs are from the generator
        # Real inputs should be classified as 1 and fake as 0
        real_inputs = imgs.to(device)
        real_outputs = D(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)

        D_loss = loss(outputs, targets)
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Training the generator
        # For generator, goal is to make the discriminator believe everything is 1
        noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
        noise = noise.to(device)

        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
        G_loss = loss(fake_outputs, fake_targets)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if idx % 100 == 0 or idx == len(train_loader):
            print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(epoch, idx, D_loss.item(), G_loss.item()))
            trainingOutput.write('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}\n'.format(epoch, idx, D_loss.item(), G_loss.item()))

    if (epoch+1) % 10 == 0:
        torch.save(G, 'MNIST Model/Generator_epoch_{}.pth'.format(epoch)) #torch.save(G, 'KMNIST Model/Generator_epoch_{}.pth'.format(epoch))
        torch.save(D, 'MNIST Model/Discriminator_epoch_{}.pth'.format(epoch)) #torch.save(D, 'KMNIST Model/Discriminator_epoch_{}.pth'.format(epoch))
        print('Model saved.')

trainingOutput.close()
