"""
Grady Landers
Advanced AI Project
Custom GAN model testing
"""

import torch
from math import sqrt
from matplotlib import pyplot as plt


# determine if gpu is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
device = torch.device(device)

# load models
generator = torch.load('MNIST Model/Generator_epoch_149.pth', map_location=device) #torch.load('KMNIST Model/Generator_epoch_149.pth', map_location=device)
print(generator.eval())
discriminator = torch.load('MNIST Model/Discriminator_epoch_149.pth', map_location=device) #torch.load('KMNIST Model/Discriminator_epoch_149.pth', map_location=device)
print(discriminator.eval())

# hyperparameter settings
n = 10
batch_size = 64
sqr = sqrt(batch_size)
if sqr / int(sqr) != 1:
    sqr += 1
sqr = int(sqr)

# do testing
print("\nDiscriminator results:")
total_avg = 0
for i in range(n):
    # generate images
    latent_space_samples = torch.randn(batch_size, 128).to(device)
    generated_samples = generator(latent_space_samples)

    # display for human
    cpu_samples = generated_samples.cpu().detach()
    for j in range(batch_size):
        ax = plt.subplot(sqr, sqr, j+1)
        plt.imshow(cpu_samples[j].reshape(28, 28), cmap="gray_r")
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # feed to discriminator and do testing
    confidence = str(discriminator(generated_samples))
    confidence = confidence.replace('tensor([', '').replace("], device='cuda:0', grad_fn=<SigmoidBackward0>)", '').replace("], grad_fn=<SigmoidBackward0>)", '').replace('[', '').replace(']', '').replace('\n', '').replace('\t', '').replace(' ', '')
    print("\nBatch "+str(i+1)+" Confidences:\n"+confidence)
    conf = confidence.split(',')
    avg = 0
    for c in conf:
        avg += float(c)

    avg = avg / batch_size
    total_avg += avg
    print("Average confidence: "+str(avg))

total_avg = total_avg / n
print("\nOverall average: "+str(total_avg))
