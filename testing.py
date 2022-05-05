import torch
from matplotlib import pyplot as plt

# load models
generator = torch.load('MNIST Model/Generator_epoch_149.pth') #torch.load('KMNIST Model/Generator_epoch_149.pth')
print(generator.eval())
discriminator = torch.load('MNIST Model/Discriminator_epoch_149.pth') #torch.load('KMNIST Model/Discriminator_epoch_149.pth')
print(discriminator.eval())

# determine if gpu is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
device = torch.device(device)

# hyperparameter settings
batch_size = 64
n = 10

# do testing
print("\nDiscriminator results:")
total_avg = 0
for i in range(n):
    # generate data
    latent_space_samples = torch.randn(batch_size, 128).to(device)
    generated_samples = generator(latent_space_samples)

    # display for human
    cpu_samples = generated_samples.cpu().detach()
    for j in range(64):
        ax = plt.subplot(8, 8, j+1)
        plt.imshow(cpu_samples[j].reshape(28, 28), cmap="gray_r")
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # feed to discriminator and do testing
    confidence = str(discriminator(generated_samples))
    confidence = confidence.replace('tensor([', '').replace("], device='cuda:0', grad_fn=<SigmoidBackward0>)", '').replace('[', '').replace(']', '').replace('\n', '').replace('\t', '').replace(' ', '')
    print("\nPass "+str(i)+" Confidences:\n"+confidence)
    conf = confidence.split(',')
    avg = 0
    for c in conf:
        avg += float(c)

    avg = avg / batch_size
    total_avg += avg
    print("Average confidence: "+str(avg))

total_avg = total_avg / n
print("\nOverall average: "+str(total_avg))
