from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import trange

import os

log_interval = 10
epochs = 50
batch_size = 128
cuda = torch.cuda.is_available()
workers=10
torch.manual_seed(1)

device = torch.device("cuda")

train_path='/proj/vondrick/datasets/color_MNIST/train'
test_path= '/proj/vondrick/datasets/color_MNIST/test'

composed_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

train_dataset = datasets.ImageFolder(
    train_path, composed_transforms
            )
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, sampler=None)

test_dataset = datasets.ImageFolder(
    test_path, composed_transforms
            )
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=None,
        num_workers=workers, pin_memory=True, sampler=None)

in_dim = 32**2 *3
emb_dim=5
hidden_dim=1000
z_hidden = 2

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_hidden)
        self.fc22 = nn.Linear(hidden_dim, z_hidden)
        self.fc3 = nn.Linear(z_hidden+emb_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, in_dim)

        self.embed = nn.Embedding(10, emb_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, emb):
        lab_emb = self.embed(emb)
        h3 = F.relu(self.fc3(torch.cat((z,lab_emb), -1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, emb):
        mu, logvar = self.encode(x.view(-1, in_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, emb), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, in_dim), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("target", target)
        # print(data.shape)
        # exit(0)

        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, target)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            recon_batch, mu, logvar = model(data, target)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 3, 32, 32)[:n]])
                save_image(comparison.cpu(),
                         './' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



from scipy.stats import norm
import numpy as np

vis_width=77

    
for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)


print("Generate cVAE Intervened Training Set")

color_mnist_intervene_path = '/proj/vondrick/datasets/color_MNIST/intervene_train/'

for i in range(10):
    os.makedirs(color_mnist_intervene_path+str(i), exist_ok=True)


for digit in range(10):
    with torch.no_grad():
        epsilon = norm.ppf(np.linspace(0, 1, vis_width + 2)[1:-1])
        sample = np.dstack(np.meshgrid(epsilon, -epsilon)).reshape(-1, 2)
        sample = torch.from_numpy(sample)
        sample = sample.float()
        sample = sample.to(device)
        target = torch.ones((vis_width ** 2,)).long() * digit
        target = target.to(device)
        sample = model.decode(sample, target).cpu()

        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        sample = sample.view(vis_width**2, 3, 32, 32)

        for i in range(sample.size(0)):
            torchvision.utils.save_image(sample[i, :, :, :], 
                                             color_mnist_intervene_path+str(digit)+'/{}.png'.format(i), 
                                             normalize=True)