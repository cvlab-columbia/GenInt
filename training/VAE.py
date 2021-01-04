from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--z_hidden', type=int, default=2,
                    help='number of neurons for mlp vae hidden layer')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.workers=10
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

train_path='/local/rcs/mcz/color_MNIST/train'
test_path='/local/rcs/mcz/color_MNIST/test'
composed_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

train_dataset = datasets.ImageFolder(
    train_path, composed_transforms
            )
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

test_dataset = datasets.ImageFolder(
    test_path, composed_transforms
            )
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True, sampler=None)

in_dim = 32**2 *3
emb_dim=5
hidden_dim=1000
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, args.z_hidden)
        self.fc22 = nn.Linear(hidden_dim, args.z_hidden)
        self.fc3 = nn.Linear(args.z_hidden+emb_dim, hidden_dim)
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
        if batch_idx % args.log_interval == 0:
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
                                      recon_batch.view(args.batch_size, 3, 32, 32)[:n]])
                save_image(comparison.cpu(),
                         '/local/rcs/mcz/2020Spring/VAE/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

from scipy.stats import norm
import numpy as np
vis_width=8
digit = 8
if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():

            # sample = torch.randn(64, 20).to(device)
            # sample = model.decode(sample).cpu()
            if args.z_hidden == 2:
                epsilon = norm.ppf(np.linspace(0, 1, vis_width + 2)[1:-1])
                sample = np.dstack(np.meshgrid(epsilon, -epsilon)).reshape(-1, 2)
                sample = torch.from_numpy(sample)
                sample = sample.float()
                sample = sample.to(device)
                target = torch.ones((vis_width ** 2,)).long() * digit
                target = target.to(device)
                sample = model.decode(sample, target).cpu()
            else:
                sample = torch.randn(vis_width ** 2, args.z_hidden).to(device)
                sample = model.decode(sample).cpu()

            save_image(sample.view(vis_width**2, 3, 32, 32),
                       '/local/rcs/mcz/2020Spring/VAE/sample_' + str(epoch) + '.png', nrow=vis_width)