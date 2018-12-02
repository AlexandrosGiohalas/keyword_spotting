from __future__ import print_function
import argparse
import torch
import random
import math
import pandas as pd
import os
import numpy as np
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


dim = 768
def load(myFeatures):
    num_lines = sum(1 for line in open(myFeatures))
    distances = open(myFeatures,'r')
    disList = list(distances)
    feature_dim = 768
    floatList = np.zeros((num_lines,feature_dim))
    for i in range(num_lines):
        tokens = disList[i].split(' ')
        for j in range(feature_dim):
            floatList[i][j] = float(tokens[j])
    ct = 0
    for i in range(num_lines):
        for j in range(feature_dim):
            if (math.isnan(floatList[i][j])):
                floatList[i][j] = 0.0
                ct += 1
    print(ct,num_lines*feature_dim)
    return torch.FloatTensor(floatList)

def getNames(path):
    names = list(open(path, 'r'))
    for i in range(len(names)):
        filename = names[i].split('\n')
        names[i] = filename[0]
    return names
data = '../feature_vectors/zah-master/distance.txt'
class MyDataset(torch.utils.data.Dataset):
    def change_data_files(self):
        global data
        data = '../results/lowDimDistance.txt'
        self.__init__()

    def __init__(self):
        self.data_files = load(data)
        self.root_dir = ('../../../Datasets/gw/words')
        self.image_names = getNames('../feature_vectors/zah-master/filenames.txt')


    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        return self.data_files[idx]

    def __len__(self):
        return len(self.data_files)

dset = MyDataset()
'''
def changeDataset():
    dset.change_data_files()
    global train_loader,test_loader
    train_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
'''
mu_size = 20

class VAE(nn.Module):
    def changeVaeMu(self,x):
        global mu_size
        mu_size = x
        self.__init__()

    def changeDim(self,x):
        global dim
        dim = x
        self.__init__()

    def __init__(self):
        super(VAE, self).__init__()
        '''
        self.fc1 = nn.Linear(dim, dim//2)
        self.fc21 = nn.Linear(dim//2, mu_size)
        self.fc22 = nn.Linear(dim//2, mu_size)
        self.fc3 = nn.Linear(mu_size, dim//2)
        self.fc4 = nn.Linear(dim//2, dim)
        '''
        self.fc1 = nn.Linear(768, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 768)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    dset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, dim), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(batch_idx, len(data), len(train_loader.dataset))
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
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print(len(test_loader.dataset))
    print('====> Test set loss: {:.4f}'.format(test_loss))


def main():
   
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        #test(epoch)
        '''
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png')
        '''
    torch.save(model, 'testDatasetModel.pth')

    print ()

if __name__ == '__main__':
    main()
