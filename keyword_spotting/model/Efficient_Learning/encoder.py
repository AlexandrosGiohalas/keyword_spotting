from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from customDataset import MyDataset

dim = 30*672

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(dim, 3*dim//4)
        self.fc11 = nn.Linear(3*dim//4,dim//2)
        self.fc12 = nn.Linear(dim//2,300)
        self.fc21 = nn.Linear(300, 25)
        self.fc22 = nn.Linear(300, 25)

    def encode(self, x):
        h01 = F.relu(self.fc1(x))
        h02 = F.relu(self.fc11(h01))
        h1 = F.relu(self.fc12(h02))
        return F.relu(self.fc21(h1)), self.fc22(h1)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 30*672))
        return mu

#    def update(self):
#        self.fc22 = nn.Linear(dim//2, 25)

def getWordsDescriptors(filename,dimension):
    wdescFile = open(filename,'r')
    wdescList = list(wdescFile)
    descriptors = np.zeros((2501,dimension))
    for i in range(len(wdescList)):
        des = list(map(float,wdescList[i].split(',')))
        descriptors[i] = des
    return torch.FloatTensor(descriptors)

parser = argparse.ArgumentParser(description='Bentham sample encoder')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

sample = getWordsDescriptors('all_desc.txt',30*672)
isomap_sample = getWordsDescriptors('iso25.txt',25)

encoder = Encoder().to(device)
optimizer = optim.Adam(encoder.parameters(), lr=1e-4)

def loss_function(mu, isomap_z):
    mse = nn.MSELoss()
    loss = mse(mu, isomap_z.view(-1, 25))
    return loss/2

def getLowDim(vaeInput):
    for i in range(len(sample)):
        if (sample[i] == vaeInput).all():
            return isomap_sample[i]

def train(epoch):
    print('Training Encoder -> Epoch = ' + str(epoch))
    encoder.train()
    train_loss = 0
    for batch_idx, data in enumerate(sample):
        data = data.to(device)
        optimizer.zero_grad()
        mu = encoder(data)
        low_iso = getLowDim(data)
        loss = loss_function(mu,low_iso)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 500 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch, batch_idx, train_loss / 2000))
            train_loss = 0.0

def test(epoch):
    encoder.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(sample):
            data = data.to(device)
            recon_batch = encoder(data)
            test_loss += loss_function(recon_batch, getLowDim(data)).item()
    test_loss /= len(sample)
    print('====> Test set loss: {:.4f}'.format(test_loss))

#def update():
#    encoder.update()

def main():
    import time
    start = time.time()
    for epoch in range(1,20):
        if epoch % 5 == 0:
            torch.save(encoder, 'encoderModel'+str(epoch)+'.pth')
        train(epoch)
        test(epoch)
    torch.save(encoder, 'encoderModel.pth')
    end = time.time()
    print((end - start) // 60)

if __name__ == '__main__':
    main()

