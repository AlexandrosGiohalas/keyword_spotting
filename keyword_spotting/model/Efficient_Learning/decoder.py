from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

dim = 30*672

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(25, dim//2)
        self.fc2 = nn.Linear(dim//2, dim)

    def decode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(h1))

    def forward(self, x):
        z = self.decode(x.view(-1, 25))
        return z

def getWordsDescriptors(filename,dimension):
    wdescFile = open(filename,'r')
    wdescList = list(wdescFile)
    descriptors = np.zeros((2501,dimension))
    for i in range(len(wdescList)):
        des = list(map(float,wdescList[i].split(',')))
        descriptors[i] = des
    return torch.FloatTensor(descriptors)

def decoder_loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 30*672), size_average=False)
    return BCE

parser = argparse.ArgumentParser(description='Bentham sample decoder')
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

decoder = Decoder().to(device)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

iso_sample = getWordsDescriptors('all_desc.txt',30*672)
decoder_sample = getWordsDescriptors('iso25.txt',25)

def getFeature(vaeInput):
    for i in range(len(decoder_sample)):
        if (decoder_sample[i] == vaeInput).all():
            return iso_sample[i]

def train(epoch):
    print('Training decoder -> Epoch = '+str(epoch))
    decoder.train()
    train_loss = 0
    for batch_idx, data in enumerate(decoder_sample):
        data = data.to(device)
        decoder_optimizer.zero_grad()
        recon_batch = decoder(data)
        X_zah = getFeature(data)
        loss = decoder_loss_function(recon_batch, X_zah)
        loss.backward()
        train_loss += loss.item()
        decoder_optimizer.step()

        if batch_idx % 500 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch, batch_idx, train_loss / 2000))
            train_loss = 0.0

def test(epoch):
    decoder.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(decoder_sample):
            data = data.to(device)
            recon_batch = decoder(data)
            test_loss += decoder_loss_function(recon_batch, getFeature(data)).item()
    test_loss /= len(decoder_sample)
    print('Epoch: '+str(epoch)+'====> Test set loss: {:.4f}'.format(test_loss))

def main():
    import time
    start = time.time()
    for epoch in range(1, 5):
        train(epoch)
        test(epoch)
        torch.save(decoder, 'decoderModel'+str(epoch)+'.pth')
    end = time.time()
    print((end - start) // 60)

if __name__ == '__main__':
    main()
