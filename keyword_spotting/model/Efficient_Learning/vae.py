from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import encoder
import decoder
from encoder import Encoder
from decoder import Decoder
from customDataset import MyDataset

encoder_model = torch.load('encoderModel1.pth',map_location='cpu')
decoder_model = torch.load('decoderModel1.pth',map_location='cpu')

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model

    def encode(self, x):
        return self.encoder.encode(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        return self.decoder.decode(x)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 30*672))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def update(self,encoder,decoder):
        self.encoder = encoder
        self.decoder = decoder

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 30*672), size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

parser = argparse.ArgumentParser(description='GW Example')
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
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def getWordsDescriptors():
    wdescFile = open('all_desc.txt','r')
    wdescList = list(wdescFile)
    descriptors = np.zeros((2501,30*672))
    for i in range(len(wdescList)):
        des = list(map(float,wdescList[i].split(',')))
        descriptors[i] = des
    return torch.FloatTensor(descriptors)

dset = MyDataset()

train_loader = torch.utils.data.DataLoader(
    dset,batch_size=args.batch_size, shuffle=True, **kwargs)
    # getWordsDescriptors()

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_vae(epoch):
    print('Training Variational Autoencoder -> Epoch = ' + str(epoch))
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 500 == 0:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch, batch_idx, train_loss / 2000))
            train_loss = 0.0

def test_vae(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(train_loader)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def main():
    global encoder_model,decoder_model,model,optimizer
    for i in range(1,5):
        for j in range(1,5):
            encoder_model = torch.load('encoderModel'+str(i)+'.pth', map_location='cpu')
            decoder_model = torch.load('decoderModel'+str(j)+'.pth', map_location='cpu')
            model = VAE().to(device)
            model.update(encoder_model,decoder_model)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            for epoch in range(1, 5):
                train_vae(epoch)
                test_vae(epoch)
            torch.save(model, 'testDatasetModel'+str(i)+str(j)+'.pth')

if __name__ == '__main__':
    main()
