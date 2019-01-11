from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from dataset import MyDataset
import loadVectors

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
dim = 28

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(672, 400)
        self.fc21 = nn.Linear(400, dim)
        self.fc22 = nn.Linear(400, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc21(h1)),self.fc22(h1)

    def forward(self, x):
        mu,logvar = self.encode(x.view(-1, 672))
        return mu,logvar

    def updateLayers(self):
        self.fc22 = nn.Linear(400,dim)

def encoder_loss_function(mu, isomap_z):
    mse = nn.MSELoss()
    loss = mse(mu,isomap_z.view(-1,dim))
    return loss/2.0

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(dim, 400)
        self.fc2 = nn.Linear(400, 672)

    def changeDim(self, x):
        global dim
        dim = x
        self.fc1 = nn.Linear(dim, 400)

    def decode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(h1))

    def forward(self, x):
        z = self.decode(x.view(-1, dim))
        return z

def decoder_loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 672), size_average=False)
    return BCE

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoderModel = encoder
        self.decoderModel = decoder

    def encode(self, x):
        return self.encoderModel.encode(x)
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return  self.decoderModel.decode(z)

    def update(self,newEncoder,newDecoder):
        self.encoderModel = newEncoder
        self.decoderModel = newDecoder

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 672))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 672), size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

dset = MyDataset()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    dset,batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    dset,batch_size=args.batch_size, shuffle=True, **kwargs)

featureVectorList = loadVectors.load('../feature_vectors/zah-master/distance.txt',672)
featureVectors = torch.FloatTensor(featureVectorList)

encoder = Encoder().to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-4)

decoder = Decoder().to(device)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

lowDimVectorList = loadVectors.load('isomap_low_dimension_vectors.txt', dim)
# lowDimVectorList = loadVectors.load('lowDimDistance.txt', dim)
lowDimVectors = torch.FloatTensor(lowDimVectorList)
def changeDim(x):
    global dim
    dim = x
    encoder.__init__()
    decoder.__init__()

def getLowDim(vaeInput):
    for i in range(len(featureVectors)):
        if (featureVectors[i] == vaeInput).all():
            return lowDimVectors[i]

def getFeature(vaeInput):
    for i in range(len(lowDimVectors)):
        if (lowDimVectors[i] == vaeInput).all():
            return featureVectors[i]

def train_encoder(epoch):
    print('Training encoder -> Epoch = '+str(epoch))
    encoder.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        encoder_optimizer.zero_grad()
        recon_batch,logvar = encoder(data)
        x_iso = getLowDim(data)
        loss = encoder_loss_function(recon_batch, x_iso)
        loss.backward()
        train_loss += loss.item()
        encoder_optimizer.step()
        #
        # if batch_idx % args.log_interval == 0:
        #     print(batch_idx, len(data), len(train_loader.dataset))
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * 2, len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader),
        #                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

def train_decoder(epoch):
    print('Training decoder -> Epoch = '+str(epoch))
    decoder.train()
    train_loss = 0
    for batch_idx, data in enumerate(lowDimVectors):
        data = data.to(device)
        decoder_optimizer.zero_grad()
        recon_batch = decoder(data)
        X_zah = getFeature(data)
        loss = decoder_loss_function(recon_batch, X_zah)
        loss.backward()
        train_loss += loss.item()
        decoder_optimizer.step()

        # if batch_idx % args.log_interval == 0:
        #     print(batch_idx, len(data), len(train_loader.dataset))
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * 2, len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader),
        #                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

def test_encoder(epoch):
    encoder.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch,logvar = encoder(data)
            test_loss += encoder_loss_function(recon_batch, getLowDim(data)).item()

def test_decoder(epoch):
    decoder.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(lowDimVectors):
            data = data.to(device)
            recon_batch = decoder(data)
            test_loss += decoder_loss_function(recon_batch, getFeature(data)).item()

def train_vae(epoch):
    print('Training Variational Autoencoder -> Epoch = ' + str(epoch))
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

def test_vae(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)

def main():
    global lowDimVectors,encoder,decoder,model
    lowDimVectorList = loadVectors.load('isomap_low_dimension_vectors.txt', dim)
    # lowDimVectorList = loadVectors.load('lowDimDistance.txt', dim)
    lowDimVectors = torch.FloatTensor(lowDimVectorList)
    for epoch in range(1,401):
        train_encoder(epoch)
        test_encoder(epoch)
    encoder.updateLayers()
    for epoch in range(1,3):
        train_decoder(epoch)
        test_decoder(epoch)

    torch.save(encoder, 'encoderModel.pth')
    torch.save(decoder, 'decoderModel.pth')
    model.update(encoder,decoder)

    for epoch in range(1, 401):
        train_vae(epoch)
        test_vae(epoch)
    torch.save(model, 'testDatasetModel.pth')

if __name__ == '__main__':
    main()