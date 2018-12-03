from __future__ import print_function
import argparse
import math
import shutil
from train import VAE
import numpy as np
import torch
import torch.utils.data
from dataset import MyDataset

parser = argparse.ArgumentParser(description='Load feature Vectors from File <distance.txt>')
parser.add_argument('--data',default = '../feature_vectors/zah-master/distance.txt'
			,type = str, help = '.../test.txt')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test',default = 'train-test-FV/test.txt'
			,type = str, help = '.../test.txt')
parser.add_argument('--images',default = '../feature_vectors/zah-master/filenames.txt'
			,type = str, help = '.../word_images')
parser.add_argument('--position',default = '../../../Datasets/gw/queries/queries.gtp'
			,type = str, help = '.../word_positions')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

dset = MyDataset()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    dset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

def loadVectorsFromFile(myFeatures):
    num_lines = sum(1 for line in open(myFeatures))
    distances = open(myFeatures, 'r')
    disList = list(distances)
    print(len(disList),len(disList[0].split(' ')))
    feature_dim = 768
    floatList = np.zeros((num_lines, feature_dim))
    for i in range(num_lines):
        tokens = disList[i].split(' ')
        for j in range(feature_dim):
            floatList[i][j] = float(tokens[j])
    return floatList

def computeDistance(query,image):
    dis = 0
    for i in range(len(query)):
        dis += math.sqrt(math.pow(query[i].item() - image[i].item(),2))
    return dis

def main():
    testVectorList = loadVectorsFromFile(args.data)
    testVectors = torch.FloatTensor(testVectorList)
    model = torch.load('testDatasetModel.pth',map_location='cpu')
    a,b = model.encode(testVectors)
    tmp = list(a)
    lowDimDistance = open('../results/vae_low_dimension_vectors.txt','w+')
    for i in range(len(tmp)):
        for j in range(len(tmp[0])):
            lowDimDistance.write(str(a[i][j].item())+' ')
        lowDimDistance.write('\n')

if __name__ == '__main__':
    main()