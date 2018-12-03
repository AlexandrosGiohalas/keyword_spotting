from __future__ import print_function

import argparse
import math
import random
import shutil
import numpy as np
import torch
import torch.utils.data

parser = argparse.ArgumentParser(description='Load feature Vectors from File <distance.txt>')
#parser.add_argument('--data',default = '../wordTrainingVAE/feature-extraction/zah-master/distance.txt'
#			,type = str, help = '.../test.txt')
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
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    dset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

dim = 768
def changeDim(x):
    global dim
    dim = x

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

imageFolder = open(args.images)
imageNames = list(imageFolder)
wordPositionFile = open(args.position)
wordPosition = list(wordPositionFile)

for i in range(len(imageNames)):
    filename = imageNames[i].split('\n')
    imageNames[i] = filename[0]

def computeDistance(query,image):
    dis = 0
    for i in range(len(query)):
        dis += math.sqrt(math.pow(query[i].item() - image[i].item(),2))
    return dis

def saveResults(neighbours,destination):
    keys = sorted(neighbours, key=neighbours.get)
    for i in range(5):
        shutil.copy2('../../../Datasets/gw/words/'+imageNames[keys[i]],destination)

def main():
    testVectorList = loadVectorsFromFile(args.data)
    testVectors = torch.FloatTensor(testVectorList)
    model = torch.load('testDatasetModel.pth',map_location='cpu')
    a,b = model.encode(testVectors)
    tmp = list(a)
    lowDimDistance = open('../results/lowDimDistance2.txt','w+')
    for i in range(len(tmp)):
        for j in range(len(tmp[0])):
            lowDimDistance.write(str(a[i][j].item())+' ')
        lowDimDistance.write('\n')

if __name__ == '__main__':
    from train import VAE
    main()