import torch
import numpy as np
import math


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
    for i in range(num_lines):
        for j in range(feature_dim):
            if (math.isnan(floatList[i][j])):
                floatList[i][j] = 0.0
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