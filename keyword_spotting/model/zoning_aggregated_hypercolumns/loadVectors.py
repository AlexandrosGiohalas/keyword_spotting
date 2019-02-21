import numpy as np
import shutil
import math

def load(myFeatures,dim):
    num_lines = sum(1 for line in open(myFeatures))
    distances = open(myFeatures,'r')
    disList = list(distances)
    feature_dim = dim
    floatList = np.zeros((num_lines,feature_dim))
    for i in range(num_lines):
        tokens = disList[i].split(' ')
        for j in range(feature_dim):
            floatList[i][j] = float(tokens[j])
    for i in range(num_lines):
        for j in range(feature_dim):
            if (math.isnan(floatList[i][j])):
                floatList[i][j] = 0.0
    return floatList

def saveResults(neighbours,destination):
    keys = sorted(neighbours, key=neighbours.get)
    for i in range(5):
        shutil.copy2('/home/alexandros/Desktop/gw/wordImages/'+imageNames[keys[i]],destination)

#imageFolder = open('../wordTrainingVAE/feature-extraction/zah-master/filenames.txt')
# imageNames = list(imageFolder)
# for i in range(len(imageNames)):
#     filename = imageNames[i].split('\n')
#     imageNames[i] = filename[0]

def saveVectors(vectors):
    lowDimDistance = open('lowDimDistance.txt', 'w+')
    print(vectors.shape)
    for i in range(len(vectors)):
        for j in range(len(vectors[0])):
            lowDimDistance.write(str(vectors[i][j]) + ' ')
        lowDimDistance.write('\n')