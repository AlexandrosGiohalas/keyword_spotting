from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import math
import numpy as np

dim = 28
def load(myFeatures):
    num_lines = sum(1 for line in open(myFeatures))
    distances = open(myFeatures,'r')
    disList = list(distances)
    feature_dim = 672
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

def saveVectors(vectors):
    lowDimDistance = open('isomap_low_dimension_vectors.txt', 'w+')

    print(vectors.shape)
    for i in range(len(vectors)):
        for j in range(len(vectors[0])):
            if j == len(vectors[0])-1:
                lowDimDistance.write(str(vectors[i][j]))
            else:
                lowDimDistance.write(str(vectors[i][j]) + ' ')
        lowDimDistance.write('\n')
def computeDistance(sample,test_image):
    dis = 0
    for i in range(len(sample)):
        dis += math.sqrt(math.pow(sample[i].item() - test_image[i].item(),2))
    return dis

featureVectorList = load('../feature_vectors/zah-master/distance.txt')

def changeDim(x):
    global dim
    dim = x

def main():
    global featureVectorList
    X_iso = manifold.Isomap(5, n_components=dim).fit_transform(featureVectorList)
    saveVectors(X_iso)


if __name__ == '__main__':
    main()
