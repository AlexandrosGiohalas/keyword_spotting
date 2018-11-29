import train
import test
import sys
import time
sys.path.insert(0, '../wordTrainingVAE/isomap')
import isomap
import retrieve
for j in range(8):
    dim = (j+2)*50
    isomap.changeDim(dim)
    for i in range(5):
        print('Dimension -> '+str(i+15))
        start = time.time()
        isomap.main()
        train.changeDim(dim)
        train.changeDataset()
        train.changeMu(i+15)
        train.main()
        test.changeDim(dim)
        test.main()
        retrieve.changeDim(i+15)
        retrieve.main()
        end = time.time()
        print((end - start)//60)
        print('\n')