import tarfile
import os
base = os.listdir('data/cdf')
for i in base:
    with tarfile.open(os.path.join(base,i), 'r:gz') as tr:
        tr.extractall(path='test_tar')