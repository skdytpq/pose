import tarfile
import os
base = os.listdir('data/file')
for i in base:
    with tarfile.open(os.path.join(base,i), 'r:gz') as tr:
        tr.extractall('data/cdf') 