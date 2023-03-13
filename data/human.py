import tarfile
import os
base = os.listdir('data/monofile')
for i in base:
    with tarfile.open(os.path.join('data/file',i), 'r:gz') as tr:
        tr.extractall('data/monocdf')   
        