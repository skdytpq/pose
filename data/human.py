import tarfile
import os

tar = tarfile.open(os.listdir('data/cdf'))
tar.extractall()
tar.close()
