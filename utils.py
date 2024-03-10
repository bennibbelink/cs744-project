import urllib.request
import os
import tarfile
import numpy as np

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def get_sift():
    dir = 'sift'
    file = 'sift1M.tar.gz'
    if not os.path.exists(file):
        print(f'Downloading sift tarball as {file}...')
        urllib.request.urlretrieve('ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz', file)
    if not os.path.exists(dir):
        print(f'Extracting sift files to ./{dir}...')
        with tarfile.open(file, 'r:gz') as tar:
            tar.extractall()

