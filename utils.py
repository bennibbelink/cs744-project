import urllib.request
import os
import tarfile
import numpy as np
import h5py
import requests

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
    xt = fvecs_read("sift/sift_learn.fvecs")
    xb = fvecs_read("sift/sift_base.fvecs")
    xq = fvecs_read("sift/sift_query.fvecs")
    gt = ivecs_read("sift/sift_groundtruth.ivecs")
    return xt, xb, xq, gt

# dims should be one of [25, 50, 100, 200]
def get_glove(dims):
    filename = f'glove-{dims}-angular.hdf5'
    if not os.path.exists(filename):
        print(f'Downloading glove data as {filename}...')
        urllib.request.urlretrieve(f'http://ann-benchmarks.com/glove-{dims}-angular.hdf5', filename)
    with h5py.File(filename, "r") as f:
        _distances = ['distances']
        neighbors = np.array(f['neighbors'])
        test = np.array(f['test'])
        train = np.array(f['train'])
        return train, train, test, neighbors

