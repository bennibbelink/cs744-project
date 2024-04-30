import urllib.request
import os
import tarfile
import numpy as np
import numpy.typing as npt
import h5py
import math
import pyarrow.parquet as pq
import faiss

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def get_sift() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """ Returns (xt, sb, xq, gt). Downloads dataset if doesn't already exist locally """
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
def get_glove(dims: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """ 
        Returns (xt, sb, xq, gt). Downloads dataset if doesn't already exist locally \n
        dims argument should be one of [25, 50, 100, 200]
    """
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

def get_openai() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    reponame = "dbpedia-entities-openai-1M"
    if not os.path.exists(reponame):
        print(f'You need to git lfs clone https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M')
        return
    dirname = reponame + '/data/'
    parquet_files = os.listdir(dirname)
    vectors = np.zeros((1000000, 1536))
    count = 0
    for f in parquet_files:
        print(f"Reading {f}...")
        par = pq.read_table(dirname + f) 
        embeddings = par['openai'].to_numpy()
        for v in embeddings:
            vectors[count] = v
            count += 1
    print("Done reading parquet files.")
    np.random.shuffle(vectors)
    total_num = len(vectors)
    xq_size = math.floor(total_num * 0.01)
    # test = vectors[:xq_size]
    # train = vectors[xq_size:]
    print("building exhaustive index")
    exhaustive_index = faiss.index_factory(vectors.shape[1], "Flat")
    exhaustive_index.train(vectors[xq_size:])
    exhaustive_index.add(vectors[xq_size:])
    _, I = exhaustive_index.search(vectors[:xq_size], k=1)
    neighbors = I[:, :1]
    print("done getting openai")
    return vectors[xq_size:], vectors[xq_size:], vectors[:xq_size], neighbors