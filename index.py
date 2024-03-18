import faiss
import os
from collections import Counter
import numpy as np
import numpy.typing as npt
import time

class Index:

    def __init__(self, dataset: str, index_type: str, xt: npt.NDArray, xb: npt.NDArray, xq: npt.NDArray, gt: npt.NDArray):
        try:
            index = faiss.read_index(f'indexes/{dataset}/{index_type}.index', faiss.IO_FLAG_MMAP)
        except:
            index = faiss.index_factory(xt.shape[1], index_type)
            print(f"Training {index_type} index on {dataset}...")
            index.train(xt)
            index.add(xb)
            print(f"Write indexes/{dataset}/{index_type}.addedindex")
            if not os.path.exists(f'indexes/{dataset}'):
                os.makedirs(f'indexes/{dataset}', exist_ok=True)
            faiss.write_index(index, f'indexes/{dataset}/{index_type}.index')
        
        self.index = index
        self.index_ivf = faiss.extract_index_ivf(index)
        self.xb = xb
        self.xt = xt
        self.xq = xq
        self.gt = gt

    def search(self, nprobe: int) -> npt.NDArray:
        """ performs the standard search on xq, returns the labels as a 1D array """
        self.index_ivf.nprobe = nprobe
        _, I = self.index_ivf.search(self.xq, k=1)
        labels = I[:, :1]
        # print(f'labels: {labels.size} - {np.min(labels)} -> {np.max(labels)}')
        return labels

    def search_centroid(self) -> npt.NDArray:
        """ finds the closest centroid to each vector in xq, returns a 1D array with the centroid ids"""
        n = len(self.xq)
        centroid_ids = np.full(n, fill_value=-1, dtype=np.int64)
        faiss.search_centroid(
            self.index_ivf, 
            faiss.swig_ptr(self.xq),
            n,
            faiss.swig_ptr(centroid_ids)
        )
        return centroid_ids

    def search_and_return_centroids(self, nprobe: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """ returns a tuple of 3 1D arrays corresponding to: closest centroid to query vector,
            actual centroid of NN found, and label of NN
        """
        self.index_ivf.nprobe = nprobe # how many clusters to search
        n = len(self.xq)
        k = 1
        distances = np.full(n * k, fill_value=-1, dtype=np.float32)
        labels = np.full(n, fill_value=-1, dtype=np.int64)
        query_centroid_ids = np.full(n, fill_value=-1, dtype=np.int64)
        result_centroid_ids = np.full(n * k, fill_value=-1, dtype=np.int64)

        faiss.search_and_return_centroids(
            self.index_ivf, 
            n, # number of query vectors
            faiss.swig_ptr(self.xq), # query vectors
            k, # num of nearest neighbors to get
            faiss.swig_ptr(distances), # distance to each of the result centroids
            faiss.swig_ptr(labels),
            faiss.swig_ptr(query_centroid_ids), # centroid ids corresponding to the query vectors (size n)
            faiss.swig_ptr(result_centroid_ids), # centroid ids corresponding to the results (size n * k)
        )
        # print(f'query_centroid_ids: {query_centroid_ids.shape} - {np.min(query_centroid_ids)} -> {np.max(query_centroid_ids)}')
        # print(f'result_centroid_ids: {result_centroid_ids.shape} - {np.min(result_centroid_ids)} -> {np.max(result_centroid_ids)}')
        # print(f'labels: {labels.shape} - {np.min(labels)} -> {np.max(labels)}')
        return query_centroid_ids, result_centroid_ids, labels
    
    def get_list_sizes(self) -> dict[int, int]:
        """ returns a dictionary containing the size of each inverted list,
            size = # of vectors
        """
        invlists = self.index_ivf.invlists
        d = {}
        for i in range(invlists.nlist):
            d[i] = invlists.list_size(i)
        return d
    
    def report_recall(self, ids: npt.NDArray) -> None:
        """ compares ids to gt, reports recall to stdout """
        recall_at_1 = (ids == self.gt[:, :1]).sum() / float(self.xq.shape[0])
        print("recall@1: %.3f" % recall_at_1)

    def find_nearest_centroids(self, k: int) -> npt.NDArray:
        """ our own method that uses faiss.knn to return the k closest centroids to each query vector\n
            returns an nxk array
        """
        centroids = self.index.quantizer.reconstruct_n(0, self.index.nlist)
        _, centroid_idxs = faiss.knn(self.xq, centroids, k)
        return centroid_idxs
    
    def simulate_cache(self, cache_size: int, nprobe: int) -> None:
        idx_cache = [] # index 0 is the MRU centroid
        n_disk_reads = 0
        n_vectors_read_from_disk = 0
        centroid_idxs = self.find_nearest_centroids(nprobe).flatten()
        list_sizes = self.get_list_sizes()
        for centroid in centroid_idxs:
            if centroid in idx_cache: # cluster is in cache
                idx_cache.remove(centroid)
            else:
                n_disk_reads += 1
                n_vectors_read_from_disk += list_sizes[centroid]

            # insert this centroid at front of list
            idx_cache.insert(0, centroid) 

            # calculate how many vectors are stored in the cache now
            new_cache_size = sum([list_sizes[x] for x in idx_cache])

            if new_cache_size > cache_size: # over cache size limit
                popped = idx_cache.pop()

        cache_hits = len(centroid_idxs) - n_disk_reads
        unique_centroids_accessed = Counter(centroid_idxs).keys()
        
        unique_vectors_read = sum([list_sizes[x] for x in unique_centroids_accessed])
        print('Old (correct?) simulation results:')
        print(f'\t{cache_hits} cache hits')
        print(f'\t{n_disk_reads} disk reads ({len(unique_centroids_accessed)} unavoidable)')
        print(f'\t{n_vectors_read_from_disk} vectors read from disk ({unique_vectors_read} unavoidable)')