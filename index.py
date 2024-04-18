import faiss
import os
import numpy as np
import numpy.typing as npt
from cache import Cache

class Index:

    def __init__(self, dataset: str, index_type: str, xt: npt.NDArray, xb: npt.NDArray, xq: npt.NDArray, gt: npt.NDArray):
        if dataset == 'sift':
            distance_metric = faiss.METRIC_L2
        elif dataset.startswith('glove'):
            distance_metric = faiss.METRIC_INNER_PRODUCT
            faiss.normalize_L2(xt)
            faiss.normalize_L2(xb)
            faiss.normalize_L2(xq)
        try:
            index = faiss.read_index(f'indexes/{dataset}/{index_type}.index', faiss.IO_FLAG_MMAP)
        except:
            index = faiss.index_factory(xt.shape[1], index_type, distance_metric)
            print(f"Training {index_type} index on {dataset}...")
            index.train(xt)
            index.add(xb)
            print(f"Write indexes/{dataset}/{index_type}.index")
            if not os.path.exists(f'indexes/{dataset}'):
                os.makedirs(f'indexes/{dataset}', exist_ok=True)
            faiss.write_index(index, f'indexes/{dataset}/{index_type}.index')
        self.index_type = index_type
        self.dataset = dataset
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
        labels = [x[0] for x in labels]
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
    
    def report_recall(self, ids: npt.NDArray, verbose=False) -> float:
        """ compares ids to gt, reports recall to stdout """
        recall_at_1 = (ids == self.gt[:, :1]).sum() / float(self.xq.shape[0])
        if verbose:
            print("recall@1: %.3f" % recall_at_1)
        return recall_at_1


    def find_nearest_centroids(self, k: int) -> npt.NDArray:
        """ our own method that uses faiss.knn to return the k closest centroids to each query vector\n
            returns an nxk array
        """
        centroids = self.index.quantizer.reconstruct_n(0, self.index.nlist)
        _, centroid_idxs = faiss.knn(self.xq, centroids, k)
        return centroid_idxs
    
    def simulate_cache(self, cache: Cache, nprobe: int) -> tuple[int, int]:
        """ performs the simulation of disk reads using the cache passed in\n
            returns the number of unique centroids and number of unique vectors accessed
        """
        print("Starting simulation...")
        print(f"\tIndex type: {self.index_type}")
        print(f"\tCache type: {cache.to_string()}")
        centroid_idxs = self.find_nearest_centroids(nprobe).flatten()
        list_sizes = self.get_list_sizes()
        
        for centroid in centroid_idxs:
            cache.access_item(centroid)

        unique_centroids_accessed = np.unique(centroid_idxs)
        num_unique_vectors_read = sum([list_sizes[x] for x in unique_centroids_accessed])
        print('Simulation results:')
        print(f'\t{cache.num_hits()} cache hits')
        print(f'\t{cache.num_misses()} disk reads ({len(unique_centroids_accessed)} unavoidable)')
        print(f'\t{cache.num_vectors_read()} vectors read from disk ({num_unique_vectors_read} unavoidable)')
        return len(unique_centroids_accessed), num_unique_vectors_read