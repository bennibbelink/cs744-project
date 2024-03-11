import faiss
import os
from collections import Counter
class Index:

    def __init__(self, index_type, xt, xb, xq, gt):
        try:
            index = faiss.read_index('indexes/' + index_type + '.index', faiss.IO_FLAG_MMAP)
        except:
            index = faiss.index_factory(xt.shape[1], index_type)
            print(f"Training {index_type} index...")
            index.train(xt)
            index.add(xb)
            print("Write indexes/" + index_type + ".index")
            if not os.path.exists('indexes'):
                os.mkdir('indexes')
            faiss.write_index(index, 'indexes/' + index_type + ".index")
        
        self.index = index
        self.index_ivf = faiss.extract_index_ivf(index)
        self.xb = xb
        self.xt = xt
        self.xq = xq
        self.gt = gt

    def find_nearest_centroids(self):
        centroids = self.index.quantizer.reconstruct_n(0, self.index.nlist)
        _, centroid_idxs = faiss.knn(self.xq, centroids, 1)
        return [x[0] for x in centroid_idxs]

    def search_and_report_recall(self):
        self.index.nprobe = 16
        _, I = self.index_ivf.search(self.xq, 1)
        recall_at_1 = (I[:, :1] == self.gt[:, :1]).sum() / float(self.xq.shape[0])
        print("recall@1: %.3f" % recall_at_1)

    def simulate_queries(self, cache_size):
        idx_cache = [] # index 0 is the MRU centroid
        num_disk_reads = 0
        centroid_idxs = self.find_nearest_centroids()
        for centroid in centroid_idxs:
            if centroid in idx_cache: # cluster is in cache
                idx_cache.remove(centroid)
            else:
                num_disk_reads += 1

            # insert this centroid at front of list
            idx_cache.insert(0, centroid) 

            if len(idx_cache) > cache_size: # over cache size limit
                popped = idx_cache.pop()
                print(f'Evicting centroid {popped} from cache')

        cache_hits = len(centroid_idxs) - num_disk_reads
        num_centroids_accessed = len(Counter(centroid_idxs).keys())
        print('Simulation results:')
        print(f'\t{cache_hits} cache hits')
        print(f'\t{num_disk_reads} disk reads ({num_centroids_accessed} unavoidable)')

