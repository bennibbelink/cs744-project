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

    def find_nearest_centroids(self, k):
        centroids = self.index.quantizer.reconstruct_n(0, self.index.nlist)
        _, centroid_idxs = faiss.knn(self.xq, centroids, k)
        return centroid_idxs

    def search_and_report_recall(self, nprobe):
        self.index.nprobe = nprobe
        _, I = self.index_ivf.search(self.xq, 1)
        recall_at_1 = (I[:, :1] == self.gt[:, :1]).sum() / float(self.xq.shape[0])
        print("recall@1: %.3f" % recall_at_1)

    def simulate_queries(self, cache_size, nprobe):
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
                print(f'Evicting centroid {popped} from cache')

        cache_hits = len(centroid_idxs) - n_disk_reads
        unique_centroids_accessed = Counter(centroid_idxs).keys()
        
        unique_vectors_read = sum([list_sizes[x] for x in unique_centroids_accessed])
        print('Simulation results:')
        print(f'\t{cache_hits} cache hits')
        print(f'\t{n_disk_reads} disk reads ({len(unique_centroids_accessed)} unavoidable)')
        print(f'\t{n_vectors_read_from_disk} vectors read from disk ({unique_vectors_read} unavoidable)')
   
    def get_list_sizes(self):
        invlists = self.index_ivf.invlists
        d = {}
        for i in range(invlists.nlist):
            d[i] = invlists.list_size(i)
        return d
        