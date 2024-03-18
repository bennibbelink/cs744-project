import utils
import index
import faiss
import ctypes
import numpy as np
import faiss.contrib.inspect_tools as tools

def main():

    xt, xb, xq, gt = utils.get_sift()
    xt, xb, xq, gt = utils.get_glove(25)

    ind = index.Index("IVF4096,Flat", xt, xb, xq, gt)

    nprobe = 16
    cache_size = 100000

    # cache_size is in "number of vectors"
    ind.simulate_cache(cache_size=cache_size, nprobe=nprobe)

    print("search_and_return_centroids...")
    _query_centroid_ids, _result_centroid_ids, labels = ind.search_and_return_centroids(nprobe=nprobe)
    ind.report_recall(labels)

    print("standard search...")
    labels = ind.search(nprobe=nprobe)
    ind.report_recall(labels)


    print("done")
    
    

if __name__ == "__main__":
    main()