import utils
import index
import faiss
import ctypes
import numpy as np
import faiss.contrib.inspect_tools as tools

def main():

    utils.get_sift()
    xt = utils.fvecs_read("sift/sift_learn.fvecs")
    xb = utils.fvecs_read("sift/sift_base.fvecs")
    xq = utils.fvecs_read("sift/sift_query.fvecs")
    gt = utils.ivecs_read("sift/sift_groundtruth.ivecs")
    ind = index.Index("IVF4096,Flat", xt, xb, xq, gt)
    
    # cache_size is in "number of vectors"
    ind.simulate_queries(cache_size=100000, nprobe=16)

    ind.search_and_report_recall(nprobe=16)


    print("done")
    
    

if __name__ == "__main__":
    main()