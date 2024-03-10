import utils
import index
import faiss
import ctypes

def main():

    utils.get_sift()
    xt = utils.fvecs_read("sift/sift_learn.fvecs")
    xb = utils.fvecs_read("sift/sift_base.fvecs")
    xq = utils.fvecs_read("sift/sift_query.fvecs")
    gt = utils.ivecs_read("sift/sift_groundtruth.ivecs")
    ind = index.Index("IVF4096,Flat", xt, xb, xq, gt)

    index_ivf = faiss.extract_index_ivf(ind.index)
    distances = []
    labels = []
    qids = []
    rids = []
    
    libfaiss = ctypes.cdll.LoadLibrary('libfaiss.so')
    q = xq[0].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    print(type(q))
    libfaiss.search_and_return_centroids(index_ivf, 1, q, distances, labels, qids, rids)
    
    faiss.search_and_return_centroids(index_ivf, n=1, xin=q, k=1, 
                                      distances=distances, labels=labels, query_centroid_ids=qids, result_centroid_ids=rids)
    ind.index.nprobe = 16
    D, I = ind.index.search(xq, 1)
    recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(xq.shape[0])
    print("recall@1: %.3f" % recall_at_1)
    

if __name__ == "__main__":
    main()