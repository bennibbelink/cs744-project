import faiss
import os
class Index:

    def __init__(self, index_type, xt, xb, xq, gt):
        try:
            self.index = faiss.read_index('indexes/' + index_type + '.index', faiss.IO_FLAG_MMAP)
        except:
            self.index = faiss.index_factory(xt.shape[1], index_type)
            print(f"Training {index_type} index...")
            self.index.train(xt)
            self.index.add(xb)
            print("Write indexes/" + index_type + ".index")
            if not os.path.exists('indexes'):
                os.mkdir('indexes')
            faiss.write_index(self.index, 'indexes/' + index_type + ".index")

        self.xb = xb
        self.xt = xt
        self.xq = xq
        self.gt = gt