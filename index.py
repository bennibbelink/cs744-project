import faiss

class Index:

    def __init__(self, index_type, train_data):
        try:
            self.index = faiss.read_index('indexes/' + index_type + '.index')
        except:
            self.index = faiss.index_factory(train_data.shape[1], index_type)
            print(f"Training {index_type} index...")
            self.index.train(train_data)
            print("Write indexes/" + index_type + ".index")
            faiss.write_index(self.index, 'indexes/' + index_type + ".index")

