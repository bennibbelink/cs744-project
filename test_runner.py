from index import Index
from cache import Cache
import utils
import csv
import os
import json

class Result():

    def __init__(self, 
                dataset,
                index_description,
                cache_description,
                hits, 
                misses, 
                vectors_read, 
                num_necessary_cluster_reads,
                num_necessary_vector_reads,
                recall,
                nprobe
                ):
        self.dataset = dataset
        self.index_description = index_description
        self.cache_description = cache_description
        self.hits = hits
        self.misses = misses
        self.vectors_read = vectors_read
        self.num_necessary_cluster_reads = num_necessary_cluster_reads
        self.num_necessary_vector_reads = num_necessary_vector_reads
        self.recall = recall
        self.nprobe = nprobe

    def to_row(self) -> list[any]:
        return [
            self.dataset,
            self.index_description,
            self.cache_description,
            self.hits,
            self.misses,
            self.vectors_read,
            self.num_necessary_cluster_reads,
            self.num_necessary_vector_reads,
            self.recall,
            self.nprobe
        ]

class TestRunner():

    def __init__(self, 
                matrix: any,
                recall_target: float
                ):
        self.matrix = matrix
        self.recall_target = recall_target
        self.results = []
        self.nprobe_cache = {}
        if os.path.isfile('nprobe_cache.json'):
            with open('nprobe_cache.json') as f:
                self.nprobe_cache = json.load(f)
            
    def __del__(self):
        with open('nprobe_cache.json', 'w') as f:
            json.dump(self.nprobe_cache, f)
    
    def run_testing_matrix(self):
        for dataset in self.matrix.keys():
            submatrix = self.matrix[dataset]
            for cluster_size in submatrix['n_clusters']:
                for cache in submatrix['caches']:
                    self.run_single_sim(dataset, cluster_size, cache)

    def run_single_sim(self, 
                       dataset: str, 
                       n_clusters: int,
                       cache: Cache
                       ):
        if dataset == 'sift':
            xt, xb, xq, gt = utils.get_sift()
        elif dataset.startswith('glove'):
            n_dims = dataset.split('-')[1]
            xt, xb, xq, gt = utils.get_glove(n_dims)

        ind = Index(dataset, f'IVF{n_clusters},Flat', xt, xb, xq, gt)
        nprobe = self.find_nprobe(ind)
        cache.reset()
        try:
            cache.setup(index=ind)
            u_centroids, u_vectors = ind.simulate_cache(cache, nprobe)
            labels = ind.search(nprobe)
            recall = ind.report_recall(labels)
            result = Result(
                dataset=dataset,
                index_description=ind.index_type,
                cache_description=cache.to_string(),
                hits=cache.num_hits(),
                misses=cache.num_misses(),
                vectors_read=cache.num_vectors_read(),
                num_necessary_cluster_reads=u_centroids,
                num_necessary_vector_reads=u_vectors,
                recall=recall,
                nprobe=nprobe
                )
            self.results.append(result)
        except:
            print("Exception occured in simulation, likely due to pincount")
            result = Result(ind.index_type, cache.to_string(), 0, 0, 0, 0, 0, 0, 0)
            self.results.append(result)

    def write_results(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'dataset',
                'index_description',
                'cache_description',
                'hits', 
                'misses', 
                'vectors_read', 
                'num_necessary_cluster_reads',
                'num_necessary_vector_reads',
                'recall',
                'nprobe'
            ])
            result_rows = [x.to_row() for x in self.results]
            writer.writerows(result_rows)

    def find_nprobe(self, index: Index) -> int:
        print(f"Finding nprobe for {index.dataset} - {index.index_type}")
        if index.dataset in self.nprobe_cache:
            if index.index_type in self.nprobe_cache[index.dataset]:
                nprobe = self.nprobe_cache[index.dataset][index.index_type]
                print(f"Found nprobe={nprobe} in nprobe cache")
                return nprobe

        search_factor = 2
        nprobe = 8
        labels = index.search(nprobe)
        recall = index.report_recall(labels)
        print(f'\tnprobe={nprobe}, recall={recall}')
        while recall <= self.recall_target:
            nprobe *= search_factor
            labels = index.search(nprobe)
            recall = index.report_recall(labels)
            print(f'\tprobe={nprobe}, recall={recall}')

        # we have gone past the threshold
        high = nprobe
        low = nprobe // search_factor

        # perform a normal binary search
        while low < high:
            # Calculate the middle point
            nprobe = (low + high) // 2

            # get recall for the middle point
            labels = index.search(nprobe)
            recall = index.report_recall(labels)
            print(f'\tnprobe={nprobe}, recall={recall}')

            if recall >= self.recall_target:
            # met the threshold, try searching lower
                high = nprobe
            else:
            # recall is too low, searching higher
                low = nprobe
            if high - low <= 1:
                break

        if recall < self.recall_target:
            nprobe += 1
            labels = index.search(nprobe)
            recall = index.report_recall(labels)
            print(f'\tnprobe={nprobe}, recall={recall}')
        # manage our nprobe cache
        if index.dataset not in self.nprobe_cache:
            self.nprobe_cache[index.dataset] = {}
        self.nprobe_cache[index.dataset][index.index_type] = nprobe
        return nprobe