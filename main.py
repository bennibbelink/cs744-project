import utils
import index
from cache import LRUCache, PinCache
import numpy as np
from test_runner import TestRunner

def main():
    matrix = {
        # 'sift': {
        #     'n_clusters': [2048, 4096],
        #     'caches': [
        #         LRUCache(capacity=50000),
        #         LRUCache(capacity=100000),
        #         PinCache(capacity=100000, pincount=5)
        #     ]
        # },
        'openai': {
            'n_clusters': [2048],
            'caches': [
                LRUCache(capacity=50000),
                # LRUCache(capacity=100000),
                # PinCache(capacity=100000, pincount=5)
            ]
        },
        # 'glove-50': {
        #     'n_clusters': [64, 128],
        #     'caches': [
        #         LRUCache(capacity=100000),
        #         PinCache(capacity=100000, pincount=5)
        #     ]
        # },
        # 'glove-100': {
        #     'n_clusters': [2048, 4096],
        #     'caches': [
        #         LRUCache(capacity=100000),
        #         PinCache(capacity=100000, pincount=5)
        #     ]
        # }
    }
    
    runner = TestRunner(matrix, recall_target=0.9)
    runner.run_testing_matrix()
    runner.write_results('results.csv')

    print("Done")
    
    

if __name__ == "__main__":
    main()