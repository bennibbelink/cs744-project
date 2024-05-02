import utils
import index
from cache import LRUCache, PinCache, RandomCache
import numpy as np
from test_runner import TestRunner

def main():
    matrix = {
        # 'sift': {
        #     'n_clusters': [2048, 4096],
        #     'caches': [
        #         LRUCache(capacity=50000),
        #         PinCache(capacity=100000, pincount=5)
        #     ]
        # },
        # 'sift': {
        #     'n_clusters': [2048, 4096, 8192, 16384, 32768, 65536, 131072],
        #     'caches': [
        #         LRUCache(capacity=0),
        #         LRUCache(capacity=10000),
        #         LRUCache(capacity=50000),
        #         LRUCache(capacity=100000),
        #         LRUCache(capacity=200000),
        #         LRUCache(capacity=500000),
        #         LRUCache(capacity=1000000),
        #         RandomCache(capacity=0),
        #         RandomCache(capacity=10000),
        #         RandomCache(capacity=50000),
        #         RandomCache(capacity=100000),
        #         RandomCache(capacity=200000),
        #         RandomCache(capacity=500000),
        #         RandomCache(capacity=1000000),
        #     ]
        # },
        # 'glove-25': {
        #     'n_clusters': [2048, 4096, 8192, 16384, 32768, 65536, 131072],
        #     'caches': [
        #         LRUCache(capacity=0),
        #         LRUCache(capacity=10000),
        #         LRUCache(capacity=50000),
        #         LRUCache(capacity=100000),
        #         LRUCache(capacity=200000),
        #         LRUCache(capacity=500000),
        #         LRUCache(capacity=1000000),
        #         RandomCache(capacity=0),
        #         RandomCache(capacity=10000),
        #         RandomCache(capacity=50000),
        #         RandomCache(capacity=100000),
        #         RandomCache(capacity=200000),
        #         RandomCache(capacity=500000),
        #         RandomCache(capacity=1000000),
        #     ]
        # },
        # 'glove-200': {
        #     'n_clusters': [2048, 4096, 8192, 16384, 32768, 65536, 131072],
        #     'caches': [
        #         LRUCache(capacity=0),
        #         LRUCache(capacity=10000),
        #         LRUCache(capacity=50000),
        #         LRUCache(capacity=100000),
        #         LRUCache(capacity=200000),
        #         LRUCache(capacity=500000),
        #         LRUCache(capacity=1000000),
        #         RandomCache(capacity=0),
        #         RandomCache(capacity=10000),
        #         RandomCache(capacity=50000),
        #         RandomCache(capacity=100000),
        #         RandomCache(capacity=200000),
        #         RandomCache(capacity=500000),
        #         RandomCache(capacity=1000000),
        #     ]
        # },
        # 'sift': {
        #     'n_clusters': [131072],
        #     'caches': [
        #         PinCache(capacity=10000, pincount=0),
        #         PinCache(capacity=10000, pincount=50),
        #         PinCache(capacity=10000, pincount=100),
        #         PinCache(capacity=10000, pincount=150),
        #         PinCache(capacity=10000, pincount=200),
        #         PinCache(capacity=100000, pincount=0),
        #         PinCache(capacity=100000, pincount=500),
        #         PinCache(capacity=100000, pincount=1000),
        #         PinCache(capacity=100000, pincount=1500),
        #         PinCache(capacity=100000, pincount=2000),
        #         PinCache(capacity=200000, pincount=0),
        #         PinCache(capacity=200000, pincount=1000),
        #         PinCache(capacity=200000, pincount=2000),
        #         PinCache(capacity=200000, pincount=3000),
        #         PinCache(capacity=200000, pincount=4000),
        #         PinCache(capacity=500000, pincount=0),
        #         PinCache(capacity=500000, pincount=2500),
        #         PinCache(capacity=500000, pincount=5000),
        #         PinCache(capacity=500000, pincount=7500),
        #         PinCache(capacity=500000, pincount=10000),
        #     ]
        # },
        # 'glove-25': {
        #     'n_clusters': [131072],
        #     'caches': [
        #         PinCache(capacity=10000, pincount=0),
        #         PinCache(capacity=10000, pincount=50),
        #         PinCache(capacity=10000, pincount=100),
        #         PinCache(capacity=10000, pincount=150),
        #         PinCache(capacity=10000, pincount=200),
        #         PinCache(capacity=100000, pincount=0),
        #         PinCache(capacity=100000, pincount=500),
        #         PinCache(capacity=100000, pincount=1000),
        #         PinCache(capacity=100000, pincount=1500),
        #         PinCache(capacity=100000, pincount=2000),
        #         PinCache(capacity=200000, pincount=0),
        #         PinCache(capacity=200000, pincount=1000),
        #         PinCache(capacity=200000, pincount=2000),
        #         PinCache(capacity=200000, pincount=3000),
        #         PinCache(capacity=200000, pincount=4000),
        #         PinCache(capacity=500000, pincount=0),
        #         PinCache(capacity=500000, pincount=2500),
        #         PinCache(capacity=500000, pincount=5000),
        #         PinCache(capacity=500000, pincount=7500),
        #         PinCache(capacity=500000, pincount=10000),
        #     ]
        # },
        # 'glove-200': {
        #     'n_clusters': [131072],
        #     'caches': [
        #         # PinCache(capacity=10000, pincount=0),
        #         # PinCache(capacity=10000, pincount=25),
        #         # PinCache(capacity=10000, pincount=50),
        #         PinCache(capacity=10000, pincount=75),
        #         PinCache(capacity=10000, pincount=100),
        #         PinCache(capacity=100000, pincount=0),
        #         PinCache(capacity=100000, pincount=250),
        #         PinCache(capacity=100000, pincount=500),
        #         PinCache(capacity=100000, pincount=750),
        #         PinCache(capacity=100000, pincount=1000),
        #         PinCache(capacity=200000, pincount=0),
        #         PinCache(capacity=200000, pincount=500),
        #         PinCache(capacity=200000, pincount=1000),
        #         PinCache(capacity=200000, pincount=1500),
        #         PinCache(capacity=200000, pincount=2000),
        #         PinCache(capacity=500000, pincount=0),
        #         PinCache(capacity=500000, pincount=1250),
        #         PinCache(capacity=500000, pincount=2500),
        #         PinCache(capacity=500000, pincount=3750),
        #         PinCache(capacity=500000, pincount=5000),
        #     ]
        # },
        'sift': {
            'n_clusters': [131072],
            'caches': [
                PinCache(capacity=500000, pincount=0),
                PinCache(capacity=500000, pincount=2500),
                PinCache(capacity=500000, pincount=5000),
                PinCache(capacity=500000, pincount=7500),
                PinCache(capacity=500000, pincount=10000),
            ]
        },
        'glove-25': {
            'n_clusters': [131072],
            'caches': [
                PinCache(capacity=500000, pincount=0),
                PinCache(capacity=500000, pincount=2500),
                PinCache(capacity=500000, pincount=5000),
                PinCache(capacity=500000, pincount=7500),
                PinCache(capacity=500000, pincount=10000),
            ]
        },
        'glove-200': {
            'n_clusters': [131072],
            'caches': [
                PinCache(capacity=500000, pincount=0),
                PinCache(capacity=500000, pincount=1250),
                PinCache(capacity=500000, pincount=2500),
                PinCache(capacity=500000, pincount=3750),
                PinCache(capacity=500000, pincount=5000),
            ]
        },
    }
    
    runner = TestRunner(matrix, recall_target=0.9)
    runner.run_testing_matrix()
    # runner.write_results('results.csv')

    print("Done")
    
    

if __name__ == "__main__":
    main()