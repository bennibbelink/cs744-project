from abc import ABC, abstractmethod, abstractproperty
import numpy as np

class Cache(ABC):
    @abstractmethod
    def access_item(self, idx: int) -> None:
        pass

    @abstractmethod
    def get_capacity(self) -> int:
        pass

    @abstractmethod
    def get_size(self) -> int:
        pass

    @abstractmethod
    def num_hits(self) -> int:
        pass
    
    @abstractmethod
    def num_misses(self) -> int:
        pass
    
    @abstractmethod
    def num_vectors_read(self) -> int:
        pass


class LRUCache(Cache):

    def __init__(self, capacity: int, list_sizes: dict[int, int]):
        self.capacity = capacity
        self.list_sizes = list_sizes
        self.centroids = np.empty(0)
        self.hits = 0
        self.misses = 0
        self.vectors_read = 0
        
    def access_item(self, cid: int) -> None:
        if cid in self.centroids: # idx is in cache
            self.centroids = np.delete(self.centroids, np.where(self.centroids == cid))
            self.hits += 1
        else:
            self.misses += 1
            self.vectors_read += self.list_sizes[cid]

        self.centroids = np.insert(self.centroids, 0, cid)
        # sum([self.list_sizes[x] for x in self.centroids]) calculates the new cache size in 
        # terms of # of vectors
        while sum([self.list_sizes[x] for x in self.centroids]) > self.capacity:
            self.centroids = self.centroids[:-1]
    
    def get_capacity(self) -> int:
        return self.capacity
        
    def get_size(self) -> int:
        return np.size(self.centroids)
    
    def num_hits(self) -> int:
        return self.hits
    
    def num_misses(self) -> int:
        return self.misses
    
    def num_vectors_read(self) -> int:
        return self.vectors_read