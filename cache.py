from abc import ABC, abstractmethod, abstractproperty
import numpy as np

class Cache(ABC):
    @abstractmethod
    def access_item(self, cid: int) -> None:
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
        if cid in self.centroids: # cid is in cache
            self.centroids = np.delete(self.centroids, np.where(self.centroids == cid))
            self.hits += 1
        else:
            self.misses += 1
            self.vectors_read += self.list_sizes[cid]

        # insert cid at the front of cache
        self.centroids = np.insert(self.centroids, 0, cid)

        # if cache is too big keep trimming fat until we are under capacity
        while self.get_size() > self.capacity:
            self.centroids = self.centroids[:-1]
    
    def get_capacity(self) -> int:
        return self.capacity
        
    def get_size(self) -> int:
        return sum([self.list_sizes[x] for x in self.centroids])
    
    def num_hits(self) -> int:
        return self.hits
    
    def num_misses(self) -> int:
        return self.misses
    
    def num_vectors_read(self) -> int:
        return self.vectors_read
    

class PinCache(Cache):
    def __init__(self, capacity: int, list_sizes: dict[int, int], pincount: int):
        self.capacity = capacity
        self.list_sizes = list_sizes
        self.pincount = pincount

        sorted_dict = dict(sorted(list_sizes.items(), key=lambda item: item[1], reverse=True))
        # Extract keys of the top pincount highest values
        top_keys = list(sorted_dict.keys())[:pincount]

        # initialize cache with pinned centroids
        # set the miss/read counts appropriately
        self.centroids = np.array(top_keys)
        self.hits = 0
        self.misses = pincount
        self.vectors_read = 0
        for cid in top_keys:
            self.vectors_read += list_sizes[cid]

        if self.get_size() > self.capacity:
            raise Exception('Error: Pinned clusters are larger than cache capacity')

    def access_item(self, cid: int) -> None:
        if cid in self.centroids[:self.pincount]: # cid is pinned
            self.hits += 1
        elif cid in self.centroids: # cid is in cache but not pinned
            self.hits += 1
            self.centroids = np.delete(self.centroids, np.where(self.centroids == cid))
        else:
            self.misses += 1
            self.vectors_read += self.list_sizes[cid]

        # if cid is not pinned, move it to the front of the cache (but still behind pinned cids)
        if cid not in self.centroids[:self.pincount]:
            self.centroids = np.insert(self.centroids, self.pincount, cid)

        # if cache is too big keep trimming fat until we are under capacity
        while self.get_size() > self.capacity:
            self.centroids = self.centroids[:-1]
    
    def get_capacity(self) -> int:
        return self.capacity
    
    def get_size(self) -> int:
        return sum([self.list_sizes[x] for x in self.centroids])

    def num_hits(self) -> int:
        return self.hits
    
    def num_misses(self) -> int:
        return self.misses
    
    def num_vectors_read(self) -> int:
        return self.vectors_read