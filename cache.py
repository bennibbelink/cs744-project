from abc import ABC, abstractmethod
import numpy as np
import random

class Cache(ABC):

    @abstractmethod
    def setup(self, index) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

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

    @abstractmethod
    def to_string(self) -> str:
        pass


class LRUCache(Cache):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.list_sizes = None
        self.centroids = np.empty(0)
        self.size = 0
        self.hits = 0
        self.misses = 0
        self.vectors_read = 0

    # must be run before using the cache!!
    def setup(self, index):
        self.list_sizes = index.get_list_sizes()

    def reset(self):
        self.list_sizes = None
        self.centroids = np.empty(0)
        self.size = 0
        self.hits = 0
        self.misses = 0
        self.vectors_read = 0
        
    def access_item(self, cid: int) -> None:
        if cid in self.centroids: # cid is in cache
            self.centroids = np.delete(self.centroids, np.where(self.centroids == cid))
            self.size -= self.list_sizes[cid] # need to remove the size here, since we add it back later
            self.hits += 1
        else:
            self.misses += 1
            self.vectors_read += self.list_sizes[cid]

        # insert cid at the front of cache
        self.centroids = np.insert(self.centroids, 0, cid)
        self.size += self.list_sizes[cid]

        # if cache is too big keep trimming fat until we are under capacity
        while self.size > self.capacity:
            self.size -= self.list_sizes[self.centroids[-1]]
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
    
    def to_string(self) -> str:
        return f"LRUCache (capacity={self.get_capacity()})"
    

class PinCache(Cache):
    def __init__(self, capacity: int, pincount: int):
        self.capacity = capacity
        self.list_sizes = None
        self.pincount = pincount
        self.size = 0
        self.hits = 0
        self.misses = 0
        self.vectors_read = 0

    def setup(self, index):
        self.list_sizes = index.get_list_sizes()
        sorted_dict = dict(sorted(self.list_sizes.items(), key=lambda item: item[1], reverse=True))
        # Extract keys of the top pincount highest values
        top_keys = list(sorted_dict.keys())[:self.pincount]

        # initialize cache with pinned centroids
        # set the miss/read counts appropriately
        self.centroids = np.array(top_keys)
        self.misses = self.pincount
        for cid in top_keys:
            self.vectors_read += self.list_sizes[cid]
        if self.get_size() > self.capacity:
            raise Exception('Error: Pinned clusters are larger than cache capacity')
        self.size = self.get_size()

    def reset(self):
        self.list_sizes = None
        self.centroids = np.empty(0)
        self.size = 0
        self.hits = 0
        self.misses = 0
        self.vectors_read = 0


    def access_item(self, cid: int) -> None:
        if cid in self.centroids[:self.pincount]: # cid is pinned
            self.hits += 1
        elif cid in self.centroids: # cid is in cache but not pinned
            self.hits += 1
            self.centroids = np.delete(self.centroids, np.where(self.centroids == cid))
            self.size -= self.list_sizes[cid] # need to remove the size here, since we add it back later
        else:
            self.misses += 1
            self.vectors_read += self.list_sizes[cid]

        # if cid is not pinned, move it to the front of the cache (but still behind pinned cids)
        if cid not in self.centroids[:self.pincount]:
            self.centroids = np.insert(self.centroids, self.pincount, cid)
            self.size += self.list_sizes[cid]

        # if cache is too big keep trimming fat until we are under capacity
        while self.size > self.capacity:
            self.size -= self.list_sizes[self.centroids[-1]]
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
    
    def to_string(self) -> str:
        return f"PinCache (capacity={self.get_capacity()}, pincount={self.pincount})"
    
    
class RandomCache(Cache):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.list_sizes = None
        self.centroids = np.empty(0)
        self.size = 0
        self.hits = 0
        self.misses = 0
        self.vectors_read = 0

    # must be run before using the cache!!
    def setup(self, index):
        self.list_sizes = index.get_list_sizes()

    def reset(self):
        self.list_sizes = None
        self.centroids = np.empty(0)
        self.size = 0
        self.hits = 0
        self.misses = 0
        self.vectors_read = 0
        
    def access_item(self, cid: int) -> None:
        if cid in self.centroids: # cid is in cache
            self.centroids = np.delete(self.centroids, np.where(self.centroids == cid))
            self.size -= self.list_sizes[cid] # need to remove the size here, since we add it back later
            self.hits += 1
        else:
            self.misses += 1
            self.vectors_read += self.list_sizes[cid]
            
        # if cache will be too big keep trimming fat until we are under capacity
        if self.capacity > 0:
            while self.size + self.list_sizes[cid] > self.capacity:
                ind_to_remove = random.randrange(len(self.centroids))
                self.size -= self.list_sizes[self.centroids[ind_to_remove]]
                self.centroids = np.delete(self.centroids, ind_to_remove)
            
            # insert cid at the front of cache
            self.centroids = np.insert(self.centroids, 0, cid)
            self.size += self.list_sizes[cid]
    
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
    
    def to_string(self) -> str:
        return f"RandomCache (capacity={self.get_capacity()})"
