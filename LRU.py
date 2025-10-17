import time
import threading

class Node:
    def __init__(self, key, val, ttl=None):
        self.key = key
        self.val = val
        self.ttl = ttl
        self.timestamp = time.time() if ttl else None
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity=100, thread_safe=False):
        self.capacity = capacity
        self.map = {}
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock() if thread_safe else None

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _evict(self):
        lru = self.tail.prev
        self._remove(lru)
        del self.map[lru.key]
        self.size -= 1

    def _expired(self, node):
        if node.ttl:
            return (time.time() - node.timestamp) > node.ttl
        return False

    def get(self, key):
        if self.lock: 
            with self.lock:
                return self._get(key)
        return self._get(key)

    def _get(self, key):
        node = self.map.get(key)
        if not node or self._expired(node):
            self.misses += 1
            if node and self._expired(node):
                self._remove(node)
                del self.map[key]
            return None
        self.hits += 1
        self._remove(node)
        self._add(node)
        return node.val

    def put(self, key, val, ttl=None):
        if self.lock: 
            with self.lock:
                return self._put(key, val, ttl)
        return self._put(key, val, ttl)

    def _put(self, key, val, ttl):
        node = self.map.get(key)
        if node:
            node.val = val
            node.ttl = ttl
            node.timestamp = time.time() if ttl else None
            self._remove(node)
            self._add(node)
        else:
            if self.size >= self.capacity:
                self._evict()
            new_node = Node(key, val, ttl)
            self.map[key] = new_node
            self._add(new_node)
            self.size += 1

    def stats(self):
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) else 0
        }
