import numpy as np

class Queue:
    def __init__(self, size):
        self.items = np.empty(
            (size, 2, 2)
        )
        self.size = size
        self.index = 0
        self.count = 0  # Track the number of valid elements in the queue
    
    def add(self, item):
        self.index %= self.size
        self.items[self.index] = item
        self.index += 1
        self.count = min(self.count + 1, self.size)
    
    def mean(self):
        return np.mean(self.items[:self.count], axis=0)
        

obstacles = np.array([[[-1.07260727, -1.57856272],
                       [-1.16650443, -0.74001908]],
                       [[-1.17740312, -1.55208812],
                       [-1.26174207, -0.71252914]],
                       [[-1.0162433,  -1.6472147],
                       [-1.25096282, -0.83673311]]])

queue = Queue(5)
for o in obstacles:
    queue.add(o)
    print(queue.mean())