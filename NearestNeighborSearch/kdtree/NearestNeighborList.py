import heapq

class NearestNeighborList:
    """
    List of nearest neighbors...
    @ivar m_queue: [(value, object)] - a heap/priority queue 
    @ivar m_capacity: int
    """
    
    def __init__(self, capacity):
        """
        @param capacity: int
        """
        self.m_queue = []
        self.m_capacity = capacity
        
    def get_max_priority(self):
        if not len(self.m_queue):
            return float("inf")
        return -self.m_queue[0][0]
        # return m_queue.get_max_priority()
        
    def insert(self, obj, priority):
        if len(self.m_queue) < self.m_capacity:
            # capacity not reached
            heapq.heappush(self.m_queue, (-priority, obj))
            return True
        if priority > -self.m_queue[0][0]:
            return False
        # remove object with highest priority
        heapq.heappop(self.m_queue)
        # add new object
        heapq.heappush(self.m_queue, (-priority, obj))
        return True
    
    def is_capacity_reached(self):
        return len(self.m_queue) >= self.m_capacity
    
    def get_highest(self):
        return self.m_queue[0][1]
    
    def is_empty(self):
        return len(self.m_queue) is 0
    
    def get_size(self):
        return len(self.m_queue)
    
    def remove_highest(self):
        # remove object with highest priority
        return heapq.heappop(self.m_queue)[1]