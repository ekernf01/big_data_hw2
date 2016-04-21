from kdtree.HPoint import HPoint
from kdtree.HRect import HRect
from kdtree.KDNode import KDNode
from kdtree.NearestNeighborList import NearestNeighborList


class KDTree(object):
    """
    KDTree is a class supporting KD-tree insertion, deletion, equality search,
    range search, and nearest neighbor(s) using double-precision floating-point
    keys. Splitting dimension is chosen naively, by depth modulo K. Semantics are
    as follows:
    * Two different keys containing identical numbers should retrieve the same
      value from a given KD-tree. Therefore keys are cloned when a node is
      inserted.
    * As with Hashtables, values inserted into a KD-tree are <I>not</I> cloned.
      Modifying a value between insertion and retrieval will therefore modify the
      value stored in the tree.
    @ivar m_K: int - number of dimensions
    @ivar m_root: KDNode - root of KD-Tree
    @ivar m_count: int - count of nodes
    """

    def __init__(self, K):
        """
        Creates a KD-tree with specified number of dimensions.
        @param K: int
        """
        self.m_K = K
        self.m_root = None
        self.m_count = 0


    def insert(self, key, value):
        """
        Insert a node in a KD-tree. Uses algorithm translated from 352.ins.c of
        Book{GonnetBaezaYates1991,
            author =    {G.H. Gonnet and R. Baeza-Yates},
            title =     {Handbook of Algorithms and Data Structures},
            publisher = {Addison-Wesley},
            year =      {1991}
        }
        @param key: [float] - key for KD-tree node
        @param value: Object - value at the key
        """
        if len(key) != self.m_K:
            raise Exception("KDTree: wrong key size!")
        else:
            self.m_root = KDNode.ins(HPoint(key), value, self.m_root, 0, self.m_K)
        self.m_count += 1


    def search(self, key):
        """
        Find KD-tree node whose key is identical to key. Uses algorithm
        translated from 352.srch.c of Gonnet & Baeza-Yates.
        @param key: [float] - key for KD-tree node
        @return: Object - object at key, or None if not found
        """
        if len(key) != self.m_K:
            raise Exception("KDTree: wrong key size!")
        kdnode = KDNode.srch(HPoint(key), self.m_root, self.m_K)
        if kdnode is None:
            return None
        else:
            return kdnode.value


    def delete(self, key):
        """
        Delete a node from a KD-tree. Instead of actually deleting node and
        rebuilding tree, marks node as deleted. Hence, it is up to the caller to
        rebuild the tree as needed for efficiency.
        @param key: [float] - key for KD-tree node
        """
        if len(key) != self.m_K:
            raise Exception("KDTree: wrong key size!")
        kdnode = KDNode.srch(HPoint(key), self.m_root, self.m_K)
        if kdnode is None:
            raise Exception("KDTree: key missing!")
        else:
            kdnode.deleted = True
        self.m_count -= 1


    def nearest(self, key, alpha):
        """
        Find KD-tree node whose key is nearest neighbor to key. Implements the
        Nearest Neighbor algorithm (Table 6.4) of
        techreport{AndrewMooreNearestNeighbor,
            author  = {Andrew Moore},
            title   = {An introductory tutorial on kd-trees},
            institution = {Robotics Institute, Carnegie Mellon University},
            year    = {1991},
            number  = {Technical Report No. 209, Computer Laboratory,
                       University of Cambridge},
            address = {Pittsburgh, PA}
        }
        @param key: [float] - key for KD-tree node
        @param alpha: float - alpha for approximate k-nn
        @return: Object - nearest element
        """
        nbrs = self.nearest_k(key, 1, alpha)
        return nbrs[0]


    def nearest_k(self, key, k, alpha):
        """
        Find KD-tree nodes whose keys are k nearest neighbors to key. Uses
        algorithm above. Neighbors are returned in ascending order of distance to
        key.
        @param key: [float] - key for KD-tree node
        @param k: int - how many neighbors to find
        @param alpha: float - alpha for approximate k-nn
        """
        if k < 0 or k > self.m_count:
            raise Exception("Number of neighbors (" + str(k) + ") cannot be negative or greater than number of nodes (" + str(self.m_count) + ").")
        if len(key) != self.m_K:
            raise Exception("KDTree: wrong key size!")

        nbrs = [None for _ in range(k)]
        nnl = NearestNeighborList(k)

        # initial call is with infinite hyper-rectangle and max distance
        hrect = HRect.infinite_hrect(len(key))
        max_dist_sqd = float("inf")
        keypoint = HPoint(key)

        KDNode.nnbr(self.m_root, keypoint, hrect, max_dist_sqd, 0, self.m_K, nnl, alpha)

        for i in range(k):
            kdnode = nnl.remove_highest()
            nbrs[k-i-1] = kdnode.value

        return nbrs


    def range(self, lowk, uppk):
        """
        Range search in a KD-tree. Uses algorithm translated from 352.range.c of
        Gonnet & Baeza-Yates.
        @param lowk: [float] - lower-bounds for key
        @param uppk: [float] - upper-bounds for key
        @return: [Object] - array of objects whose keys fall in range [lowk, uppk]
        """
        if len(lowk) != len(uppk):
            raise Exception("KDTree: wrong key size!")
        elif len(lowk) != self.m_K:
            raise Exception("KDTree: wrong key size!")
        else:
            kdnodes = []
            KDNode.rsearch(lowk, uppk, self.m_root, 0, self.m_K, kdnodes)
            values = []
            for kdnode in kdnodes:
                values.append(kdnode.value)
            return values


    def __str__(self):
        return str(self.m_root)
