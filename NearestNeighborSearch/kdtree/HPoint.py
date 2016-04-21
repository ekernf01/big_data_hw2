import math


class HPoint(object):
    """
    Hyper-Point class supporting KDTree class
    @ivar coord: [float]
    """

    def __init__(self, x):
        """
        Constructor
        @param x: [float] or int: possible initialization
        """
        if isinstance(x, int):
            self.coord = [0.0 for _ in range(x)]
        elif isinstance(x, list):
            self.coord = [val for val in x]
        else:
            self.coord = []


    def clone(self):
        return HPoint(self.coord)


    def __eq__(self, other):
        if len(self.coord) != len(other.coord):
            return False
        for val1, val2 in zip(self.coord, other.coord):
            if val1 != val2:
                return False
        return True


    @classmethod
    def sqrdist(cls, hpointX, hpointY):
        """
        Computes squared distance between to HPoint instances.
        @param hpointX: HPoint
        @param hpointY: HPoint
        """
        dist= 0.0
        for val1, val2 in zip(hpointX.coord, hpointY.coord):
            diff = val1 - val2
            dist += diff * diff
        return dist


    @classmethod
    def eucdist(cls, hpointX, hpointY):
        """
        Computes Euclidean distance between to HPoint instances.
        @param hpointX: HPoint
        @param hpointY: HPoint
        """
        return math.sqrt(cls.sqrdist(hpointX, hpointY))


    def __str__(self):
        return " ".join(str(x) for x in self.coord)
