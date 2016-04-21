from kdtree.HPoint import HPoint


class HRect(object):
    """
    Hyper-Rectangle class supporting KDTree class
    @ivar min: HPoint
    @ivar max: HPoint
    """

    def __init__(self, vmin, vmax):
        if vmin:
            self.min = vmin.clone()
        if vmax:
            self.max = vmax.clone()


    def clone(self):
        return HRect(self.min, self.max)


    def closest(self, hpoint):
        """
        Return point closest to HRect (from Moore's eqn. 6.6)
        @param hpoint: HPoint
        """
        p = HPoint(len(hpoint.coord))
        for i, val in enumerate(hpoint.coord):
            if val <= self.min.coord[i]:
                p.coord[i] = self.min.coord[i]
            elif val >= self.max.coord[i]:
                p.coord[i] = self.max.coord[i]
            else:
                p.coord[i] = val
        return p


    @classmethod
    def infinite_hrect(cls, dim):
        """
        Create an infinite rectangle. Used in the initial conditions of KDTree.nearest()
        """
        vmin = HPoint(dim)
        vmax = HPoint(dim)
        vmin.coord = [-float("inf") for _ in range(dim)]
        vmax.coord = [float("inf") for _ in range(dim)]
        return HRect(vmin, vmax)


    def __str__(self):
        return str(self.min) + "\n" + str(self.max) + "\n"
