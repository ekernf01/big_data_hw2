import math

from kdtree.HPoint import HPoint


class KDNode(object):
    """
    K-D Tree node class
    @ivar key: HPoint
    @ivar value: Object
    @ivar left: KDNode
    @ivar right: KDNode
    @ivar deleted: Boolean
    """

    def __init__(self, key, value):
        """
        Constructor. This is used only by class; other methods are static
        @param key: HPoint
        @param value: Object
        """
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.deleted = False


    @classmethod
    def ins(cls, key, value, kdnode, level, K):
        """
        Insert KDNode into KDTree
        Method ins translated from 352.ins.c of Gonnet & Baeza-Yates
        @param key: HPoint
        @param value: Object
        @param kdnode: KDNode
        @param level: int
        @param K: int
        """
        if kdnode is None:
            kdnode = KDNode(key, value)
        elif key == kdnode.key:
            # "re-insert"
            if kdnode.deleted:
                kdnode.deleted = False
                kdnode.value = value
            #else:
            #    raise Exception("Key Duplicate Exception")
        elif key.coord[level] > kdnode.key.coord[level]:
            kdnode.right = cls.ins(key, value, kdnode.right, (level + 1) % K, K)
        else:
            kdnode.left = cls.ins(key, value, kdnode.left, (level + 1) % K, K)
        return kdnode


    @classmethod
    def srch(cls, key, t, K):
        """
        Search for
        Method srch translated from 352.srch.c of Gonnet & Baeza-Yates
        @param key: HPoint
        @param t: KDNode
        @param K: int
        """
        level = 0
        while t is not None:
            if t.deleted and key == t.key:
                return t
            elif key.coord[level] > t.key.coord[level]:
                t = t.right
            else:
                t = t.left
            level = (level + 1) % K
        return None


    @classmethod
    def rsearch(cls, lowk, uppk, t, level, K, v):
        """
        Method rsearch translated from 352.range.c of Gonnet & Baeza-Yates
        @param lowk: HPoint
        @param uppk: HPoint
        @param t: KDNode
        @param level: int
        @param K: int
        @param v: [KDNode]
        """
        if t is None:
            return
        if lowk.coord[level] <= t.key.coord[level]:
            cls.rsearch(lowk, uppk, t.left, (level + 1) % K, K, v)
        j = 0
        while j < K and lowk.coord[j] <= t.key.coord[j] and uppk.coord[j] >= t.key.coord[j]:
            j+=1
        if j == K:
            v.append(t)
        else:
            cls.rsearch(lowk, uppk, t.right, (level + 1) % K, K, v)


    @classmethod
    def nnbr(cls, kdnode, target, hrect, max_dist_sqd, level, K, nnl, alpha):
        """
        Method Nearest Neighbor from Andrew Moore's thesis. Numbered
        comments are direct quotes from there. Step "SDL" is added to
        make the algorithm work correctly. NearestNeighborList solution
        courtesy of Bjoern Heckel.
        @param kdnode: KDNode
        @param target: HPoint
        @param hrect: HRect
        @param max_dist_sqd: float
        @param level: int
        @param K: int
        @param nnl: NearestNeighborList
        @param alpha: float
        """
        # 1. if kd is empty then set dist-sqd to infinity and exit.
        if kdnode is None:
            return

        # 2. s := split field of kd
        split = level % K

        # 3. pivot := dom-elt field of kd
        pivot = kdnode.key;
        pivot_to_target = HPoint.sqrdist(pivot, target);

        # 4. Cut hr into to sub-hyperrectangles left-hr and right-hr.
        # The cut plane is through pivot and perpendicular to the s
        # dimension.
        left_hrect = hrect    # optimize by not cloning
        right_hrect = hrect.clone()
        left_hrect.max.coord[split] = pivot.coord[split]
        right_hrect.min.coord[split] = pivot.coord[split]

        # 5. target-in-left := target_s <= pivot_s
        target_in_left = target.coord[split] < pivot.coord[split]

        # 6. if target-in-left then
        # 6.1. nearer-kd := left field of kd and nearer-hr := left-hr
        # 6.2. further-kd := right field of kd and further-hr := right-hr
        if target_in_left:
            nearer_kdnode = kdnode.left
            nearer_hrect = left_hrect
            further_kdnode = kdnode.right
            further_hrect = right_hrect
        # 7. if not target-in-left then
        # 7.1. nearer-kd := right field of kd and nearer-hr := right-hr
        # 7.2. further-kd := left field of kd and further-hr := left-hr
        else:
            nearer_kdnode = kdnode.right;
            nearer_hrect = right_hrect;
            further_kdnode = kdnode.left;
            further_hrect = left_hrect;

        # 8. Recursively call Nearest Neighbor with parameters
        # (nearer-kd, target, nearer-hr, max-dist-sqd), storing the
        # results in nearest and dist-sqd
        cls.nnbr(nearer_kdnode, target, nearer_hrect, max_dist_sqd, level + 1, K, nnl, alpha)

        # furthest node in acceptable set
        if not nnl.is_capacity_reached():
            dist_sqd = float("inf")
        else:
            # furthest distance in accepted set
            dist_sqd = nnl.get_max_priority()

        # 9. max-dist-sqd := minimum of max-dist-sqd and dist-sqd
        if dist_sqd < max_dist_sqd:
            max_dist_sqd = dist_sqd

        # 10. A nearer point could only lie in further-kd if there were some
        # part of further-hr within distance sqrt(max-dist-sqd) of
        # target. If this is the case then
        # CHAD: Scale this comparison by alpha?
        closest = further_hrect.closest(target)
        if HPoint.eucdist(closest, target) < math.sqrt(max_dist_sqd) / alpha:
            # 10.1 if (pivot-target)^2 < dist-sqd then
            if pivot_to_target < dist_sqd:
                # 10.1.1 nearest := (pivot, range-elt field of kd)

                # 10.1.2 dist-sqd = (pivot-target)^2
                dist_sqd = pivot_to_target

                # add to nnl
                if not kdnode.deleted:
                    nnl.insert(kdnode, dist_sqd)

                # 10.1.3 max-dist-sqd = dist-sqd
                # max_dist_sqd = dist_sqd;
                if nnl.is_capacity_reached():
                    max_dist_sqd = nnl.get_max_priority()
                else:
                    max_dist_sqd = float("inf")

            # 10.2 Recursively call Nearest Neighbor with parameters
            # (further-kd, target, further-hr, max-dist_sqd),
            # storing results in temp-nearest and temp-dist-sqd
            cls.nnbr(further_kdnode, target, further_hrect, max_dist_sqd, level + 1, K, nnl, alpha)
            temp_dist_sqd = nnl.get_max_priority()

            # 10.3 If tmp-dist-sqd < dist-sqd then
            if temp_dist_sqd < dist_sqd:
                # 10.3.1 nearest := temp_nearest and dist_sqd := temp_dist_sqd
                dist_sqd = temp_dist_sqd
        # SDL: otherwise, current point is nearest
        elif pivot_to_target < max_dist_sqd:
            #nearest = kdnode
            dist_sqd = pivot_to_target


    def pad(self, n):
        s = ""
        for _ in range(n):
            s += " "
        return s


    def strdepth(self, depth):
        if self.deleted:
            delstring = "*"
        else:
            delstring = ""
        s = str(self.key) + " " + str(self.value) + delstring
        if self.left is not None:
            s += "\n" + self.pad(depth) + "L " +  self.left.strdepth(depth + 1)
        if self.right is not None:
            s += "\n" + self.pad(depth) + "R " + self.right.strdepth(depth + 1)
        return s


    def __str__(self):
        return self.stringHelper(0)


    @classmethod
    def hrcopy(cls, hrect_src, hrect_dst):
        """
        @param hrect_src: HRect
        @param hrect_dst: HRect
        """
        cls.hpcopy(hrect_src.min, hrect_dst.min)
        cls.hpcopy(hrect_src.max, hrect_dst.max)


    @classmethod
    def hpcopy(cls, hpoint_src, hpoint_dst):
        """
        @param hpoint_src: HPoint
        @param hpoint_dst: HPoint
        """
        for i, val in range(hpoint_src):
            hpoint_dst.coord[i] = val
