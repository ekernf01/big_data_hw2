class TestResult(object):
    """
    Test Result
    @ivar method: string
    @ivar n: int
    @ivar D: int
    @ivar alpha: float
    @ivar avg_time: float
    @ivar avg_distance: float
    """
    def __init__(self, method, n, D, alpha, avg_time, avg_distance):
        self.method = method
        self.n = n
        self.D = D
        self.alpha = alpha
        self.avg_time = avg_time
        self.avg_distance = avg_distance
