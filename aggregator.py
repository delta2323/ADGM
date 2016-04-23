class Aggregator(object):

    def __init__(self):
        self.reset()

    def mean(self):
        return self.sum / self.n

    def reset(self):
        self.sum = 0.0
        self.n = 0
