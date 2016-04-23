import numpy


class DataFeeder(object):

    def __init__(self, data, batchsize=128):
        if not isinstance(data, tuple):
            data = (data,)
        self.data = data
        self.batchsize = batchsize
        self.reset()

    def reset(self):
        self.pos = 0
        self.epoch = 0
        self.perm = numpy.random.permutation(len(self.data[0]))
        self.exhaust = False

    def next_epoch(self):
        self.pos = 0
        self.epoch += 1
        self.perm = numpy.random.permutation(len(self.data[0]))
        self.exhaust = True

    def __len__(self):
        return len(self.data[0])

    def get_minibatch(self, batchsize=None):
        self.exhaust = False
        if batchsize is None:
            batchsize = self.batchsize
        indices = self.perm[self.pos: self.pos + batchsize]
        ret = [d[indices] for d in self.data]
        if len(ret) == 1:
            ret = ret[0]

        self.pos += batchsize
        # assume all data have same length
        if self.pos >= len(self.data[0]):
            self.next_epoch()

        return ret, indices
