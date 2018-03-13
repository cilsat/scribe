"""Module that implements the Buffer class."""
import numpy as np


class Buffer(object):
    """
    A class to manage the data in a fixed size FIFO buffer.

    Automatically accounts for where new samples are placed in the buffer when
    pushing, and where old samples are relocated when popping.
    """

    def __init__(self, x, y=None):
        """Initialize class with dimensions of buffer."""
        self.len = x
        if y:
            self.size = x * y
            self.data = np.empty((x, y))
        else:
            self.size = self.len
            self.data = np.empty((x,))
        self.idx = 0

    def push(self, samples):
        """Add (multi-dimensional) samples to buffer."""
        len_s = len(samples)
        if self.idx + len_s < self.len:
            self.data[self.idx:self.idx + len_s] = samples
            self.idx += len_s
        else:
            if self.idx == self.len:
                self.data[:-len_s] = self.data[len_s:]
            else:
                self.data[:-len_s] = self.data[len_s -
                                               self.len + self.idx:self.idx]
                self.idx = self.len
            self.data[-len_s:] = samples

    def pop(self, idx=None):
        """Pop a number of samples from buffer."""
        if not idx:
            samples = np.copy(self.data[:self.idx])
            self.data[:] = np.empty(self.data.shape)
            self.idx = 0
        else:
            if idx > self.idx:
                raise ValueError()
            samples = np.copy(self.data[:idx])
            data = np.copy(self.data[idx:self.idx])
            self.data[:] = np.empty(self.data.shape)
            self.data[:self.idx - idx] = data
            self.idx -= idx
        return samples

    def get_data(self):
        return self.data[:self.idx]

    def is_full(self):
        """Return whether the buffer is full."""
        return self.idx == self.len

    def is_empty(self):
        """Return whether the buffer is empty."""
        return self.idx == 0


class ProcBuffer(Buffer):
    def __init__(self, x, y, thr, dur):
        Buffer.__init__(self, x, y)
        self.thr = thr
        self.dur = dur
        self.num = 0
        self.can = False

    def eval_buffer(self, evaluator):
        if not self.is_full():
            return
        elif evaluator(self.data) < self.thr:
            self.num += 1
            if self.can and self.num > self.dur:
                self.can = False
                return
            else:
                return
        else:
            self.num = 0
            self.can = True
            return
