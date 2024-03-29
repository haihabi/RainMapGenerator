import numpy as np


class ResultsAveraging(object):
    def __init__(self):
        self.batch_acc = dict()
        self.n = 0
        self.val = None

    def is_best(self, val):
        if self.val is None:
            self.val = val
            status = True
        elif self.val < val:
            status = False
        else:
            self.val = val
            status = True
        return status

    def reset(self):
        self.n = 0
        self.batch_acc = dict()

    def results(self):
        res = {k: np.sum(v) / self.n for k, v in self.batch_acc.items()}
        self.reset()
        return res

    def update_results(self, input_results_dict: dict, n=1):
        self.n += 1
        for k, v in input_results_dict.items():
            if self.batch_acc.get(k) is None:
                self.batch_acc.update({k: []})
            self.batch_acc.get(k).append(v)

    def end_epoch(self, extra_results):
        pass
