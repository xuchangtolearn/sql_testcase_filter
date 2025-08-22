import random
from typing import List


class BaseFuzzer:

    def __init__(self, elements: List, p: float, max_l0: float = float('inf')):
        self.elements = [e for e in elements if e is not None]
        self.p = p if len(self.elements) != 0 else 1

        self.max_l0 = max_l0
        self.rand_elements = []

    def sample(self, fuzzing, num_samples=1):
        if fuzzing:
            return [self.one_sample() for _ in range(num_samples)]
        else:
            return random.choices(self.elements, k=num_samples)

    def one_sample(self):
        raise NotImplementedError

