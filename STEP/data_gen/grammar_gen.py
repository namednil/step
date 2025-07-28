import dataclasses
from typing import Callable, Optional, Union

import random

import numpy as np


def weighted_choice(vec):
    """
    Return an index of vec with probability proportional to the entries
    """
    Z = sum(vec)
    rnd = random.random() * Z
    s = 0
    for i in range(len(vec)):
        s += vec[i]
        if s >= rnd:
            return i

@dataclasses.dataclass
class ProductionRule:
    lhs: int
    fname: str
    fint: int # fname as int
    map_term: str
    rhs: list[Union[str, int]]




