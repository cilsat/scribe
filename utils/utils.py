#!/usr/bin/python3

import pandas as pd
import numpy as np
from numpy.linalg import norm
from multiprocessing import Pool, cpu_count
from subprocess import run, PIPE

def apply_parallel(func, args):
    """
    Parallelize functions applied to dataframe groups for python 3.
    This function is suboptimal for multi-indexed groups ie. dataframes
    grouped on more than one axis, due to pickling.
    """
    with Pool(cpu_count()) as p:
        ret = p.starmap(func, args)
    return pd.concat(ret)

def lb_keough_md(x, y, r=40):
    """
    Lowerbound Keough algorithm to calculate DTW distance of (possibly multi-D)
    signals in linear time
    """
    big_y = len(y) > len(x)
    a = norm(x, axis=1) if big_y else norm(y, axis=1)
    b = norm(y, axis=1) if big_y else norm(x, axis=1)

    lb_sum = 0
    for n, i in enumerate(a):
        win = b[(n - r if n - r >= 0 else 0):(n + r)]
        lb = win.min()
        ub = win.max()
        if i > ub: lb_sum += (i - ub)**2
        elif i < lb: lb_sum += (i - lb)**2
    return lb_sum**0.5

def lb_keough(x, y, r=40):
    """
    Lowerbound Keough algorithm to calculate DTW distance of (possibly multi-D)
    signals in linear time
    """
    lb_sum = 0
    for n, i in enumerate(x):
        win = y[(n - r if n - r >= 0 else 0):(n + r)]
        lb = min(win)
        ub = max(win)
        if i > ub: lb_sum += (i - ub)**2
        elif i < lb: lb_sum += (i - lb)**2
    return lb_sum**0.5

