#!/usr/bin/python3

import pandas as pd
from multiprocessing import Pool, cpu_count

# parallelize functions applied to dataframe groups
# it should be noted that this is *very inefficient* for multi-indexed groups
# i.e. dataframes grouped on more than one axis due to pickling
def apply_parallel(func, args):
    with Pool(cpu_count()) as p:
        ret = p.starmap(func, args)
    return pd.concat(ret)

