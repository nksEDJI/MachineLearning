from re import I
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob2 import glob
import os
import re
folder = 'machinelearning/1st_test/1st_test/*'
path = 'machinelearning/1st_test/1st_test/'


file = glob(folder)
print(file[0])

for i in file:
    print(i)
    df = pd.read_csv(i, sep = '\t')
#tirar os valores como se fossem variaveis

    print(df)    

    break

