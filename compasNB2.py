# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:56:36 2018

@author: TapperR
"""

import numpy as np
#data manipulation
import pandas as pd
#matrix data structure
from patsy import dmatrices
#for error logging
import warnings


import time
start_time = time.time()


import os 
#os.chdir('C:\\Users\\Robin\\Desktop\\compas-analysis')
os.chdir('C:\\Users\\TapperR\\Desktop\\compas\\compas-analysis')

data = pd.read_csv('spam.csv')