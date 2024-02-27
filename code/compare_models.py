"""
Author:       Eva Nieuwenhuis
University:   UvA
Student id':  13717405
Description:  There are two types of models made that both simulate the dynamics
              between the drug-sensitive MM cells (MMd), resistant MM cells (MMr),
              osteoblasts (OBs) and osteoclasts (OCs) in the multiple myeloma (MM)
              microenvironment. One model (MM_model_fractions.py) simulates the cell
              fractions and the other model (MM_model_numbers.py) simulates the cell
              numbers. Here there is attemped to equalize the results of both models
              by ............
"""

# Import the needed libraries
import math
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import csv
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import doctest

"""
Example payoff matrix:
M = np.array([
       Goc Gob Gmmd Gmmr
    OC  [a,  b,  c,  d],
    OB  [e,  f,  g,  h],
    MMd [i,  j,  k,  l],
    MMr [m,  n,  o,  p]])
"""
