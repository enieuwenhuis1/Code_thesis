"""
Author:            Eva Nieuwenhuis
University group:  Biosystems Data Analysis Group, UvA
Student ID:        13717405

Description:  The code of the model that simulates the dynamics in the multiple
              myeloma (MM) microenvironment with four cell types: drug-sensitive
              MM cells (MMd), resistant MM cells (MMr), osteoblasts (OB) and
              osteoclasts (OC). The model is a public goods game in the framework
              of evolutionary game theory with collective interactions. In this
              model, there is looked at the numbers of the four cell types. The
              IHs have not only an influence on the MMd but also on the OB and OC.

Example interaction matrix:
M = np.array([
       Foc Fob Fmmd Fmmr
    OC  [a,  b,  c,  d],
    OB  [e,  f,  g,  h],
    MMd [i,  j,  k,  l],
    MMr [m,  n,  o,  p]])
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

def main():
    # Do doc tests
    doctest.testmod()

    # # Make a figure showing the cell number dynamics by traditional therapy and
    # # by adaptive therapy (original situation)
    # list_t_steps_drug = [10, 10, 10]
    # Figure_continuous_MTD_vs_AT(20, list_t_steps_drug)
    #
    # # Make a figure showing the cell number dynamics by traditional therapy and
    # # by adaptive therapy for shorter holiday and administration periods compared
    # # to the original situation
    # list_t_steps_drug = [5, 5, 5]
    # Figure_continuous_MTD_vs_AT_short_a_h(40, list_t_steps_drug)
    #
    # # Make a figure showing the cell number dynamics by traditional therapy and
    # # by adaptive therapy for weaker IHs compared to the original situation
    # list_t_steps_drug = [10, 10, 10]
    # Figure_continuous_MTD_vs_AT_weak_a_h(20, list_t_steps_drug)
    #
    # # Make a figure showing the cell number dynamics by traditional therapy and
    # # by adaptive therapy
    # list_t_steps_drug = [3, 3, 3]
    # Figure_continuous_MTD_vs_AT_realistic(90, list_t_steps_drug)
    #
    # # Make a 3D figure showthing the effect of different drug holiday and
    # # administration periods
    # Figure_3D_MM_numb_IH_add_and_holiday()
    #
    # # Make a figure that shows the MM number for different bOC,MMd values
    # Figure_best_b_OC_MMd()
    #
    # # Make a figure that shows the MM number for different WMMd IH values
    # Figure_best_WMMd_IH()
    #
    # # Make a 3D figure showing the effect of different WMMd and MMd GF IH
    # # strengths
    # Figure_3D_MM_numb_MMd_IH_strength()
    #
    # # Make line plots showing the dynamics when the IH administration is longer
    # # than the holiday and one it is the other way around.
    # list_t_steps_drug = [3, 10]
    # list_t_steps_no_drug = [10, 3]
    # list_n_steps = [40, 40]
    # Figure_duration_a_h_MMd_IH(list_n_steps, list_t_steps_drug,
    #                                                       list_t_steps_no_drug)
    #
    # # Make a figure of the number dynamics whereby there is a low limit for the
    # # MMd and MMr number. The last paramter is low, middel or high-> low: nMMr
    # # limit < 400, middel: nMMr limit 400-800, high: MMr limit > 800
    # Figure_AT_MMd_MMr_limit(300, 150, 'low')
    #
    # # Make a figure of the number dynamics whereby there is a limit for the MMd
    # # and MMr number. The last paramter is low, middel or high-> low: nMMr limit
    # # < 400, middel: nMMr limit 400-800, high: MMr limit > 800
    # Figure_AT_MMd_MMr_limit(500, 250, 'middel')
    #
    # # Make a figure of the number dynamics whereby there is a high limit for the
    # # MMd and MMr number. The last paramter is low, middel or high-> low: nMMr
    # # limit < 400, middel: nMMr limit 400-800, high: MMr limit > 800
    # Figure_AT_MMd_MMr_limit(1200, 700, 'high')

    # """ The unweighted optimisation situations """
    # # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # # holiday
    # minimise_MM_GF_W_h()
    #
    # # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # # holiday
    # minimise_MM_W_GF_h()
    # #
    # # Optimise IH administration duration, holiday duration and strength for
    # # MMd GF IH -> WMMd IH -> holiday
    # minimise_MM_GF_W_h_IH()
    #
    # # Optimise IH administration duration, holiday duration and strength for
    # # WMMd IH -> MMd GF IH ->  holiday
    # minimise_MM_W_GF_h_IH()
    #
    # # Optimise IH administration duration, holiday duration and strength for
    # # MMd GF IH -> holiday -> WMMd IH -> holiday
    # minimise_MM_GF_h_W_h_IH()
    #
    # # Optimise IH administration duration, holiday duration and strength for
    # # WMMd IH -> holiday -> MMd GF IH ->  holiday
    # minimise_MM_W_h_GF_h_IH()
    #
    # # Optimise IH administration duration and holiday duration for MMd GF IH
    # # -> IH combination -> WMMd IH -> holiday
    # minimise_MM_GF_comb_W_h()
    #
    # # Optimise IH administration duration and holiday duration for WMMd IH ->
    # # IH combination -> MMd GF IH -> holiday
    # minimise_MM_W_comb_GF_h()
    #
    # # Optimise IH administration duration, holiday duration and strengths for
    # # MMd GF IH -> IH combination -> WMMd IH -> holiday
    # minimise_MM_GF_comb_W_h_IH()
    #
    # # Optimise IH administration duration, holiday duration and strengths for
    # # WMMd IH -> IH combination -> MMd GF IH -> holiday
    # minimise_MM_W_comb_GF_h_IH()
    #
    # # Optimise IH administration duration and holiday duration for MMd GF IH ->
    # # WMMd IH + MMd GF IH -> WMMd IH -> holiday
    # minimise_MM_GF_GFandW_W_h()
    #
    # # Optimise IH administration duration and holiday duration for WMMd IH ->
    # # WMMd IH + MMd GF IH -> MMd GF IH -> holiday
    # minimise_MM_W_WandGF_GF_h()
    #
    # # Optimise IH administration duration, holiday duration and strengths for
    # # MMd GF IH -> WMMd IH + MMd GF IH -> WMMd IH -> holiday
    # minimise_MM_GF_GFandW_W_h_IH()
    #
    # # Optimise IH administration duration, holiday duration and strengths for
    # # WMMd IH -> WMMd IH + MMd GF IH -> MMd GF IH -> holiday
    # minimise_MM_W_WandGF_GF_h_IH()
    #
    # # Optimise IH administration duration, holiday duration and strengths for
    # # MMd GF IH -> IH combination -> holiday
    # minimise_MM_GF_comb_h_IH()
    # #
    # # Optimise IH administration duration, holiday duration and strengths for
    # # WMMd IH -> IH combination -> holiday
    # minimise_MM_W_comb_h_IH()

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different WMMd IH strengths and MMd GF IH = 0.4
    minimise_MM_GF_W_h_changing_W_IH([0.8, 1.2, 0.3, 0.3], [0.7, 1.3, 0.3, 0.3],
    [0.9, 0.08, 0.2, 0.1], [1.0, 0.08, 0.2, 0.1], 'df_MM_GF_W_h_changing_W_IH.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different WMMd IH strengths and MMd GF IH = 0.4
    minimise_MM_W_GF_h_changing_W_IH([0.8, 1.2, 0.3, 0.3], [0.7, 1.3, 0.3, 0.3],
    [0.9, 0.08, 0.2, 0.1], [1.0, 0.08, 0.2, 0.1], 'df_MM_W_GF_h_changing_W_IH.csv')

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different MMd GF IH strengths and WMMd IH = 0.4
    minimise_MM_GF_W_h_changing_GF_IH([0.8, 1.2, 0.3, 0.3], [0.7, 1.3, 0.3, 0.3],
    [0.9, 0.08, 0.2, 0.1], [1.0, 0.08, 0.2, 0.1], 'df_MM_GF_W_h_changing_GF_IH.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different MMd GF IH strengths and WMMd IH = 0.4
    minimise_MM_W_GF_h_changing_GF_IH([0.8, 1.2, 0.3, 0.3], [0.7, 1.3, 0.3, 0.3],
    [0.9, 0.08, 0.2, 0.1], [1.0, 0.08, 0.2, 0.1], 'df_MM_W_GF_h_changing_GF_IH.csv')

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different WMMd IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_GF_W_h_changing_W_IH([0.88, 1.32, 0.33, 0.33], [0.77, 1.43, 0.33,
                       0.33], [0.99, 0.088, 0.22, 0.11], [1.1, 0.088, 0.22, 0.11],
                                        'df_MM_GF_W_h_changing_W_IH_h_gr_dr.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different WMMd IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_W_GF_h_changing_W_IH([0.88, 1.32, 0.33, 0.33], [0.77,1.43, 0.33,
                      0.33], [0.99, 0.088, 0.22, 0.11], [1.1, 0.088, 0.22, 0.11],
                                        'df_MM_W_GF_h_changing_W_IH_h_gr_dr.csv')

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different MMd GF IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_GF_W_h_changing_GF_IH([0.88, 1.32, 0.33, 0.33], [0.77,1.43, 0.33,
                      0.33], [0.99, 0.088, 0.22, 0.11], [1.1, 0.088, 0.22, 0.11],
                                        'df_MM_GF_W_h_changing_GF_IH_h_gr_dr.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different MMd GF IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_W_GF_h_changing_GF_IH([0.88, 1.32, 0.33, 0.33], [0.77, 1.43, 0.33,
                       0.33], [0.99, 0.088, 0.22, 0.11], [1.1, 0.088, 0.22, 0.11],
                                      'df_MM_W_GF_h_changing_GF_IH_h_gr_dr.csv')

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different WMMd IH strengths whereby the growth and decay rate are
    # decreased with 10%
    minimise_MM_GF_W_h_changing_W_IH([0.72, 1.08, 0.27, 0.27], [0.63, 1.17, 0.27,
                      0.27], [0.81, 0.072, 0.18, 0.09], [0.9, 0.072, 0.18, 0.09],
                                        'df_MM_GF_W_h_changing_W_IH_l_gr_dr.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different WMMd IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_W_GF_h_changing_W_IH([0.72, 1.08, 0.27, 0.27], [0.63, 1.17, 0.27,
                      0.27], [0.81, 0.072, 0.18, 0.09], [0.9, 0.072, 0.18, 0.09],
                                        'df_MM_W_GF_h_changing_W_IH_l_gr_dr.csv')

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different MMd GF IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_GF_W_h_changing_GF_IH([0.72, 1.08, 0.27, 0.27], [0.63, 1.17, 0.27,
                    0.27], [0.81, 0.072, 0.18, 0.09], [0.9, 0.072, 0.18, 0.09],
                                    'df_MM_GF_W_h_changing_GF_IH_l_gr_dr.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different MMd GF IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_W_GF_h_changing_GF_IH([0.72, 1.08, 0.27, 0.27], [0.63, 1.17, 0.27,
                       0.27], [0.81, 0.072, 0.18, 0.09], [0.9, 0.072, 0.18, 0.09],
                                    'df_MM_W_GF_h_changing_GF_IH_l_gr_dr.csv')

    # Make a figure of the MM number after optimisation by different IH strengths
    Figure_optimisation()

    """ The weighted optimisation situations """
    # Optimise IH administration and holiday duration and strength for MMd GF IH
    # -> WMMd IH -> holiday where the weight of the MMr relative to the MMd can
    # be specified
    relative_weight_MMr = 1.2
    minimise_MM_GF_W_h_IH_w(relative_weight_MMr)

    # Optimise IH administration and holiday duration and strength for WMMd IH ->
    # MMd GF IH -> holiday where the weight of the MMr relative to the MMd can be
    # specified
    relative_weight_MMr = 1.2
    minimise_MM_W_GF_h_IH_w(relative_weight_MMr)

    # Optimise IH administration duration, holiday duration and strength for MMd
    # GF IH -> holiday -> WMMd IH -> holiday where the weight of the MMr relative
    # to the MMd can be specified
    relative_weight_MMr = 1.2
    minimise_MM_GF_h_W_h_IH_w(relative_weight_MMr)

    # Optimise IH administration duration, holiday duration and strength for WMMd
    # IH -> holiday -> MMd GF IH -> holiday where the weight of the MMr relative
    # to the MMd can be specified
    relative_weight_MMr = 1.2
    minimise_MM_W_h_GF_h_IH_w(relative_weight_MMr)

    # Optimise IH administration and holiday duration and strengths for MMd GF IH
    # -> IH combination -> WMMd IH -> holiday where the weight of the MMr relative
    # to the MMd can be specified
    relative_weight_MMr = 1.2
    minimise_MM_GF_comb_W_h_IH_w(relative_weight_MMr)

    # Optimise IH administration and holiday duration and strengths for WMMd IH
    # -> IH combination -> MMd GF IH -> holiday where the weight of the MMr
    # relative to the MMd can be specified
    relative_weight_MMr = 1.2
    minimise_MM_W_comb_GF_h_IH_w(relative_weight_MMr)

    # Optimise IH administration duration, holiday duration and strengths for MMd
    # GF IH -> WMMd IH + MMd GF IH -> WMMd IH -> holiday where the weight of the
    # MMr relative to the MMd can be specified
    relative_weight_MMr = 1.2
    minimise_MM_GF_GFandW_W_h_IH_w(relative_weight_MMr)

    # Optimise IH administration duration, holiday duration and strengths for WMMd
    # IH -> WMMd IH + MMd GF IH -> MMd GF IH -> holiday where the weight of the
    # MMr relative to the MMd can be specified
    relative_weight_MMr = 1.2
    minimise_MM_W_WandGF_GF_h_IH_w(relative_weight_MMr)

    # Optimise IH administration duration, holiday duration and strengths for
    # MMd GF IH -> IH combination -> holiday
    relative_weight_MMr = 1.2
    minimise_MM_GF_comb_h_IH_w(relative_weight_MMr)

    # Optimise IH administration duration, holiday duration and strengths for
    # WMMd IH -> IH combination -> holiday
    relative_weight_MMr = 1.2
    minimise_MM_W_comb_h_IH_w(relative_weight_MMr)



def dOC_dt(nOC, nOB, nMMd, nMMr, gr_OC, dr_OC, matrix):
    """
    Function that calculates the change in number of osteoclasts.

    Parameters:
    -----------
    nOC: Float
         of OC.
    nOB: Float
         of OB.
    nMMd: Float
         of the MMd.
    nMMr: Float
         of the MMr.
    gr_OC: Float
        Growth rate of the OC.
    dr_OC: Float
        Dacay rate of the OC.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    change_nOC: Float
        Change in the number of OC.

    Example:
    -----------
    >>> dOC_dt(10, 20, 10, 5, 0.8, 0.4, np.array([
    ...    [0.7, 1, 2.5, 2.1],
    ...    [1, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0, -0.2, 1.2]]))
    744654.2266544278
    """
    # Extract the needed matrix values
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[0, 2]
    d = matrix[0, 3]

    # Calculate the Change on in the number of OC
    change_nOC = (gr_OC * nOC**a * nOB**b * nMMd**c * nMMr**d) - (dr_OC * nOC)
    return change_nOC

def dOB_dt(nOC, nOB, nMMd, nMMr, gr_OB, dr_OB, matrix):
    """
    Function that calculates the change in the number of osteoblast.

    Parameters:
    -----------
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    gr_OB: Float
        Growth rate of the OB.
    dr_OB: Float
        Dacay rate of the OB.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    change_nOB: Float
        Change in the number of OB.

    Example:
    -----------
    >>> dOB_dt(10, 20, 10, 5, 0.8, 0.4, np.array([
    ...    [0.7, 1, 2.5, 2.1],
    ...    [1, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0, -0.2, 1.2]]))
    1320.9296319483412
    """
    # Extract the necessary matrix values
    e = matrix[1, 0]
    f = matrix[1, 1]
    g = matrix[1, 2]
    h = matrix[1, 3]

    # Calculate the change in number of OB
    change_nOB = (gr_OB * nOC**e * nOB**f * nMMd**g * nMMr**h) - (dr_OB * nOB)
    return change_nOB

def dMMd_dt(nOC, nOB, nMMd, nMMr, gr_MMd, dr_MMd, matrix, WMMd_inhibitor = 0):
    """
    Function that calculates the change in the number of a drug-senstive MM cells.

    Parameters:
    -----------
    nOC: Float
        Number of OC.
    nOB: Float
         Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    gr_MMd: Float
        Growth rate of the MMd.
    dr_MMd: Float
        Decay rate of the MMd.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    change_nMMd: Float
        Change in the number of MMd.

    Example:
    -----------
    >>> dMMd_dt(10, 20, 10, 5, 0.8, 0.4, np.array([
    ...    [0.7, 1, 2.5, 2.1],
    ...    [1, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0, -0.2, 1.2]]))
    4198.444487046028
    """
    # Extract the necessary matrix values
    i = matrix[2, 0]
    j = matrix[2, 1]
    k = matrix[2, 2]
    l = matrix[2, 3]

    # Calculate the change in the number of MMd
    change_nMMd = (gr_MMd * nOC**i * nOB**j * nMMd**k * nMMr**l - nMMd * \
                                             WMMd_inhibitor) - (dr_MMd * nMMd)

    return change_nMMd

def dMMr_dt(nOC, nOB, nMMd, nMMr, gr_MMr, dr_MMr, matrix):
    """
    Function that calculates the change in the number of the MMr.

    Parameters:
    -----------
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    gr_MMr: Float
        Growth rate of the MMr.
    dr_MMr: Float
        Decay rate of the MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    change_nMMr: Float
        Change in the number of MMr.

    Example:
    -----------
    >>> dMMr_dt(10, 20, 10, 5, 0.8, 0.4, np.array([
    ...    [0.7, 1, 2.5, 2.1],
    ...    [1, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0, -0.2, 1.2]]))
    436.383290554087
    """
    # Extract the necessary matrix values
    m = matrix[3, 0]
    n = matrix[3, 1]
    o = matrix[3, 2]
    p = matrix[3, 3]

    # Calculate the change in the number of MMr
    change_nMMr = (gr_MMr * nOC**m * nOB**n * nMMd**o * nMMr**p) - (dr_MMr * nMMr)
    return change_nMMr


def model_dynamics(y, t, growth_rates, decay_rates, matrix, WMMd_inhibitor = 0):
    """Function that determines the number dynamics in a population over time.

    Parameters:
    -----------
    y: List
        List with the values of nOC, nOB, nMMd and nMMr.
    t: Numpy.ndarray
        Array with all the time points.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    [nOC_change, nOB_change, nMMd_change, nMMr_change]: List
        List containing the changes in nOC, nOB, nMMd and nMMr.

    Example:
    -----------
    >>> model_dynamics([10, 20, 10, 5], 1, [0.8, 0.9, 1.3, 0.5],
    ...    [0.4, 0.3, 0.3, 0.6], np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]))
    [744654.2266544278, 1489.0458359418838, 6825.972291449797, 270.98955659630434]
    """
    nOC, nOB, nMMd, nMMr = y

    # Determine the change values
    nOC_change = dOC_dt(nOC, nOB, nMMd, nMMr, growth_rates[0], decay_rates[0],
                                                                        matrix)
    nOB_change = dOB_dt(nOC, nOB, nMMd, nMMr, growth_rates[1], decay_rates[1],
                                                                        matrix)
    nMMd_change = dMMd_dt(nOC, nOB, nMMd, nMMr, growth_rates[2], decay_rates[2],
                                                        matrix, WMMd_inhibitor)
    nMMr_change = dMMr_dt(nOC, nOB, nMMd, nMMr, growth_rates[3], decay_rates[3],
                                                                        matrix)

    # Make floats of the arrays
    nOC_change = float(nOC_change)
    nOB_change = float(nOB_change)
    nMMd_change = float(nMMd_change)
    nMMr_change = float(nMMr_change)

    return [nOC_change, nOB_change, nMMd_change, nMMr_change]


def dynamics_MMd_MMr_limits(time_IH, time_end, upper_limit_MMd, upper_limit_MMr,
            nOC, nOB, nMMd, nMMr, growth_rates, decay_rates, matrix_no_drugs,
            matrix_drugs, WMMd_inhibitor = 0):
    """Function that determines the number dynamics. It ensures that the MMr
    number and MMd number do not become too high.

    Parameters:
    -----------
    time_IH: Int
        Number of generations before the therapy start
    time_end: Int
        The last generation for which the numbers have to be calculated
    upper_limit_MMd: Int
        The maximum number of MMd, when reached the IH administration starts
    upper_limit_MMr: Int
        The maximum number of MMr, when reached the IH administration stops
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    matrix_no_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when no IHs are given.
    matrix_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when IHs are given.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_numbers: Dataframe
        The dataframe with the number per cell type over the time
    average_a_duration: Float
        The average administration duration
    average_h_duration: Float
        The average holiday duration
    """
    # Create a dataframe and lists
    df_numbers = pd.DataFrame(columns = ['Generation', 'nOC', 'nOB', 'nMMd',
                                                        'nMMr', 'total nMM'])
    duration_holiday = []
    duration_administration = []

    # Set the start values
    times_holiday = 0
    times_administration = 0
    duration = 0
    x = int(1)
    t = np.linspace(0, time_IH, time_IH)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
            'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Combine the dataframes
    df_numbers = combine_dataframes(df_numbers, df_2)

    # Calculate the number of generations the therapy is given
    time_step_t = time_end - time_IH

    # Loop over the generations
    for time_step in range(time_step_t):

        # Determine the start numbers
        nOC = df_numbers['nOC'].iloc[-1]
        nOB = df_numbers['nOB'].iloc[-1]
        nMMd = df_numbers['nMMd'].iloc[-1]
        nMMr = df_numbers['nMMr'].iloc[-1]

        # If x = 1 add IHs and if x = 0 add no IHs
        if x == 1:
            # Determine the change values
            nOC_change = dOC_dt(nOC, nOB, nMMd, nMMr, growth_rates[0],
                                                decay_rates[0], matrix_drugs)
            nOB_change = dOB_dt(nOC, nOB, nMMd, nMMr, growth_rates[1],
                                                decay_rates[1], matrix_drugs)
            nMMd_change = dMMd_dt(nOC, nOB, nMMd, nMMr, growth_rates[2],
                                decay_rates[2], matrix_drugs, WMMd_inhibitor)
            nMMr_change = dMMr_dt(nOC, nOB, nMMd, nMMr, growth_rates[3],
                                                decay_rates[3], matrix_drugs)

        if x == 0:
            # Determine the change values
            nOC_change = dOC_dt(nOC, nOB, nMMd, nMMr, growth_rates[0],
                                                decay_rates[0], matrix_no_drugs)
            nOB_change = dOB_dt(nOC, nOB, nMMd, nMMr, growth_rates[1],
                                                decay_rates[1], matrix_no_drugs)
            nMMd_change = dMMd_dt(nOC, nOB, nMMd, nMMr, growth_rates[2],
                                                decay_rates[2], matrix_no_drugs)
            nMMr_change = dMMr_dt(nOC, nOB, nMMd, nMMr, growth_rates[3],
                                                decay_rates[3], matrix_no_drugs)

        # Calculate the new cell numbers
        nOC = nOC + nOC_change
        nOB = nOB + nOB_change
        nMMd = nMMd + nMMd_change
        nMMr = nMMr + nMMr_change
        nMMt = nMMd + nMMr

        # If there are too many MMr stop drug administration
        if nMMr > upper_limit_MMr and x == int(1) and duration > 5:

            # Add the administration duration to the list withd durations
            duration_administration.append(duration)
            times_administration += 1
            duration = 0

            # Start the IH holliday
            x = int(0)

        # If there are too many MMd stop drug holiday
        if nMMd > upper_limit_MMd and x == int(0) and duration > 5:

            # Add the holiday duration to the list withd durations
            duration_holiday.append(duration)
            times_holiday += 1
            duration = 0

            # Start the IH administration
            x = int(1)

        # Add results to the dataframe
        new_row_df = pd.DataFrame([{'Generation': time_IH+time_step, 'nOC': nOC,
        'nOB': nOB, 'nMMd': nMMd, 'nMMr': nMMr, 'total nMM': nMMt}])
        df_numbers = combine_dataframes(df_numbers, new_row_df)

        # Add one to the duration
        duration += 1

    # Calculate average administration and holiday duration
    average_a_duration = sum(duration_administration) / times_administration
    average_h_duration = sum(duration_holiday) / times_holiday

    return df_numbers, average_a_duration, average_h_duration

def save_dataframe(dataframe, file_name, folder_path):
    """ Function that saves a dataframe as csv file.

    Parameters:
    -----------
    dataframe: DataFrame
        The dataframe containing the collected data.
    file_name: String
        The name of the csv file.
    folder_path: String
        Path to the folder where the data will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    dataframe.to_csv(file_path, index=False)

def save_dictionary(dictionary, file_path):
    """ Function that saves a dictionary as csv file.

    Parameters:
    -----------
    dictionary: Dictionary
        The dictionary containing the collected data.
    file_path: String
        The name of the csv file and the path where the dictionary will be saved.
    """
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Create a header
        writer.writerow(['Key', 'Value'])

        # Loop over the rows
        for key, value in dictionary.items():
            writer.writerow([str(key), str(value)])

def save_optimised_results(results, file_path):
    """ Function that saves the results of the optimised function as csv file.

    Parameters:
    -----------
    results: OptimizeResult
        The results of the scipy.optimize funtion
    file_path: String
        The name of the csv file and the path where the results will be saved.
    """
    # Extract the results
    optimised_para = results.x
    optimal_value = results.fun
    number_iterations = results.nit
    number_evaluations = results.nfev

    # Save the results in dictionary form
    results_to_saved = [ {"Optimised parameters": optimised_para.tolist(),
            "Optimal MM nr": optimal_value, 'nr iterations': number_iterations,
            'nr evaluations': number_evaluations}]

    with open(file_path, 'w', newline='') as csvfile:

        # Create header names
        header_names = ['Optimised parameters', 'Optimal MM nr', 'nr iterations',
                                                                'nr evaluations']
        writer = csv.DictWriter(csvfile, fieldnames = header_names)

        # Loop over the results
        writer.writeheader()
        for result in results_to_saved:
            writer.writerow(result)

def save_Figure(Figure, file_name, folder_path):
    """Save the Figure to a specific folder.

    Parameters:
    -----------
    Figure: Matplotlib Figure
        Figure object that needs to be saved.
    file_name: String
        The name for the plot.
    folder_path: String:
        Path to the folder where the data will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    Figure.savefig(os.path.join(folder_path, file_name))

def combine_dataframes(df_1, df_2):
    """ Function that combines two datafranes in on dataframe

    Parameters:
    -----------
    df_1: DataFrame
        The first dataframe containing the collected data.
    df_2: DataFrame
        The second dataframe containing the collected data.

    Returns:
    --------
    combined_df: DataFrame
        Dataframe that is a combination of the two dataframes
    """
    # Check if the dataframes are empty
    if df_1.empty or df_2.empty:
        # return the dataframe that is not empty
        combined_df = df_1 if not df_1.empty else df_2

    else:
        # delete the NA columns
        df_1 = df_1.dropna(axis=1, how='all')
        df_2 = df_2.dropna(axis=1, how='all')

        # Combine the dataframes
        combined_df = pd.concat([df_1, df_2], ignore_index=True)

    return(combined_df)

def switch_dataframe(time_IH, n_switches, t_steps_drug, t_steps_no_drug, nOC,
            nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates,
            decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time for a given time of drug holiday and administration periods.

    Parameters:
    -----------
    time_IH: Int
        The time point at witch the drugs are administered
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: Int
        The number of generations drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = time_IH
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0],
                            'nOB': y[:, 1], 'nMMd': y[:, 2], 'nMMr': y[:, 3],
                            'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of switches
    for i in range(n_switches):

        # If x = 0 make sure the MMd is inhibited
        if x == 0:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH

            t = np.linspace(time, time + t_steps_drug, t_steps_drug)
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(1)
            time += t_steps_drug

        # If x = 1 make sure the MMd is not inhibited
        else:
            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , t_steps_no_drug)
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

def switch_dataframe_GF_W_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
                            t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                            growth_rates_IH, decay_rates, decay_rates_IH,
                            matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a MMd GF IH is administered, then a WMMd IH and then there
    is a IH holiday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 30
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0],
                                'nOB': y[:, 1], 'nMMd': y[:, 2], 'nMMr': y[:, 3],
                                'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # MMd GF IH
        if x == 0:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH

            t = np.linspace(time, time + t_steps_GF_IH, int(t_steps_GF_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(1)
            time += t_steps_GF_IH

        # WMMd IH
        if x == 1:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_WMMd_IH, int(t_steps_WMMd_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(2)
            time += t_steps_WMMd_IH

        # No IH
        if x == 2:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch


def switch_dataframe_W_comb_h(n_rounds, t_steps_WMMd_IH, t_steps_comb,
            t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a WMMd IH is administered, then a IH combination, then a MMd
    GF IH and then a IH holiday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_comb: Int
        The number of generations WMMd IH and MMd GF IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.
    WMMd_inhibitor_comb: Float
        The effect of a drug on the MMd fitness when also a MMd GF IH is given.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 30
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # WMMd IH
        if x == 0:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_WMMd_IH, int(t_steps_WMMd_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(1)
            time += t_steps_WMMd_IH

        # IH combination
        if x == 1:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH_comb

            t = np.linspace(time, time + t_steps_comb, int(t_steps_comb))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix,
                                                            WMMd_inhibitor_comb)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(2)
            time += t_steps_comb

        # No IH
        if x == 2:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

def switch_dataframe_GF_comb_h(n_rounds, t_steps_GF_IH, t_steps_comb,
            t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH,
            matrix_GF_IH_comb, WMMd_inhibitor_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a MMd GF IH is administered, the a IH combination, then a
    MMd GF IH and then a IH holiday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_comb: Int
        The number of generations WMMd IH and MMd GF IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.
    WMMd_inhibitor_comb: Float
        The effect of a drug on the MMd fitness when also a MMd GF IH is given.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 30
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': \
      y[:, 1], 'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # MMd GF IH
        if x == 0:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH

            t = np.linspace(time, time + t_steps_GF_IH, int(t_steps_GF_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(1)
            time += t_steps_GF_IH

        # IH combination
        if x == 1:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH_comb

            t = np.linspace(time, time + t_steps_comb, int(t_steps_comb))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix,
                                                         WMMd_inhibitor_comb)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(2)
            time += t_steps_comb

        # No IH
        if x == 2:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch


def switch_dataframe_GF_h_W_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
                    t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                    growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a MMd GF IH is administered, then a IH holiday, then a WMMd
    IH and then a IH holiday again.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 30
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # MMd GF IH
        if x == 0:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH

            t = np.linspace(time, time + t_steps_GF_IH, int(t_steps_GF_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(1)
            time += t_steps_GF_IH

        # No IH
        if x == 1:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(2)
            time += t_steps_no_drug

        # WMMd IH
        if x == 2:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_WMMd_IH, int(t_steps_WMMd_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(3)
            time += t_steps_WMMd_IH

        # No IH
        if x == 3:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

def switch_dataframe_W_h_GF_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
                    t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                    growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a WMMd IH is administered, then a IH holiday, then a MMd GF
    IH and then a IH holiday again.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 30
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # WMMd IH
        if x == 0:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_WMMd_IH, int(t_steps_WMMd_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(1)
            time += t_steps_WMMd_IH

        # No IH
        if x == 1:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(2)
            time += t_steps_no_drug

        # MMd GF IH
        if x == 2:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH

            t = np.linspace(time, time + t_steps_GF_IH, int(t_steps_GF_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(3)
            time += t_steps_GF_IH

        # No IH
        if x == 3:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch


def switch_dataframe_W_GF_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
                    t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                    growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time. First a WMMd IH is administered, then a MMd GF IH and then there is a
    IH holiday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 30
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # WMMd IH
        if x == 0:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_WMMd_IH, int(t_steps_WMMd_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(1)
            time += t_steps_WMMd_IH

        # MMd GF IH
        if x == 1:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH

            t = np.linspace(time, time + t_steps_GF_IH, int(t_steps_GF_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(2)
            time += t_steps_GF_IH

        # No IH
        if x == 2:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch


def switch_dataframe_W_comb_GF_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
            t_steps_comb, t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a WMMd IH is administered, then a IH combination, then a MMd
    GF IH and then a IH holiday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_comb: Int
        The number of generations WMMd IH and MMd GF IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.
    WMMd_inhibitor_comb: Float
        The effect of a drug on the MMd fitness when also a MMd GF IH is given.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 30
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': \
       y[:, 1], 'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # WMMd IH
        if x == 0:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_WMMd_IH, int(t_steps_WMMd_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(1)
            time += t_steps_WMMd_IH

        # IH combination
        if x == 1:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH_comb

            t = np.linspace(time, time + t_steps_comb, int(t_steps_comb))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix,
                                                            WMMd_inhibitor_comb)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(2)
            time += t_steps_comb

        # MMd GF IH
        if x == 2:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH

            t = np.linspace(time, time + t_steps_GF_IH, int(t_steps_GF_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(3)
            time += t_steps_GF_IH

        # No IH
        if x == 3:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

def switch_dataframe_GF_comb_W_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
            t_steps_comb, t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a MMd GF IH is administered, the a IH combination, then a
    MMd GF IH and then a IH holiday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_comb: Int
        The number of generations WMMd IH and MMd GF IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.
    WMMd_inhibitor_comb: Float
        The effect of a drug on the MMd fitness when also a MMd GF IH is given.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 30
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # MMd GF IH
        if x == 0:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH

            t = np.linspace(time, time + t_steps_GF_IH, int(t_steps_GF_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(1)
            time += t_steps_GF_IH

        # IH combination
        if x == 1:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH_comb

            t = np.linspace(time, time + t_steps_comb, int(t_steps_comb))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix,
                                                         WMMd_inhibitor_comb)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(2)
            time += t_steps_comb

        # WMMd IH
        if x == 2:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_WMMd_IH, int(t_steps_WMMd_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(3)
            time += t_steps_WMMd_IH

        # No IH
        if x == 3:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch


def switch_dataframe_GF_WandGF_W_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
            t_steps_comb, t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time. First a MMd GF IH is administered, then the WMMd IH and MMd GF IH, then
    a MMd GF IH and then there is a drug holliday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_comb: Int
        The number of generations WMMd IH and GF MMd IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 30
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # MMd GF IH
        if x == 0:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH

            t = np.linspace(time, time + t_steps_GF_IH, int(t_steps_GF_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 1
            time += t_steps_GF_IH

        # WMMd IH and MMd GF IH
        if x == 1:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH_comb

            t = np.linspace(time, time + t_steps_comb, int(t_steps_comb))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 2
            time += t_steps_comb

        # WMMd IH
        if x == 2:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_WMMd_IH, int(t_steps_WMMd_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 3
            time += t_steps_WMMd_IH

        # No drug
        if x == 3:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 0
            time += t_steps_no_drug

    return df_total_switch



def switch_dataframe_W_WandGF_GF_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
            t_steps_comb, t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time. First a WMMd IH is administered, then the WMMd IH and MMd GF IH, then a
    MMd GF IH and then there is a drug holliday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_comb: Int
        The number of generations WMMd IH and GF MMd IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 30
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # WMMd IH
        if x == 0:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_WMMd_IH, int(t_steps_WMMd_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 1
            time += t_steps_WMMd_IH

        # WMMd IH and MMd GF IH
        if x == 1:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH_comb

            t = np.linspace(time, time + t_steps_comb, int(t_steps_comb))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 2
            time += t_steps_comb

        # MMd GF IH
        if x == 2:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH

            t = np.linspace(time, time + t_steps_GF_IH, int(t_steps_GF_IH))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates_IH, decay_rates_IH, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 3
            time += t_steps_GF_IH

        # No drug
        if x == 3:

            # Determine the start numbers
            nOC = df_total_switch['nOC'].iloc[-1]
            nOB = df_total_switch['nOB'].iloc[-1]
            nMMd = df_total_switch['nMMd'].iloc[-1]
            nMMr = df_total_switch['nMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , int(t_steps_no_drug))
            y0 = [nOC, nOB, nMMd, nMMr]
            parameters = (growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 0
            time += t_steps_no_drug

    return df_total_switch


def minimal_tumour_nr_t_3_situations(t_steps_IH_strength, function_order, nOC,
                nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates,
                decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time for a given MMd GF IH administration, WMMd IH administration and
    holiday duration and IH strength.

    Parameters:
    -----------
    t_steps_IH_strength: List
        List with the number of generations the MMD GF IH, the WMMd IH and no
        drugs are administared and the MMD GF IH and WMMd IH strength.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    average_MM_number: float
        The average total MM number in the last period.
    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_WMMd_IH, t_steps_no_drug, = t_steps_IH_strength
    n_rounds = 50

    # Determine the round duration and the matrix value
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_WMMd_IH

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
     t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
     decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine the average MM number in the last period with and without drugs
    last_MM_numbers = df['total nMM'].tail(int(time_round))
    average_MM_number = last_MM_numbers.sum() / (int(time_round))

    return float(average_MM_number)


def minimal_tumour_nr_t_3_situations_IH(t_steps_IH_strength, function_order,
                weight_MMr, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
                decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time for a given MMd GF IH administration, WMMd IH administration and holiday
    duration and IH strength.

    Parameters:
    -----------
    t_steps_IH_strength: List
        List with the number of generations the MMD GF IH, the WMMd IH and no
        drugs are administared and the MMD GF IH and WMMd IH strength.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.

    Returns:
    --------
    average_MM_number: float
        The average (weighted) MM number in the last period.
    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_WMMd_IH, t_steps_no_drug, GF_IH,\
                                            WMMd_inhibitor = t_steps_IH_strength
    n_rounds = 50

    # Determine the round duration and the matrix value
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_WMMd_IH
    matrix_GF_IH[2, 0] = 0.6 - GF_IH

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
     t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
     decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine if the normal or weighted MMM number should be calculated
    if weight_MMr == False:

        # Determine the average MM number in the last period
        last_MM_numbers = df['total nMM'].tail(int(time_round))
        average_MM_number = last_MM_numbers.sum() / (int(time_round))

    else:
        # Determine the weighted MM number in the last period
        last_MMd_numbers = df['nMMd'].tail(int(time_round))
        last_MMr_numbers = df['nMMr'].tail(int(time_round)) * weight_MMr
        average_MM_number = (last_MMd_numbers.sum() + last_MMr_numbers.sum())/ \
                                                                (int(time_round))

    return float(average_MM_number)

def minimal_tumour_nr_t_3_sit_GF_IH(t_steps_IH_strength, function_order,
    weight_MMr, nOC,nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates,
    decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, matrix_GF_IH_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time for a given MMd GF IH administration, WMMd IH administration, IH
    combination administration and holiday duration and Ih strength.

    Parameters:
    -----------
    t_steps_IH_strength: List
        List with the number of generations the MMD GF IH, the WMMd IH and no
        drugs are administared and the MMD GF IH and WMMd IH strength.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    weight_MMr: Int
          The weight of the MMr relative to that of the MMd.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.

    Returns:
    --------
    average_MM_number: float
        The (weighted) total MM number in the last period.
    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_comb, t_steps_no_drug, GF_IH, GF_IH_comb, \
                                        WMMd_inhibitor_comb = t_steps_IH_strength
    n_rounds = 50

    # Determine the round duration and the matrix values
    matrix_GF_IH[2, 0] = 0.6 - GF_IH
    matrix_GF_IH_comb[2, 0] = 0.6 - GF_IH_comb
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_comb

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_comb,
        t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
        decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH,
        matrix_GF_IH_comb, WMMd_inhibitor_comb)

    # Determine if the normal or weighted MMM number should be calculated
    if weight_MMr == False:

        # Determine the average MM number in the last period
        last_MM_numbers = df['total nMM'].tail(round(time_round))
        average_MM_number = last_MM_numbers.sum() / (round(time_round))

    else:
        # Determine the weighted MM number in the last period
        last_MMd_numbers = df['nMMd'].tail(round(time_round))
        last_MMr_numbers = df['nMMr'].tail(round(time_round)) * weight_MMr
        average_MM_number = (last_MMd_numbers.sum() + last_MMr_numbers.sum())/ \
                                                            (round(time_round))

    return float(average_MM_number)


def minimal_tumour_nr_t_3_sit_W_IH(t_steps_IH_strength, function_order,
                weight_MMr, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
                decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time for a given MMd GF IH administration, WMMd IH administration, IH
    combination administration and holiday duration and Ih strength.

    Parameters:
    -----------
    t_steps_IH_strength: List
        List with the number of generations the MMD GF IH, the WMMd IH and no
        drugs are administared and the MMD GF IH and WMMd IH strength.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    weight_MMr: Int
          The weight of the MMr relative to that of the MMd.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.

    Returns:
    --------
    average_MM_number: float
        The (weighted) total MM number in the last period.
    """
    # Unpack the values that should be optimised
    t_steps_WMMd_IH, t_steps_comb, t_steps_no_drug, GF_IH_comb, \
                    WMMd_inhibitor,  WMMd_inhibitor_comb = t_steps_IH_strength
    n_rounds = 50

    # Determine the round duration and the matrix values
    matrix_GF_IH_comb[2, 0] = 0.6 - GF_IH_comb
    time_round =  t_steps_no_drug + t_steps_WMMd_IH + t_steps_comb

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_WMMd_IH, t_steps_comb,
        t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
        decay_rates, decay_rates_IH, matrix_no_GF_IH,
        matrix_GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb)

    # Determine if the normal or weighted MMM number should be calculated
    if weight_MMr == False:

        # Determine the average MM number in the last period with and without drugs
        last_MM_numbers = df['total nMM'].tail(round(time_round))
        average_MM_number = last_MM_numbers.sum() / (round(time_round))

    else:
        # Determine the weighted MM number in the last period
        last_MMd_numbers = df['nMMd'].tail(round(time_round))
        last_MMr_numbers = df['nMMr'].tail(round(time_round)) * weight_MMr
        average_MM_number = (last_MMd_numbers.sum() + last_MMr_numbers.sum())/ \
                                                                (round(time_round))

    return float(average_MM_number)

def minimal_tumour_nr_t_3_4_situations_IH(t_steps_IH_strength, function_order,
            weight_MMr, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time for a given MMd GF IH administration, WMMd IH administration and
    holiday duration and IH strength.

    Parameters:
    -----------
    t_steps_IH_strength: List
        List with the number of generations the MMD GF IH, the WMMd IH and no
        drugs are administared and the MMD GF IH and WMMd IH strength.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.

    Returns:
    --------
    average_MM_number: float
        The (weighted) total MM number in the last period.
    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_WMMd_IH, t_steps_no_drug, GF_IH,\
                                            WMMd_inhibitor = t_steps_IH_strength
    n_rounds = 50

    # Determine the round duration and the matrix value
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_WMMd_IH + \
                                                                t_steps_no_drug
    matrix_GF_IH[2, 0] = 0.6 - GF_IH

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
      t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
      decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine if the normal or weighted MMM number should be calculated
    if weight_MMr == False:

        # Determine the average MM number in the last period
        last_MM_numbers = df['total nMM'].tail(int(time_round))
        average_MM_number = last_MM_numbers.sum() / (int(time_round))

    else:
        # Determine the weighted MM number in the last period
        last_MMd_numbers = df['nMMd'].tail(int(time_round))
        last_MMr_numbers = df['nMMr'].tail(int(time_round)) * weight_MMr
        average_MM_number = (last_MMd_numbers.sum() + last_MMr_numbers.sum())/ \
                                                                (int(time_round))

    return float(average_MM_number)

def minimal_tumour_nr_t_4_situations(t_steps, function_order, nOC, nOB, nMMd,
                        nMMr, growth_rates, growth_rates_IH, decay_rates,
                        decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH,
                        matrix_GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time for a given MMd GF IH administration, WMMd IH administration, IH
    combination administration and holiday duration and IH strength.

    Parameters:
    -----------
    t_steps: List
        List with the number of generations the MMD GF IH, the WMMd IH, the IH
        combination and no drugs are administared.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.
    WMMd_inhibitor_comb: Float
        The effect of a drug on the MMd fitness when also a MMd GF IH is given.

    Returns:
    --------
    average_MM_number: float
        The average total MM number in the last period.
    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_WMMd_IH, t_steps_comb, t_steps_no_drug = t_steps
    n_rounds = 50

    # Determine the round duration
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_WMMd_IH + t_steps_comb

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH, t_steps_comb,
        t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
        decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH,
        matrix_GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb)

    # Determine the average MM number in the last period with and without drugs
    last_MM_numbers = df['total nMM'].tail(int(time_round))
    average_MM_number = last_MM_numbers.sum() / (int(time_round))

    return float(average_MM_number)

def minimal_tumour_nr_t_4_sit_equal(t_steps_IH_strength, function_order, nOC, nOB,
        nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
        matrix_no_GF_IH, matrix_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time for a given MMd GF IH administration, WMMd IH administration, WMMd IH +
    MMd GF IH combination administration and holiday duration.

    Parameters:
    -----------
    t_steps_IH_strength: List
        List with the number of generations the MMD GF IH, the WMMd IH and no
        drugs are administared and the MMD GF IH and WMMd IH strength.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    average_MM_number: float
        The average total MM number in the last period.

    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_WMMd_IH, t_steps_comb, t_steps_no_drug = \
                                                            t_steps_IH_strength

    # Determine the round duration and the matrix values
    n_rounds = 50
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_WMMd_IH + t_steps_comb

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH, t_steps_comb,
        t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
        decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH,
        matrix_GF_IH_comb, WMMd_inhibitor)

    # Determine the average MM number in the last period with and without drugs
    last_MM_numbers = df['total nMM'].tail(int(time_round))
    average_MM_number = last_MM_numbers.sum() / (int(time_round))

    return float(average_MM_number)

def minimal_tumour_nr_t_4_sit_equal_IH(t_steps_IH_strength, function_order,
    weight_MMr, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates,
    decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, matrix_GF_IH_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time for a given MMd GF IH administration, WMMd IH administration, WMMd IH +
    MMd GF IH combination administration and holiday duration.

    Parameters:
    -----------
    t_steps_IH_strength: List
        List with the number of generations the MMD GF IH, the WMMd IH and no
        drugs are administared and the MMD GF IH and WMMd IH strength.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    weight_MMr: Int
          The weight of the MMr relative to that of the MMd.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.

    Returns:
    --------
    average_MM_number: float
        The average total MM number in the last period.

    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_WMMd_IH, t_steps_comb, t_steps_no_drug, GF_IH, \
                                            WMMd_inhibitor = t_steps_IH_strength

    # Determine the round duration and the matrix values
    matrix_GF_IH[2, 0] = 0.6 - GF_IH
    matrix_GF_IH_comb[2, 0] = 0.6 - GF_IH
    n_rounds = 50
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_WMMd_IH + t_steps_comb

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH, t_steps_comb,
        t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
        decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH,
        matrix_GF_IH_comb, WMMd_inhibitor)

    # Determine if the normal or weighted MMM number should be calculated
    if weight_MMr == False:

        # Determine the average MM number in the last period
        last_MM_numbers = df['total nMM'].tail(int(time_round))
        average_MM_number = last_MM_numbers.sum() / (int(time_round))

    else:
        # Determine the weighted MM number in the last period
        last_MMd_numbers = df['nMMd'].tail(int(time_round))
        last_MMr_numbers = df['nMMr'].tail(int(time_round)) * weight_MMr
        average_MM_number = (last_MMd_numbers.sum() + last_MMr_numbers.sum())/ \
                                                                (int(time_round))

    return float(average_MM_number)


def minimal_tumour_nr_t_4_situations_IH(t_steps_IH_strength, function_order,
    weight_MMr, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates,
    decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, matrix_GF_IH_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time for a given MMd GF IH administration, WMMd IH administration, IH
    combination administration and holiday duration and Ih strength.

    Parameters:
    -----------
    t_steps_IH_strength: List
        List with the number of generations the MMD GF IH, the WMMd IH and no drugs
        are administared and the MMD GF IH and WMMd IH strength.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_GF_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd

    Returns:
    --------
    average_MM_number: float
        The average (weighted) MM number in the last period.
    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_WMMd_IH, t_steps_comb, t_steps_no_drug, GF_IH, \
         GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb = t_steps_IH_strength
    n_rounds = 50

    # Determine the round duration and the matrix values
    matrix_GF_IH[2, 0] = 0.6 - GF_IH
    matrix_GF_IH_comb[2, 0] = 0.6 - GF_IH_comb
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_WMMd_IH + t_steps_comb

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH, t_steps_comb,
        t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
        decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH,
        matrix_GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb)

    # Determine if the normal or weighted MMM number should be calculated
    if weight_MMr == False:

        # Determine the average MM number in the last period
        last_MM_numbers = df['total nMM'].tail(int(time_round))
        average_MM_number = last_MM_numbers.sum() / (int(time_round))

    else:
        # Determine the weighted MM number in the last period
        last_MMd_numbers = df['nMMd'].tail(int(time_round))
        last_MMr_numbers = df['nMMr'].tail(int(time_round)) * weight_MMr
        average_MM_number = (last_MMd_numbers.sum() + last_MMr_numbers.sum())/ \
                                                                (int(time_round))

    return float(average_MM_number)

def continuous_add_IH_df(time_IH, end_generation, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the cell type numbers when the IHs
    are administered continuously.

    Parameters:
    -----------
    time_IH: Int
        The time point at which the IHs get administered
    end_generation: Int
        The last generation for which the numbers have to be calculated
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
                                                                administrated.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total: DataFrame
        The dataframe with the cell numbers when IHs are continiously administered.
    """
    # Set the start values
    t = np.linspace(0, time_IH, time_IH)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Determine the current numbers
    nOC = df_1['nOC'].iloc[-1]
    nOB = df_1['nOB'].iloc[-1]
    nMMd = df_1['nMMd'].iloc[-1]
    nMMr = df_1['nMMr'].iloc[-1]

    # Set the currect values
    t = np.linspace(time_IH, end_generation, 400)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates_IH, decay_rates_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Combine the dataframes
    df_total = combine_dataframes(df_1, df_2)

    return df_total

def x_y_z_axis_values_3d_plot(dataframe, name):
    """ Function that determines the x, y and z axis values from the given
    dataframe. It also prints the administration and holiday duration leading
    to the lowest total MM number in the equilibrium

    Parameters:
    -----------
    Dataframe: dataFrame
        The dataframe with the generated data
    name: String
        The name of the administered IH(s)

    Returns:
    --------
    X_values: Numpy.ndarray
        Array with the values for the x-axis
    Y_values: Numpy.ndarray
        Array with the values for the y-axis
    Z_values: Numpy.ndarray
        Array with the values for the z-axis
    """

    # Find the drug administration and holiday period causing the lowest MM
    # number
    min_index =  dataframe['MM number'].idxmin()
    g_no_drug_min = dataframe.loc[min_index, 'Generations no drug']
    g_drug_min = dataframe.loc[min_index, 'Generations drug']
    frac_min = dataframe.loc[min_index, 'MM number']

    print(f"""Lowest MM number: {frac_min}-> MMd {name} holidays are
            {g_no_drug_min} generations and MMd {name} administrations
            are {g_drug_min} generations""")

    # Avoid errors because of the wrong datatype
    dataframe['Generations no drug'] = pd.to_numeric(dataframe[\
                                        'Generations no drug'], errors='coerce')
    dataframe['Generations drug'] = pd.to_numeric(dataframe[\
                                        'Generations drug'],errors='coerce')
    dataframe['MM number'] = pd.to_numeric(dataframe['MM number'],
                                                            errors='coerce')

    # Make a meshgrid for the plot
    X_values = dataframe['Generations no drug'].unique()
    Y_values = dataframe['Generations drug'].unique()
    X_values, Y_values = np.meshgrid(X_values, Y_values)
    Z_values = np.zeros((20, 20))

    # Fill the 2D array with the MM number values by looping over each row
    for index, row in dataframe.iterrows():
        i = int(row.iloc[0]) - 2
        j = int(row.iloc[1]) - 2
        Z_values[j, i] = row.iloc[2]

    return (X_values, Y_values, Z_values)

def minimal_tumour_numb_t_steps(t_steps_drug, t_steps_no_drug, nOC, nOB, nMMd,
                nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time for a given time of a drug holiday.

    Parameters:
    -----------
    t_steps_drug: Int
        The number of generations drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    average_MM_number: float
        The average total MM number in the last period.

    """
    # Deteremine the number of switches
    time_step = (t_steps_drug + t_steps_no_drug) / 2
    n_switches = int((400 // time_step) -1)

    # Create a dataframe of the numbers
    df = switch_dataframe(30, n_switches, t_steps_drug, t_steps_no_drug, nOC,
                  nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates,
                  decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine the average MM number in the last period with and without drugs
    last_MM_numbers = df['total nMM'].tail(int(time_step *2))
    average_MM_number = last_MM_numbers.sum() / (int(time_step*2))

    return float(average_MM_number)

def minimal_tumour_numb_b_OC_MMd(b_OC_MMd, nOC, nOB, nMMd, nMMr, growth_rates,
        growth_rates_IH, decay_rates, decay_rates_IH, matrix, b_OC_MMd_array):
    """Function that determines the number of the population being MM for a
    specific b_OC_MMd value.

    Parameters:
    -----------
    b_OC_MMd: Float
        Interaction value that gives the effect of the GFs of OC on MMd.
    nOC: Float
        number of OC.
    nOB: Float
        number of OB.
    nMMd: Float
        number of the MMd.
    nMMr: Float
        number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    b_OC_MMd_array: Float
        If True b_OC_MMd is an array and if False b_OC_MMd is a float.

    Returns:
    --------
    last_MM_number: Float
        The total MM number.

    Example:
    --------
    >>> matrix = np.array([
    ...    [0.0, 0.4, 0.6, 0.5],
    ...    [0.3, 0.0, -0.3, -0.3],
    ...    [0.6, 0.0, 0.2, 0.0],
    ...    [0.55, 0.0, -0.6, 0.4]])
    >>> minimal_tumour_numb_b_OC_MMd(0.4, 20, 30, 20, 5, [0.8, 1.2, 0.3, 0.3],
    ...                            [0.7, 1.3, 0.3, 0.3], [0.9, 0.08, 0.2, 0.1],
    ...                            [1.0, 0.08, 0.2, 0.1], matrix, False)
    21.716742987793786
    """
    # Set the initial conditions
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix)
    t = np.linspace(0, 70, 70)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Determine the current numbers
    nOC = df_1['nOC'].iloc[-1]
    nOB = df_1['nOB'].iloc[-1]
    nMMd = df_1['nMMd'].iloc[-1]
    nMMr = df_1['nMMr'].iloc[-1]

    # Change b_OC_MMd to a float if it is an array
    if b_OC_MMd_array == True:
        b_OC_MMd = b_OC_MMd[0]

    # Change the b_OC_MMd value to the specified value
    matrix[2, 0]= b_OC_MMd

    # Set initial conditions
    t_over = np.linspace(70, 200, 140)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates_IH, decay_rates_IH, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t_over, args=parameters)
    df_2 = pd.DataFrame({'Generation': t_over, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Determine the total MM number
    last_MM_number = df_2['total nMM'].iloc[-1]

    return float(last_MM_number)

def minimal_tumour_numb_WMMd_IH(WMMd_inhibitor, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix, WMMd_inhibitor_array):
    """Function that determines the number of the population being MM for a
    specific WMMd drug inhibitor value.

    Parameters:
    -----------
    WMMd_inhibitor: Float
        Streght of the drugs that inhibits the MMd fitness.
    nOC: Float
        number of OC.
    nOB: Float
        number of OB.
    nMMd: Float
        number of the MMd.
    nMMr: Float
        number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor_array: Float
        If True WMMd_inhibitor is an array and if False WMMd_inhibitor is a float.

    Returns:
    --------
    last_MM_number: Float
        The total MM number.

    Example:
    -----------
    average_MM_numbers: float
        The total MM number.

    >>> matrix = np.array([
    ...    [0.0, 0.4, 0.6, 0.5],
    ...    [0.3, 0.0, -0.3, -0.3],
    ...    [0.6, 0.0, 0.2, 0.0],
    ...    [0.55, 0.0, -0.6, 0.4]])
    >>> minimal_tumour_numb_WMMd_IH(0.3, 20, 30, 20, 5, [0.8, 1.2, 0.3, 0.3],
    ...                            [0.7, 1.3, 0.3, 0.3], [0.9, 0.08, 0.2, 0.1],
    ...                            [1.0, 0.08, 0.2, 0.1], matrix, False)
    24.805076184735565
    """
    # Determine if WMMd_inhibitor is an array
    if WMMd_inhibitor_array == True:
        WMMd_inhibitor = WMMd_inhibitor[0]

    # Set initial conditions
    t = np.linspace(0, 60, 60)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Determine the current numbers
    nOC = df_1['nOC'].iloc[-1]
    nOB = df_1['nOB'].iloc[-1]
    nMMd = df_1['nMMd'].iloc[-1]
    nMMr = df_1['nMMr'].iloc[-1]

    # Set initial conditions
    t_over = np.linspace(60, 200, 140)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates_IH, decay_rates_IH, matrix, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t_over, args=parameters)
    df_2 = pd.DataFrame({'Generation': t_over, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Determine the total MM number
    last_MM_number = df_2['total nMM'].iloc[-1]

    return float(last_MM_number)


""" Figure that shows the MM number after optimisation for different MMd GF IH
and WMMd IH strengths"""
def Figure_optimisation():
    """ Function that makes a figure of the the MM number after optimisation for
    different MMd GF IH and WMMd IH strengths. It shows the MM number for the
    repeating stitautions WMMd IH -> MMd GF IH -> holiday and MMd GF IH -> WMMd
    IH -> holiday.
    """
    # Collect needed data -> normal growth and decay rate
    df_GF_W_h_changing_GF = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_GF_W_h_changing_GF_IH.csv')
    df_W_GF_h_changing_GF = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_W_GF_h_changing_GF_IH.csv')
    df_GF_W_h_changing_W = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_GF_W_h_changing_W_IH.csv')
    df_W_GF_h_changing_W = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_W_GF_h_changing_W_IH.csv')

    # Collect needed data -> increased growth and decay rate
    df_GF_W_h_changing_GF_h = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_GF_W_h_changing_GF_IH_h_gr_dr.csv')
    df_W_GF_h_changing_GF_h = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_W_GF_h_changing_GF_IH_h_gr_dr.csv')
    df_GF_W_h_changing_W_h = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_GF_W_h_changing_W_IH_h_gr_dr.csv')
    df_W_GF_h_changing_W_h = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_W_GF_h_changing_W_IH_h_gr_dr.csv')

    # Collect needed data -> decreased growth and decay rate
    df_GF_W_h_changing_GF_l = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_GF_W_h_changing_GF_IH_l_gr_dr.csv')
    df_W_GF_h_changing_GF_l = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_W_GF_h_changing_GF_IH_l_gr_dr.csv')
    df_GF_W_h_changing_W_l = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_GF_W_h_changing_W_IH_l_gr_dr.csv')
    df_W_GF_h_changing_W_l = pd.read_csv(\
    r'..\data\data_model_nr_IH_inf\df_MM_W_GF_h_changing_W_IH_l_gr_dr.csv')

    # Create a plot with two sublot next to eachother
    fig, axs = plt.subplots(2, 3, figsize=(22, 12))

    # First subplot, here MMd GF IH is changing
    axs[0, 0].plot(df_W_GF_h_changing_GF['GF IH strength'],
        df_GF_W_h_changing_GF['MM number'], color = 'mediumpurple')
    axs[0, 0].plot(df_W_GF_h_changing_GF['GF IH strength'],
        df_W_GF_h_changing_GF['MM number'],color = 'teal')
    axs[0, 0].set_title(r"""Changing MMd GF IH strengths (normal)""",
                                                                fontsize = 13)
    axs[0, 0].set_xlabel('Strength MMd GF IH', fontsize = 12)
    axs[0, 0].set_ylabel('MM number', fontsize = 12)

    # Second subplot, here WMMd IH is changing
    axs[1, 0].plot(df_W_GF_h_changing_W['W IH strength'],
     df_GF_W_h_changing_W['MM number'], color = 'mediumpurple')
    axs[1, 0].plot(df_W_GF_h_changing_W['W IH strength'],
      df_W_GF_h_changing_W['MM number'], color = 'teal')
    axs[1, 0].set_title(r"""Changing $W_{MMd}$ IH strength (normal)""",
                                                                fontsize = 13)
    axs[1, 0].set_xlabel('Strength $W_{MMd}$ IH', fontsize = 12)
    axs[1, 0].set_ylabel('MM number', fontsize = 12)
    axs[1, 0].set_xticks([0.20, 0.23, 0.26, 0.29, 0.32, 0.35, 0.38])

    # Third subplot, here MMd GF IH is changing
    axs[0, 1].plot(df_W_GF_h_changing_GF_h['GF IH strength'],
        df_GF_W_h_changing_GF_h['MM number'], color = 'mediumpurple')
    axs[0, 1].plot(df_W_GF_h_changing_GF_h['GF IH strength'],
        df_W_GF_h_changing_GF_h['MM number'],color = 'teal')
    axs[0, 1].set_title(r"""Changing MMd GF IH strength (increased)""",
                                                                fontsize = 13)
    axs[0, 1].set_xlabel('Strength MMd GF IH', fontsize = 12)
    axs[0, 1].set_ylabel('MM number', fontsize = 12)

    # Fourth subplot, here WMMd IH is changing
    axs[1, 1].plot(df_W_GF_h_changing_W_h['W IH strength'],
     df_GF_W_h_changing_W_h['MM number'], color = 'mediumpurple')
    axs[1, 1].plot(df_W_GF_h_changing_W_h['W IH strength'],
      df_W_GF_h_changing_W_h['MM number'], color = 'teal')
    axs[1, 1].set_title(r"""Changing $W_{MMd}$ IH strength (increased)""",
                                                                fontsize = 13)
    axs[1, 1].set_xlabel('Strength $W_{MMd}$ IH', fontsize = 12)
    axs[1, 1].set_ylabel('MM number', fontsize = 12)
    axs[1, 1].set_xticks([0.20, 0.23, 0.26, 0.29, 0.32, 0.35, 0.38])

    # Fifth subplot, here MMd GF IH is changing
    axs[0, 2].plot(df_W_GF_h_changing_GF_l['GF IH strength'],
        df_GF_W_h_changing_GF_l['MM number'], color = 'mediumpurple')
    axs[0, 2].plot(df_W_GF_h_changing_GF_l['GF IH strength'],
        df_W_GF_h_changing_GF_l['MM number'],color = 'teal')
    axs[0, 2].set_xlabel('Strength MMd GF IH', fontsize = 12)
    axs[0, 2].set_title(r"""Changing MMd GF IH strength (decreased)""",
                                                                fontsize = 13)
    axs[0, 2].set_ylabel('MM number', fontsize = 12)

    # Sixth subplot, here WMMd IH is changing
    axs[1, 2].plot(df_W_GF_h_changing_W_l['W IH strength'],
     df_GF_W_h_changing_W_l['MM number'], color = 'mediumpurple')
    axs[1, 2].plot(df_W_GF_h_changing_W_l['W IH strength'],
      df_W_GF_h_changing_W_l['MM number'], color = 'teal')
    axs[1, 2].set_title(r"""Changing $W_{MMd}$ IH strength (decreased)""",
                                                                fontsize = 13)
    axs[1, 2].set_xlabel('Strength $W_{MMd}$ IH', fontsize = 12)
    axs[1, 2].set_ylabel('MM number', fontsize = 12)
    axs[1, 2].set_xticks([0.20, 0.23, 0.26, 0.29, 0.32, 0.35, 0.38])

    # Create a single legend outside of all plots
    legend_labels = [r'MMd GF IH → $W_{MMd}$ IH → holiday',
                                     r'$W_{MMd}$ IH → MMd GF IH → holiday']

    fig.legend(labels = legend_labels, loc='upper center', ncol=4,
                                                            fontsize='large')

    # Save and show the plot
    save_Figure(plt, 'Figure_optimisation_comb_n_h_l',
                                    r'..\visualisation\results_model_nr_IH_inf')
    plt.show()

""" Figure to determine the difference between traditional and adaptive therapy
(original situation)"""
def Figure_continuous_MTD_vs_AT(n_switches, t_steps_drug):
    """ Function that makes a figure with 6 subplots showing the cell number
    dynamics by traditional therapy (continuous MTD) and adaptive therapy.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the number of time steps drugs are administared and the breaks
        are for the different Figures.
    """
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.13, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.6, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.24, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.18

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.74

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_GF = switch_dataframe(60, n_switches, t_steps_drug[0],
            t_steps_drug[0], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH)
    df_total_switch_WMMd = switch_dataframe(60, n_switches, t_steps_drug[1],
            t_steps_drug[1], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_no_GF_IH,
            WMMd_inhibitor)
    df_total_switch_comb = switch_dataframe(60, n_switches, t_steps_drug[2],
            t_steps_drug[2], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH_comb,
            WMMd_inhibitor_comb)

    # Make dataframes for continiously administration
    df_total_GF = continuous_add_IH_df(60, 260, nOC, nOB, nMMd, nMMr,
                        growth_rates, growth_rates_IH, decay_rates,
                        decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH)
    df_total_WMMd = continuous_add_IH_df(60, 260, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_comb = continuous_add_IH_df(60, 260, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)

    # Save the data
    save_dataframe(df_total_switch_GF, 'df_cell_nr_IH_inf_switch_GF_IH.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_switch_WMMd, 'df_cell_nr_IH_inf_switch_WMMd_IH.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_switch_comb, 'df_cell_nr_IH_inf_switch_comb_IH.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_GF, 'df_cell_nr_IH_inf_continuous_GF_IH.csv',
                                             r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_WMMd, 'df_cell_nr_IH_inf_continuous_WMMd_IH.csv',
                                             r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_comb, 'df_cell_nr_IH_inf_continuous_comb_IH.csv',
                                             r'..\data\data_model_nr_IH_inf')

    # Create a Figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))

    # Plot the data without drug holidays in the first plot
    df_total_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 0])
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel('Number', fontsize=11)
    axs[0, 0].set_title(f"Traditional therapy MMd GF IH ")
    axs[0, 0].grid(True)

    # Plot the data without drug holidays in the second plot
    df_total_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 1])
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel(' ')
    axs[0, 1].set_title(r"Traditional therapy $W_{MMd}$ IH")
    axs[0, 1].grid(True)

    # Plot the data without drug holidays in the third plot
    df_total_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 2])
    axs[0, 2].set_xlabel(' ')
    axs[0, 2].set_ylabel(' ')
    axs[0, 2].set_title(r"Traditional therapy MMd GF IH and $W_{MMd}$ IH")
    axs[0, 2].grid(True)

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations', fontsize=11)
    axs[1, 0].set_ylabel('Number', fontsize=11)
    axs[1, 0].set_title(f"Adaptive therapy MMd GF IH")
    axs[1, 0].grid(True)
    plt.grid(True)

    # Plot the data with drug holidays in the fifth plot
    df_total_switch_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 1])
    axs[1, 1].set_xlabel('Generations', fontsize=11)
    axs[1, 1].set_ylabel(' ')
    axs[1, 1].set_title(r"Adaptive therapy $W_{MMd}$ IH")
    axs[1, 1].grid(True)

    # Plot the data with drug holidays in the sixth plot
    df_total_switch_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 2])
    axs[1, 2].set_xlabel('Generations', fontsize=11)
    axs[1, 2].set_ylabel(' ')
    axs[1, 2].set_title(r"Adaptive therapy MMd GF IH and $W_{MMd}$ IH")
    axs[1, 2].grid(True)

    # Create a single legend outside of all plots
    legend_labels = ['Number of OC', 'Number of OB', 'Number of MMd',
                                                                'Number of MMr']
    fig.legend(labels = legend_labels, loc='upper center', ncol=4,
                                                                fontsize='large')
    save_Figure(plt, 'line_plot_cell_nr_IH_inf_AT_MTD',
                                 r'..\visualisation\results_model_nr_IH_inf')
    plt.show()


""" Figure to determine the difference between traditional and adaptive therapy
The interaction matrix is changed to make it more realistic"""
def Figure_continuous_MTD_vs_AT_realistic(n_switches, t_steps_drug):
    """ Function that makes a figure with 6 subplots showing the cell number
    dynamics by traditional therapy (continuous MTD) and adaptive therapy. It
    also prints the number values in the new equilibrium during adaptive therapy.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the number of time steps drugs are administared and the breaks
        are for the different Figures.
    """
    # Set start values
    nOC = 35
    nOB = 40
    nMMd = 35
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.54],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.6, 0.65]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.54],
        [0.3, 0.0, -0.3, -0.3],
        [0.09, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.6, 0.65]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.6, 0.54],
        [0.3, 0.0, -0.3, -0.3],
        [0.22, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.8, 0.65]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.43

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 4.2

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_GF = switch_dataframe(30, n_switches, t_steps_drug[0],
            t_steps_drug[0], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH)
    df_total_switch_WMMd = switch_dataframe(30, n_switches, t_steps_drug[1],
            t_steps_drug[1], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_no_GF_IH,
            WMMd_inhibitor)
    df_total_switch_comb = switch_dataframe(30, n_switches, t_steps_drug[2],
            t_steps_drug[2], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH_comb,
            WMMd_inhibitor_comb)

    # Make dataframes for continiously administration
    df_total_GF = continuous_add_IH_df(30, 300, nOC, nOB, nMMd, nMMr,
                            growth_rates, growth_rates_IH, decay_rates,
                            decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH)
    df_total_WMMd = continuous_add_IH_df(30, 300, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_comb = continuous_add_IH_df(30, 300, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)

    # Print the equilibrium MMd and MMr values caused by the adaptive therapy
    last_MMd_numbers_GF = df_total_switch_GF['nMMd'].tail(int(6))
    average_MMd_number_GF = round(last_MMd_numbers_GF.sum() / 6, 2)
    last_MMr_numbers_GF = df_total_switch_GF['nMMr'].tail(int(6))
    average_MMr_number_GF = round(last_MMr_numbers_GF.sum() / 6, 2)
    print('Adaptive therapy MMd GF IH: nMMd =',average_MMd_number_GF,
                                        'and nMMr =', average_MMr_number_GF)

    last_MMd_numbers_WMMd = df_total_switch_WMMd['nMMd'].tail(int(6))
    average_MMd_number_WMMd = round(last_MMd_numbers_WMMd.sum() / 6, 2)
    last_MMr_numbers_WMMd = df_total_switch_WMMd['nMMr'].tail(int(6))
    average_MMr_number_WMMd = round(last_MMr_numbers_WMMd.sum() / 6, 2)
    print('Adaptive therapy WMMd IH: nMMd =',average_MMd_number_WMMd,
                                        'and nMMr =', average_MMr_number_WMMd)

    last_MMd_numbers_comb = df_total_switch_comb['nMMd'].tail(int(6))
    average_MMd_number_comb = round(last_MMd_numbers_comb.sum() / 6, 2)
    last_MMr_numbers_comb = df_total_switch_comb['nMMr'].tail(int(6))
    average_MMr_number_comb = round(last_MMr_numbers_comb.sum() / 6, 2)
    print('Adaptive therapy IH combination: nMMd =',average_MMd_number_comb,
                                        'and nMMr =', average_MMr_number_comb)

    # Save the data
    save_dataframe(df_total_switch_GF, 'df_cell_nr_IH_inf_switch_GF_IH_r.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_switch_WMMd, 'df_cell_nr_IH_inf_switch_WMMd_IH_r.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_switch_comb, 'df_cell_nr_IH_inf_switch_comb_IH_r.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_GF, 'df_cell_nr_IH_inf_continuous_GF_IH_r.csv',
                                             r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_WMMd, 'df_cell_nr_IH_inf_continuous_WMMd_IH_r.csv',
                                             r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_comb, 'df_cell_nr_IH_inf_continuous_comb_IH_r.csv',
                                             r'..\data\data_model_nr_IH_inf')

    # Create a Figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))

    # Plot the data without drug holidays in the first plot
    df_total_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0, 0])
    axs[0, 0].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[0, 0].set_xlim(1, 302)
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel(r'Cell number ($n_{i}$)', fontsize=12)
    axs[0, 0].set_title(f"Traditional therapy MMd GF IH ", fontsize=14)
    axs[0, 0].grid(True, linestyle='--')

    # Plot the data without drug holidays in the second plot
    df_total_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0, 1])
    axs[0, 1].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[0, 1].set_xlim(1, 302)
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel(' ')
    axs[0, 1].set_title(r"Traditional therapy $W_{MMd}$ IH", fontsize=14)
    axs[0, 1].grid(True, linestyle='--')
    axs[0, 1].set_yticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])

    # Plot the data without drug holidays in the third plot
    df_total_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0, 2])
    axs[0, 2].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[0, 2].set_xlim(1, 302)
    axs[0, 2].set_xlabel(' ')
    axs[0, 2].set_ylabel(' ')
    axs[0, 2].set_title(r"Traditional therapy IH combination", fontsize=14)
    axs[0, 2].grid(True, linestyle='--')

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 0])
    axs[1, 0].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[1, 0].set_xlim(1, 302)
    axs[1, 0].set_xlabel('Generations', fontsize=12)
    axs[1, 0].set_ylabel(r'Cell number ($n_{i}$)', fontsize=12)
    axs[1, 0].set_title(f"Adaptive therapy MMd GF IH", fontsize=14)
    axs[1, 0].grid(True, linestyle='--')
    plt.grid(True)

    # Plot the data with drug holidays in the fifth plot
    df_total_switch_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 1])
    axs[1, 1].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[1, 1].set_xlim(1, 302)
    axs[1, 1].set_xlabel('Generations', fontsize=12)
    axs[1, 1].set_ylabel(' ')
    axs[1, 1].set_title(r"Adaptive therapy $W_{MMd}$ IH", fontsize=14)
    axs[1, 1].grid(True, linestyle='--')

    # Plot the data with drug holidays in the sixth plot
    df_total_switch_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 2])
    axs[1, 2].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[1, 2].set_xlim(1, 302)
    axs[1, 2].set_xlabel('Generations', fontsize=12)
    axs[1, 2].set_ylabel(' ')
    axs[1, 2].set_title(r"Adaptive therapy IH combination", fontsize=14)
    axs[1, 2].grid(True, linestyle='--')

    # Create a single legend outside of all plots
    legend_labels = ['OC number', 'OB number', 'MMd number', 'MMr number',
                                                                    'Therapy']
    fig.legend(labels = legend_labels, loc='upper center', ncol=5,
                                                            fontsize='x-large')
    save_Figure(plt, 'line_plot_cell_nr_IH_inf_AT_MTD_r',
                                 r'..\visualisation\results_model_nr_IH_inf')
    plt.show()



""" Figure to determine the difference between traditional and adaptive therapy
The AT administration and holiday durations depend on the MMd and MMr number"""
def Figure_AT_MMd_MMr_limit(upper_limit_MMd, upper_limit_MMr, limit):
    """ Function that makes a figure with 3 subplots showing the cell number
    dynamics adaptive therapy. The IH administration starts when MMd because
    too high and stops when the MMr becomes too high. It prints the average
    adinistration and holiday duration

    Parameters:
    -----------
    upper_limit_MMd: Int
        The maximum number of MMd, when reached the IH administration starts
    upper_limit_MMr: Int
        The maximum number of MMr, when reached the IH administration stops
    limit: String
        low -> The MMr limit values are low (< 500), high-> the MMd limit values
        are high (> 500). Ensures that figures are saved under different names
    """
    # Set start values
    nOC = 90
    nOB = 160
    nMMd = 100
    nMMr = 10
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.51, 0.51],
        [0.3, 0.0, -0.3, -0.3],
        [0.55, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.5, 0.7]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.51, 0.51],
        [0.3, 0.0, -0.3, -0.3],
        [0.08, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.5, 0.7]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_IH_comb = np.array([
        [0.0, 0.4, 0.51, 0.51],
        [0.3, 0.0, -0.3, -0.3],
        [0.25, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.7, 0.7]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.24

    # determine MMr limit and set correct WMMd IH strength
    if limit == 'high':
        # WMMd inhibitor effect when only WMMd IH is present
        WMMd_inhibitor = 0.72
    else:
        # WMMd inhibitor effect when only WMMd IH is present
        WMMd_inhibitor = 0.55


    # Make dataframe for the different drug hollyday duration values
    df_switch_GF, a_dur_GF, h_dur_GF = dynamics_MMd_MMr_limits(30, 500,
        upper_limit_MMd,  upper_limit_MMr, nOC, nOB, nMMd, nMMr, growth_rates,
        decay_rates, matrix_no_GF_IH, matrix_GF_IH)
    df_switch_WMMd, a_dur_W, h_dur_W = dynamics_MMd_MMr_limits(30, 500,
        upper_limit_MMd, upper_limit_MMr, nOC, nOB, nMMd, nMMr, growth_rates,
        decay_rates, matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_switch_comb, a_dur_comb, h_dur_comb = dynamics_MMd_MMr_limits(30, 500,
        upper_limit_MMd, upper_limit_MMr, nOC, nOB, nMMd, nMMr, growth_rates,
        decay_rates, matrix_no_GF_IH, matrix_IH_comb, WMMd_inhibitor_comb)

    # Print average holiday and administration duration
    print(f"""The average MMd GF IH administration duration is
    {round(a_dur_GF, 0)} generations and the average MMd GF IH holiday duration
    is {round(h_dur_GF, 0)} generations""")
    print(f"""The average WMMd IH administration duration is {round(a_dur_W, 0)}
    generations and the average WMMd IH holiday duration is {round(h_dur_W, 0)}
    generations""")
    print(f"""The average IH combination administration duration is
    {round(a_dur_comb, 0)} generations and the average IH combination holiday
    duration is {round(h_dur_comb, 0)} generations""")

    # Determine if the MMr limit and save the data under the correct name
    if limit == 'low':
        # Save the data
        save_dataframe(df_switch_GF, 'df_cell_nr_GF_IH_AT_MM_limit_l.csv',
                                            r'..\data\data_model_nr_IH_inf')
        save_dataframe(df_switch_WMMd, 'df_cell_nr_W_IH_AT_MM_limit_l.csv',
                                            r'..\data\data_model_nr_IH_inf')
        save_dataframe(df_switch_comb, 'df_cell_nr_IH_comb_AT_MM_limit_l.csv',
                                            r'..\data\data_model_nr_IH_inf')

    if limit == 'middel':
        # Save the data
        save_dataframe(df_switch_GF, 'df_cell_nr_GF_IH_AT_MM_limit.csv',
                                            r'..\data\data_model_nr_IH_inf')
        save_dataframe(df_switch_WMMd, 'df_cell_nr_W_IH_AT_MM_limit.csv',
                                            r'..\data\data_model_nr_IH_inf')
        save_dataframe(df_switch_comb, 'df_cell_nr_IH_comb_AT_MM_limit.csv',
                                            r'..\data\data_model_nr_IH_inf')

    else:
        # Save the data
        save_dataframe(df_switch_GF, 'df_cell_nr_GF_IH_AT_MM_limit_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
        save_dataframe(df_switch_WMMd, 'df_cell_nr_W_IH_AT_MM_limit_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
        save_dataframe(df_switch_comb, 'df_cell_nr_IH_comb_AT_MM_limit_h.csv',
                                            r'..\data\data_model_nr_IH_inf')

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the data of the AT based on the MMd and MMr number (MMd GF IH)
    df_switch_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0])
    axs[0].axvspan(xmin = 30, xmax = 402, color = 'lightgray', alpha = 0.45)
    axs[0].set_xlim(1, 402)
    axs[0].set_xlabel('Generations', fontsize=12)
    axs[0].set_ylabel(r'Cell number ($n_{i}$)', fontsize=12)
    axs[0].set_title(f"Traditional therapy MMd GF IH ", fontsize=14)
    axs[0].grid(True, linestyle='--')

    # Plot the data of the AT based on the MMd and MMr number (WMMd IH)
    df_switch_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1])
    axs[1].axvspan(xmin = 30, xmax = 402, color = 'lightgray', alpha = 0.45)
    axs[1].set_xlim(1, 402)
    axs[1].set_xlabel('Generations', fontsize=12)
    axs[1].set_ylabel(' ')
    axs[1].set_title(r"Traditional therapy $W_{MMd}$ IH", fontsize=14)
    axs[1].grid(True, linestyle='--')

    # Plot the data of the AT based on the MMd and MMr number (IH combination)
    df_switch_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[2])
    axs[2].axvspan(xmin = 30, xmax = 402, color = 'lightgray', alpha = 0.45)
    axs[2].set_xlim(1, 402)
    axs[2].set_xlabel('Generations', fontsize=12)
    axs[2].set_ylabel(' ')
    axs[2].set_title(r"Traditional therapy IH combination", fontsize=14)
    axs[2].grid(True, linestyle='--')

    # Create a single legend outside of all plots
    legend_labels = ['OC number', 'OB number', 'MMd number', 'MMr number',
                                                                    'Therapy']
    fig.legend(labels = legend_labels, loc='upper center', ncol=5,
                                                            fontsize='x-large')

    # Determine if the MMr limit and save the figure under the correct name
    if limit == 'low':
        save_Figure(plt, 'line_plot_cell_nr_AT_l_limit_MMd_MMr',
                                r'..\visualisation\results_model_nr_IH_inf')
    if limit == 'middel':
        save_Figure(plt, 'line_plot_cell_nr_AT_limit_MMd_MMr',
                                r'..\visualisation\results_model_nr_IH_inf')
    else:
        save_Figure(plt, 'line_plot_cell_nr_AT_h_limit_MMd_MMr',
                                 r'..\visualisation\results_model_nr_IH_inf')
    plt.show()


""" Figure to determine the difference between traditional and adaptive therapy.
Shorter holiday and administration periods compared to the original situation"""
def Figure_continuous_MTD_vs_AT_short_a_h(n_switches, t_steps_drug):
    """ Function that makes a figure with 6 subplots showing the cell number
    dynamics by traditional therapy (continuous MTD) and adaptive therapy. The
    holiday and administration periods are short (5 generations).

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the number of time steps drugs are administared and the breaks
        are for the different Figures.
    """
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.13, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.6, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.24, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.18

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.74

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_GF = switch_dataframe(60, n_switches, t_steps_drug[0],
            t_steps_drug[0], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH)
    df_total_switch_WMMd = switch_dataframe(60, n_switches, t_steps_drug[1],
            t_steps_drug[1], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_no_GF_IH,
            WMMd_inhibitor)
    df_total_switch_comb = switch_dataframe(60, n_switches, t_steps_drug[2],
            t_steps_drug[2], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH_comb,
            WMMd_inhibitor_comb)

    # Make dataframes for continiously administration
    df_total_GF = continuous_add_IH_df(60, 260, nOC, nOB, nMMd, nMMr,
                            growth_rates, growth_rates_IH, decay_rates,
                            decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH)
    df_total_WMMd = continuous_add_IH_df(60, 260, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_comb = continuous_add_IH_df(60, 260, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)

    # Save the data
    save_dataframe(df_total_switch_GF, 'df_cell_nr_IH_inf_switch_GF_IH_short_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_switch_WMMd, 'df_cell_nr_IH_inf_switch_WMMd_IH_short_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_switch_comb, 'df_cell_nr_IH_inf_switch_comb_IH_short_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_GF, 'df_cell_nr_IH_inf_continuous_GF_IH_short_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_WMMd, 'df_cell_nr_IH_inf_continuous_WMMd_IH_short_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_comb, 'df_cell_nr_IH_inf_continuous_comb_IH_short_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')

    # Create a Figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))

    # Plot the data without drug holidays in the first plot
    df_total_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 0])
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel('Number', fontsize=11)
    axs[0, 0].set_title(f"Traditional therapy MMd GF IH ")
    axs[0, 0].grid(True)

    # Plot the data without drug holidays in the second plot
    df_total_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 1])
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel(' ')
    axs[0, 1].set_title(r"Traditional therapy $W_{MMd}$ IH")
    axs[0, 1].grid(True)

    # Plot the data without drug holidays in the third plot
    df_total_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 2])
    axs[0, 2].set_xlabel(' ')
    axs[0, 2].set_ylabel(' ')
    axs[0, 2].set_title(r"Traditional therapy IH combination")
    axs[0, 2].grid(True)

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations', fontsize=11)
    axs[1, 0].set_ylabel('Number', fontsize=11)
    axs[1, 0].set_title(f"Adaptive therapy MMd GF IH")
    axs[1, 0].grid(True)
    plt.grid(True)

    # Plot the data with drug holidays in the fifth plot
    df_total_switch_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 1])
    axs[1, 1].set_xlabel('Generations', fontsize=11)
    axs[1, 1].set_ylabel(' ')
    axs[1, 1].set_title(r"Adaptive therapy $W_{MMd}$ IH")
    axs[1, 1].grid(True)

    # Plot the data with drug holidays in the sixth plot
    df_total_switch_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 2])
    axs[1, 2].set_xlabel('Generations', fontsize=11)
    axs[1, 2].set_ylabel(' ')
    axs[1, 2].set_title(r"Adaptive therapy IH combination")
    axs[1, 2].grid(True)

    # Create a single legend outside of all plots
    legend_labels = ['Number of OC', 'Number of OB', 'Number of MMd',
                                                                'Number of MMr']
    fig.legend(labels = legend_labels, loc='upper center', ncol=4, fontsize='large')
    save_Figure(plt, 'line_plot_cell_nr_IH_inf_AT_MTD_short_a_h',
                                 r'..\visualisation\results_model_nr_IH_inf')
    plt.show()


""" Figure to determine the difference between traditional and adaptive therapy
Weaker IHs compared to the original situation"""
def Figure_continuous_MTD_vs_AT_weak_a_h(n_switches, t_steps_drug):
    """ Function that makes a figure with 6 subplots showing the cell number
    dynamics by traditional therapy (continuous MTD) and adaptive therapy.The IHs
    are realively weak.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the number of time steps drugs are administared and the breaks
        are for the different Figures.
    """
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.195, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.6, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.27, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.16

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.57

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_GF = switch_dataframe(60, n_switches, t_steps_drug[0],
            t_steps_drug[0], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH)
    df_total_switch_WMMd = switch_dataframe(60, n_switches, t_steps_drug[1],
            t_steps_drug[1], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_no_GF_IH,
            WMMd_inhibitor)
    df_total_switch_comb = switch_dataframe(60, n_switches, t_steps_drug[2],
            t_steps_drug[2], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH_comb,
            WMMd_inhibitor_comb)

    # Make dataframes for continiously administration
    df_total_GF = continuous_add_IH_df(60, 260, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH)
    df_total_WMMd = continuous_add_IH_df(60, 260, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_comb = continuous_add_IH_df(60, 260, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)

    # Save the data
    save_dataframe(df_total_switch_GF, 'df_cell_nr_IH_inf_switch_GF_IH_weak_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_switch_WMMd, 'df_cell_nr_IH_inf_switch_WMMd_IH_weak_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_switch_comb, 'df_cell_nr_IH_inf_switch_comb_IH_weak_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_GF, 'df_cell_nr_IH_inf_continuous_GF_IH_weak_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_WMMd, 'df_cell_nr_IH_inf_continuous_WMMd_IH_weak_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_comb, 'df_cell_nr_IH_inf_continuous_comb_IH_weak_a_h.csv',
                                            r'..\data\data_model_nr_IH_inf')

    # Create a Figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))

    # Plot the data without drug holidays in the first plot
    df_total_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 0])
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel('Number', fontsize=11)
    axs[0, 0].set_title(f"Traditional therapy MMd GF IH ")
    axs[0, 0].grid(True)

    # Plot the data without drug holidays in the second plot
    df_total_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 1])
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel(' ')
    axs[0, 1].set_title(r"Traditional therapy $W_{MMd}$ IH")
    axs[0, 1].grid(True)

    # Plot the data without drug holidays in the third plot
    df_total_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 2])
    axs[0, 2].set_xlabel(' ')
    axs[0, 2].set_ylabel(' ')
    axs[0, 2].set_title(r"Traditional therapy MMd GF IH and $W_{MMd}$ IH")
    axs[0, 2].grid(True)

    # Plot the data without drug holidays in the fourth plot
    df_total_switch_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations', fontsize=11)
    axs[1, 0].set_ylabel('Number', fontsize=11)
    axs[1, 0].set_title(f"Adaptive therapy MMd GF IH")
    axs[1, 0].grid(True)
    plt.grid(True)

    # Plot the data with drug holidays in the fifth plot
    df_total_switch_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 1])
    axs[1, 1].set_xlabel('Generations', fontsize=11)
    axs[1, 1].set_ylabel(' ')
    axs[1, 1].set_title(r"Adaptive therapy $W_{MMd}$ IH")
    axs[1, 1].grid(True)

    # Plot the data with drug holidays in the sixth plot
    df_total_switch_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 2])
    axs[1, 2].set_xlabel('Generations', fontsize=11)
    axs[1, 2].set_ylabel(' ')
    axs[1, 2].set_title(r"Adaptive therapy MMd GF IH and $W_{MMd}$ IH")
    axs[1, 2].grid(True)

    # Create a single legend outside of all plots
    legend_labels = ['Number of OC', 'Number of OB', 'Number of MMd',
                                                                'Number of MMr']
    fig.legend(labels = legend_labels, loc='upper center', ncol=4,
                                                                fontsize='large')
    save_Figure(plt, 'line_plot_cell_nr_IH_inf_AT_MTD_weak_a_h',
                                 r'..\visualisation\results_model_nr_IH_inf')
    plt.show()


""" 3D plot showing the best IH holiday and administration periods"""
def Figure_3D_MM_numb_IH_add_and_holiday():
    """ Figure that makes three 3D plot that shows the average number of MM for
    different holiday and administration periods of only MMd GF inhibitor, only
    WMMd inhibitor or both. It prints the IH administration periods and holidays
    that caused the lowest total MM number."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.24, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.3

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.42

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM number']
    df_holiday_GF_IH = pd.DataFrame(columns=column_names)

    # Loop over all the t_step values for drug administration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            numb_tumour = minimal_tumour_numb_t_steps(t_steps_drug,
            t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': \
                    int(t_steps_no_drug), 'Generations drug': int(t_steps_drug),
                                             'MM number': float(numb_tumour)}])
            df_holiday_GF_IH = combine_dataframes(df_holiday_GF_IH, new_row_df)

    # Save the data
    save_dataframe(df_holiday_GF_IH, 'df_cell_nr_IH_inf_best_MMd_GF_IH_holiday.csv',
                                             r'..\data\data_model_nr_IH_inf')

    # Determine the axis values
    X_GF_IH, Y_GF_IH, Z_GF_IH = x_y_z_axis_values_3d_plot(df_holiday_GF_IH,
                                                                        "GF IH")

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM number']
    df_holiday_W_IH = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            numb_tumour = minimal_tumour_numb_t_steps(t_steps_drug,
                            t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                            growth_rates_IH, decay_rates, decay_rates_IH,
                            matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': \
                    int(t_steps_no_drug), 'Generations drug': int(t_steps_drug),
                                             'MM number': float(numb_tumour)}])
            df_holiday_W_IH = combine_dataframes(df_holiday_W_IH, new_row_df)

    # Save the data
    save_dataframe(df_holiday_W_IH, 'df_cell_nr_IH_inf_best_WMMd_IH_holiday.csv',
                                             r'..\data\data_model_nr_IH_inf')

    # Determine the axis values
    X_W_IH, Y_W_IH, Z_W_IH = x_y_z_axis_values_3d_plot(df_holiday_W_IH, 'W IH')

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM number']
    df_holiday_comb = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            numb_tumour = minimal_tumour_numb_t_steps(t_steps_drug,
                        t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                        growth_rates_IH, decay_rates, decay_rates_IH,
                        matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': \
                    int(t_steps_no_drug), 'Generations drug': int(t_steps_drug),
                                            'MM number': float(numb_tumour)}])
            df_holiday_comb = combine_dataframes(df_holiday_comb, new_row_df)

    # Save the data
    save_dataframe(df_holiday_comb, 'df_cell_nr_IH_inf_best_MMd_IH_holiday.csv',
                                             r'..\data\data_model_nr_IH_inf')

    # Determine the axis values
    X_comb, Y_comb, Z_comb = x_y_z_axis_values_3d_plot(df_holiday_comb,
                                                                "IH combination")

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), subplot_kw={'projection': '3d'},
                                    gridspec_kw={'hspace': 0.25, 'wspace': 0.25})

    # Plot each subplot
    for i, ax in enumerate(axes.flat, start=1):
        if i == 1:
            surf = ax.plot_surface(X_W_IH, Y_W_IH, Z_W_IH, cmap='coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IH')
            ax.set_ylabel('Generations IH')
            ax.set_zlabel('MM number')
            ax.set_title(r'A) $W_{MMd}$ IH', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 12, azim = -138)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('MM number')

        elif i == 2:
            surf = ax.plot_surface(X_GF_IH, Y_GF_IH, Z_GF_IH, cmap = 'coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IH')
            ax.set_ylabel('Generations IH')
            ax.set_zlabel('MM number')
            ax.set_title('B)  MMd GF IH', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 22, azim = -155)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')

            color_bar.set_label('MM number')

        elif i == 3:
            surf = ax.plot_surface(X_comb, Y_comb, Z_comb, cmap = 'coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IHs')
            ax.set_ylabel('Generations IHs')
            ax.set_zlabel('MM number')
            ax.set_title('C)  IH combination', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 25, azim = -126)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('MM number')

        else:
            # Hide the emply subplot
            ax.axis('off')

    # Add a color bar
    save_Figure(fig, '3d_plot_MM_nr_IH_inf_best_IH_h_a_periods',
                                r'..\visualisation\results_model_nr_IH_inf')
    plt.show()


""" 3D plot showing the best IH strengths """
def Figure_3D_MM_numb_MMd_IH_strength():
    """ 3D plot that shows the average MM number for different MMd GF inhibitor
    and WMMd inhibitor strengths. It prints the IH streghts that caused the
    lowest total MM number."""

    # Set initial parameter values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.65, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.65, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # administration and holiday periods
    t_steps_drug = 4
    t_steps_no_drug = 4

    # Make a dataframe
    column_names = ['Strength WMMd IH', 'Strength MMd GF IH', 'MM number']
    df_holiday = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for strength_WMMd_IH in range(0, 21):

        # Drug inhibitor effect
        WMMd_inhibitor = strength_WMMd_IH / 50
        for strength_MMd_GF_IH in range(0, 21):

            # Change effect of GF of OC on MMd
            matrix_GF_IH[2, 0] = 0.65 - round((strength_MMd_GF_IH / 50), 3)

            # Change how fast the MMr will be stronger than the MMd
            extra_MMr_IH = round(round((WMMd_inhibitor/ 50) + \
                                            (strength_MMd_GF_IH/ 50), 3)/ 8, 3)
            matrix_GF_IH[3, 2] = -0.6 - extra_MMr_IH

            # Determine the minimal tumour size
            numb_tumour = minimal_tumour_numb_t_steps(t_steps_drug,
                t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
                matrix_GF_IH, WMMd_inhibitor)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Strength WMMd IH':\
                        round(strength_WMMd_IH/ 50, 3), 'Strength MMd GF IH': \
                round(strength_MMd_GF_IH/ 50, 3), 'MM number': numb_tumour}])

            df_holiday = combine_dataframes(df_holiday, new_row_df)

    # Save the data
    save_dataframe(df_holiday, 'df_cell_nr_IH_inf_best_MMd_IH_strength.csv',
                                             r'..\data\data_model_nr_IH_inf')


    # Find the drug administration and holiday period causing the lowest MM number
    min_index = df_holiday['MM number'].idxmin()
    strength_WMMd_min = df_holiday.loc[min_index, 'Strength WMMd IH']
    strength_MMd_GF_min = df_holiday.loc[min_index, 'Strength MMd GF IH']
    numb_min = df_holiday.loc[min_index, 'MM number']

    print(f"""Lowest MM number: {numb_min}-> MMd GF IH strength is
        {strength_MMd_GF_min} and WMMd IH strength is {strength_WMMd_min}""")

    # Avoid errors because of the wrong datatype
    df_holiday['Strength WMMd IH'] = pd.to_numeric(df_holiday[\
                                        'Strength WMMd IH'], errors='coerce')
    df_holiday['Strength MMd GF IH'] = pd.to_numeric(df_holiday[\
                                        'Strength MMd GF IH'], errors='coerce')
    df_holiday['MM number'] = pd.to_numeric(df_holiday['MM number'],
                                                                errors='coerce')

    # Make a meshgrid for the plot
    X = df_holiday['Strength WMMd IH'].unique()
    Y = df_holiday['Strength MMd GF IH'].unique()
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((21, 21))

    # Fill the 2D array with the MM number values by looping over each row
    for index, row in df_holiday.iterrows():
        i = int(row.iloc[0]*50)
        j = int(row.iloc[1]*50)
        Z[j, i] = row.iloc[2]

    # Make a 3D Figure
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = '3d')
    surf = ax.plot_surface(X, Y, Z, cmap = 'coolwarm')

    # Add labels
    ax.set_xlabel(r'Strength $W_{MMd}$ IH')
    ax.set_ylabel('Strength MMd GF IH')
    ax.set_zlabel('Number of MM')
    ax.set_title(r"""Average MM number with varying $W_{MMd}$ IH and MMd
    GF IH strengths""")

    # Turn to the right angle
    ax.view_init(elev = 40, azim = -134)

    # Add a color bar
    color_bar = fig.colorbar(surf, shrink = 0.6, location= 'left')
    color_bar.set_label('Number of MM')

    save_Figure(fig, '3d_plot_MM_nr_IH_inf_best_IH_strength',
                                r'..\visualisation\results_model_nr_IH_inf')
    plt.show()


""" Figure to determine the best WMMd IH value """
def Figure_best_WMMd_IH():
    """ Function that shows the effect of different OB and OC cost values for
    different WMMd drug inhibitor values. It also determines the WMMd IH value
    causing the lowest total MM number."""

    # Set initial parameter values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix
    matrix = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.65, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    WMMd_IH_start = 0.2

    # Perform the optimization
    result = minimize(minimal_tumour_numb_WMMd_IH, WMMd_IH_start, args = (nOC,
                                nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
                                decay_rates, decay_rates_IH, matrix, True),
                                bounds=[(0, 0.8)], method='Nelder-Mead')

    # Retrieve the optimal value
    optimal_WMMd_IH = result.x
    print("Optimal value for the WMMd IH:", float(optimal_WMMd_IH[0]),
                                        ', gives tumour number:', result.fun)

    # Make a dictionary
    dict_numb_tumour = {}

    # Loop over the different WMMd_inhibitor values
    for WMMd_inhibitor in range(800):
        WMMd_inhibitor = WMMd_inhibitor/1000
        numb_tumour = minimal_tumour_numb_WMMd_IH(WMMd_inhibitor, nOC, nOB, nMMd,
                        nMMr, growth_rates, growth_rates_IH, decay_rates,
                        decay_rates_IH, matrix, False)
        dict_numb_tumour[WMMd_inhibitor] = numb_tumour

    # Save the data
    save_dictionary(dict_numb_tumour,
            r'..\data\data_model_nr_IH_inf\dict_cell_nr_IH_inf_WMMd_IH.csv')

    # Make lists of the keys and the values
    WMM_IH = list(dict_numb_tumour.keys())
    MM_number = list(dict_numb_tumour.values())

    # Create a Figure
    plt.plot(WMM_IH, MM_number, color='purple')
    plt.title(r"""MM number for various $W_{MMd}$ IH strengths""")
    plt.xlabel(r' $W_{MMd}$ strength')
    plt.ylabel('Number of MM')
    plt.grid(True)
    plt.tight_layout()
    save_Figure(plt, 'line_plot_cell_nr_IH_inf_change_WMMd_IH',
                                 r'..\visualisation\results_model_nr_IH_inf')
    plt.show()


""" Figure to determine the best b_OC_MMd value """
def Figure_best_b_OC_MMd():
    """ Function that makes a Figure that shows the total MM number for different
    b_OC_MMd values. It also determines the b_OC_MMd value causing the lowest
    total number of MM cells"""

    # Set initial parameter values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix
    matrix = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.65, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    b_OC_MMd_start = 0.45

    # Perform the optimization
    result = minimize(minimal_tumour_numb_b_OC_MMd, b_OC_MMd_start, args = (nOC,
                    nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates,
                    decay_rates_IH, matrix, True), bounds=[(0, 0.8)],
                    method='Nelder-Mead')

    # Retrieve the optimal value
    optimal_b_OC_MMd= result.x
    print("Optimal value for b_OC_MMd:", float(optimal_b_OC_MMd[0]),
                                            'gives tumour number:', result.fun)

    # Make a dictionary
    dict_numb_tumour_GF = {}

    # Loop over all the b_OC_MMd values
    for b_OC_MMd in range(800):
        b_OC_MMd = b_OC_MMd/1000

        # Determine the total MM number
        numb_tumour = minimal_tumour_numb_b_OC_MMd(b_OC_MMd, nOC, nOB, nMMd, nMMr,
        growth_rates, growth_rates_IH, decay_rates, decay_rates_IH, matrix, False)
        dict_numb_tumour_GF[b_OC_MMd] = numb_tumour

    # Save the data
    save_dictionary(dict_numb_tumour_GF,
             r'..\data\data_model_nr_IH_inf\dict_cell_nr_IH_inf_b_OC_MMd.csv')

    # Make a list of the keys and one of the values
    b_OC_MMd_values = list(dict_numb_tumour_GF.keys())
    MM_numbers = list(dict_numb_tumour_GF.values())

    # Create the plot
    plt.plot(b_OC_MMd_values, MM_numbers, linestyle='-')
    plt.xlabel(r'$b_{OC, MMd}$ value ')
    plt.ylabel(r'Number of MM')
    plt.title(r'MM number for different $b_{OC, MMd}$ values')
    plt.grid(True)
    save_Figure(plt, 'line_plot_cell_nr_IH_inf_change_b_OC_MMd',
                                r'..\visualisation\results_model_nr_IH_inf')
    plt.show()


""" Figure with a longer IH administration than holiday and the other way around"""
def Figure_duration_a_h_MMd_IH(n_switches, t_steps_drug, t_steps_no_drug):
    """ Function that makes a Figure with two subplots one of the dynamics by a
    longer IH administration than holiday and one of the dynamics by a longer IH
    than administration.

    Parameters:
    -----------
    n_switches: List
        List with the number of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the number of time steps drugs are administared.
    t_steps_no_drug: List
        List with the number of time steps drugs are not administared (holiday).
    """
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_half = np.array([
        [0.0, 0.4, 0.65, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.35, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_half = 0.24

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_1 = switch_dataframe(60, n_switches[0], t_steps_drug[0],
                        t_steps_no_drug[0], nOC, nOB, nMMd, nMMr, growth_rates,
                        growth_rates_IH, decay_rates, decay_rates_IH,
                        matrix_no_GF_IH, matrix_GF_IH_half, WMMd_inhibitor_half)
    df_total_switch_2 = switch_dataframe(60, n_switches[1], t_steps_drug[1],
                        t_steps_no_drug[1], nOC, nOB, nMMd, nMMr, growth_rates,
                        growth_rates_IH, decay_rates, decay_rates_IH,
                        matrix_no_GF_IH, matrix_GF_IH_half, WMMd_inhibitor_half)

    # Save the data
    save_dataframe(df_total_switch_1, 'df_cell_nr_IH_inf_short_a_long_h_MMd_IH.csv',
                                             r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_switch_2,
                            'df_cell_nr_IH_inf_long_a_short_h_MMd_IH.csv.csv',
                                             r'..\data\data_model_nr_IH_inf')

    # Create a Figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    g = 'generations'
    ta = t_steps_drug
    th = t_steps_no_drug

    # Plot the data with short administrations in the first plot
    df_total_switch_1.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
        label=['Number OC', 'Number OB', 'Number MMd', 'Number MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Number of MM')
    axs[0].set_title(f"""Dynamics when the IH administrations lasted {ta[0]} {g}
    and the IH holidays lasted {th[0]} {g}""")
    axs[0].legend(loc = 'upper right')
    axs[0].grid(True)

    # Plot the data with long administrations in the second plot
    df_total_switch_2.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
        label=['Number OC', 'Number OB', 'Number MMd','Number MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Number of MM')
    axs[1].set_title(f"""Dynamics when the IH administrations lasted {ta[1]} {g}
    and the IH holidays lasted {th[1]} {g}""")
    axs[1].legend(loc = 'upper right')
    axs[1].grid(True)
    plt.grid(True)
    save_Figure(plt, 'line_plot_cell_nr_IH_inf_diff_h_and_a_MMd_IH',
                                 r'..\visualisation\results_model_nr_IH_inf')
    plt.show()

"""optimise IH administration duration and holiday duration for MMd GF IH -> WMMd
IH -> holiday """
def minimise_MM_GF_W_h():
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH -> WMMd IH -> holiday -> MMd GF IH
    etc."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    WMMd_inhibitor = 0.4

    # Optimise the administration and holiday durations
    # t_step_IH_strength = [GF IH t, W IH t, h t]
    t_step_IH_strength = [2.733, 3.298, 2.799]
    result = minimize(minimal_tumour_nr_t_3_situations, t_step_IH_strength,
            args=(switch_dataframe_GF_W_h, nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_GF_IH, WMMd_inhibitor), bounds = [(0, None), (0, None),
            (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration and holiday duration')
    print('Repeated order: MMd GF IH -> WMMd IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                        r'..\data\data_model_nr_IH_inf\optimise_GF_W_h.csv')


"""Optimise IH administration duration and holiday duration for WMMd IH -> MMd GF
IH -> holiday """
def minimise_MM_W_GF_h():
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> MMd GF IH -> holiday -> WMMd IH etc."""
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    WMMd_inhibitor = 0.4

    # Optimise the administration and holiday durations
    # t_step_IH_strength = [GF IH t, W IH t, h t]
    t_step_IH_strength = [3.703, 2.416, 3.174]
    result = minimize(minimal_tumour_nr_t_3_situations, t_step_IH_strength,
            args=(switch_dataframe_W_GF_h, nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_GF_IH,WMMd_inhibitor), bounds = [(0, None), (0, None),
            (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration and holiday duration')
    print('Repeated order: WMMd IH -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                        r'..\data\data_model_nr_IH_inf\optimise_W_GF_h.csv')


"""optimise IH administration duration, holiday duration and strength for
MMd GF IH -> WMMd IH -> holiday """
def minimise_MM_GF_W_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH -> WMMd IH -> holiday -> MMd GF IH
    etc. It also determines the best MMd GF IH and WMMd IH strength."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.447, 2.384, 2.536, 0.368, 0.311]
    result = minimize(minimal_tumour_nr_t_3_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_GF_W_h, False,  nOC, nOB, nMMd, nMMr,
            growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.6), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> WMMd IH -> holiday.')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                        r'..\data\data_model_nr_IH_inf\optimise_GF_W_h_IH.csv')


"""Optimise IH administration duration, holiday duration and strength for
WMMd IH -> MMd GF IH -> holiday """
def minimise_MM_W_GF_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> MMd GF IH -> holiday -> WMMd IH etc.
    It also determines the best MMd GF IH and WMMd IH strength."""
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.105, 2.065, 2.007, 0.321, 0.466]
    result = minimize(minimal_tumour_nr_t_3_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_W_GF_h, False,  nOC, nOB, nMMd, nMMr,
            growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.6), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                        r'..\data\data_model_nr_IH_inf\optimise_W_GF_h_IH.csv')


"""Optimise IH administration duration, holiday duration and strength for
MMd GF IH -> holiday -> WMMd IH -> holiday """
def minimise_MM_GF_h_W_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH -> holiday -> WMMd IH -> holiday -> MMd
    GF IH etc. It also determines the best MMd GF IH and WMMd IH strength."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.816, 3.305, 3.620, 0.317, 0.321]
    result = minimize(minimal_tumour_nr_t_3_4_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_GF_h_W_h, False,  nOC, nOB, nMMd, nMMr,
            growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.55), (0, 0.6)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> holiday -> WMMd IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                    r'..\data\data_model_nr_IH_inf\optimise_GF_h_W_h_IH.csv')

"""Optimise IH administration duration, holiday duration and strength for
WMMd IH -> holiday -> MMd GF IH -> holiday """
def minimise_MM_W_h_GF_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> holiday -> MMd GF IH -> holiday ->
    WMMd IH etc. It also determines the best MMd GF IH and WMMd IH strength."""
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [3.878, 3.202, 3.514, 0.392, 0.344]
    result = minimize(minimal_tumour_nr_t_3_4_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_W_h_GF_h, False,  nOC, nOB, nMMd, nMMr,
            growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.55), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> holiday -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                    r'..\data\data_model_nr_IH_inf\optimise_W_h_GF_h_IH.csv')

"""Optimise IH administration duration and holiday duration for WMMd IH ->
IH combination -> MMd GF IH -> holiday"""
def minimise_MM_W_comb_GF_h():
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> IH combination -> MMd GF IH -> holiday
    -> WMMd IH etc."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.2

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.4

    # Optimise the administration and holiday durations
    # t_step_guess = [GF IH t, W IH t, comb t, h t]
    t_step_guess = [3.095, 3.803, 3.763, 3.528]
    result = minimize(minimal_tumour_nr_t_4_situations, t_step_guess, args=(\
        switch_dataframe_W_comb_GF_h, nOC, nOB, nMMd, nMMr, growth_rates,
        growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
        matrix_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb),
        bounds = [(0, None), (0, None), (0, None), (0, None)],
        method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration and holiday duration')
    print('Repeated order: WMMd IH -> IH combination -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best IH combination duration is {result.x[2]} generations
    The best holiday duration is {result.x[3]} generations
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_W_comb_GF_h.csv')

"""Optimise IH administration duration and holiday duration for MMd GF IH->
IH combination -> WMMd IH -> holiday"""
def minimise_MM_GF_comb_W_h():
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH-> IH combination -> WMMd IH -> holiday
    -> MMd GF IH etc."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.2

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.4

    # Optimise the administration and holiday durations
    t_step_guess = [3.795, 3.511, 2.508, 2.098]
    result = minimize(minimal_tumour_nr_t_4_situations, t_step_guess, args=(\
        switch_dataframe_GF_comb_W_h, nOC, nOB, nMMd, nMMr, growth_rates,
        growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
        matrix_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb),
        bounds = [(0, None), (0, None), (0, None), (0, None)],
        method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration and holiday duration')
    print("""Repeated order: MMd GF IH-> IH combination -> WMMd IH -> holiday""")
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best IH combination duration is {result.x[2]} generations
    The best holiday duration is {result.x[3]} generations
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                  r'..\data\data_model_nr_IH_inf\optimise_GF_comb_W_h.csv')

"""Optimise IH administration duration and holiday duration for WMMd IH->
IH combination -> MMd GF IH -> holiday"""
def minimise_MM_W_comb_GF_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> IH combination -> MMd GF IH -> holiday
    -> WMMd IH etc.It also determines the best MMd GF IH and WMMd IH strength."""


    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [2.493, 3.227, 2.509, 3.520, 0.409, 0.085, 0.365, 0.089]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
     args=(switch_dataframe_W_comb_GF_h, False,  nOC, nOB, nMMd, nMMr, growth_rates,
     growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH,
     matrix_GF_IH_comb), bounds = [(0, None), (0, None), (0, None), (0, None),
     (0, 0.6), (0, 0.6), (0, None), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> IH combination -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best IH combination duration is {result.x[2]} generations
    The best holiday duration is {result.x[3]} generations
    The best MMd GF IH strength when given alone is {result.x[4]}
    The best WMMd IH strength when given alone is {result.x[6]}
    The best MMd GF IH strength when given as a combination is {result.x[5]}
    The best WMMd IH strength when given as a combination is {result.x[7]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_W_comb_GF_h_IH.csv')


"""Optimise IH administration duration and holiday duration for MMd GF IH->
IH combination -> WMMd IH -> holiday"""
def minimise_MM_GF_comb_W_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH-> IH combination -> WMMd IH -> holiday
    -> MMd GF IH etc.It also determines the best MMd GF IH and WMMd IH
    strength."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [2.612, 2.135, 2.357, 2.288, 0.267, 0.088, 0.377, 0.106]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
        args=(switch_dataframe_GF_comb_W_h, False,  nOC, nOB, nMMd, nMMr,
        growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
        matrix_no_GF_IH, matrix_GF_IH, matrix_GF_IH_comb), bounds = [(0, None),
        (0, None), (0, None), (0, None), (0, 0.6), (0, 0.6), (0, None),
        (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> IH combination -> WMMd IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best IH combination duration is {result.x[2]} generations
    The best holiday duration is {result.x[3]} generations
    The best MMd GF IH strength when given alone is {result.x[4]}
    The best WMMd IH strength when given alone is {result.x[6]}
    The best MMd GF IH strength when given as a combination is {result.x[5]}
    The best WMMd IH strength when given as a combination is {result.x[7]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_GF_comb_W_h_IH.csv')

"""Optimise IH administration duration and holiday duration for WMMd IH -> WMMd
IH + MMd GF IH -> MMd GF IH -> holiday"""
def minimise_MM_W_WandGF_GF_h():
    """Function that determines the best IH administration durations and holliday
    durations when the order is WMMd IH -> WMMd IH + MMd GF IH -> MMd GF IH ->
    holiday -> WMMd IH etc."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.3, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.3, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    WMMd_inhibitor = 0.3

    # Optimize the administration and holliday durations and the IH stregths
    # t_step_IH_strength = [GF IH t, W IH t, both IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.133, 2.662, 3.969, 3.900]
    result = minimize(minimal_tumour_nr_t_4_sit_equal, t_step_IH_strength,
        args=(switch_dataframe_W_WandGF_GF_h, nOC, nOB, nMMd, nMMr, growth_rates,
        growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
        matrix_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor), bounds = [(0, None),
        (0, None), (0, None), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration and holiday duration')
    print('Repeated order: WMMd IH -> WMMd IH + MMd GF IH -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best WMMd IH + MMd GF IH add duration is {result.x[2]} generations
    The best holliday duration is {result.x[3]} generations
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_W_WandGF_GF_h.csv')


"""Optimise IH administration duration and holiday duration for MMd GF IH-> MMd
GF IH + WMMd IH -> WMMd IH -> holiday"""
def minimise_MM_GF_GFandW_W_h():
    """Function that determines the best IH administration durations and holliday
    durations when the order is MMd GF IH-> MMd GF IH + WMMd IH -> WMMd IH ->
    holiday -> MMd GF IH etc."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.3, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.3, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    WMMd_inhibitor = 0.3

    # Optimize the administration and holliday durations and the IH stregths
    # t_step_IH_strength = [GF IH t, W IH t, both IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [3.353, 2.355, 3.171, 2.999]
    result = minimize(minimal_tumour_nr_t_4_sit_equal, t_step_IH_strength,
        args=(switch_dataframe_GF_WandGF_W_h, nOC, nOB, nMMd, nMMr, growth_rates,
        growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
        matrix_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor), bounds = [(0, None),
        (0, None), (0, None), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration and holiday duration')
    print('Repeated order: MMd GF IH -> WMMd IH + MMd GF IH -> WMMd IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best WMMd IH + MMd GF IH add duration is {result.x[2]} generations
    The best holliday duration is {result.x[3]} generations
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_GF_WandGF_W_h.csv')

"""Optimise IH administration duration and holiday duration for WMMd IH -> WMMd
IH + MMd GF IH -> MMd GF IH -> holiday"""
def minimise_MM_W_WandGF_GF_h_IH():
    """Function that determines the best IH administration durations and holliday
    durations when the order is WMMd IH -> WMMd IH + MMd GF IH -> MMd GF IH ->
    holiday -> WMMd IH etc.It also determines the best MMd GF IH and WMMd IH
    strength."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimize the administration and holliday durations and the IH stregths
    # t_step_IH_strength = [GF IH t, W IH t, both IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [3.253, 3.920, 3.483, 2.302, 0.428, 0.474]
    result = minimize(minimal_tumour_nr_t_4_sit_equal_IH, t_step_IH_strength,
        args=(switch_dataframe_W_WandGF_GF_h, False,  nOC, nOB, nMMd, nMMr,
        growth_rates, growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
        matrix_GF_IH, matrix_GF_IH_comb), bounds = [(0, None), (0, None),
        (0, None), (0, None), (0, None), (0.0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> WMMd IH + MMd GF IH -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best WMMd IH + MMd GF IH add duration is {result.x[2]} generations
    The best holliday duration is {result.x[3]} generations
    The best MMd GF IH strength is {result.x[4]}
    The best WMMd IH strength is {result.x[5]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_W_WandGF_GF_h_IH.csv')


"""Optimise IH administration duration and holiday duration for MMd GF IH-> MMd
GF IH + WMMd IH -> WMMd IH -> holiday"""
def minimise_MM_GF_GFandW_W_h_IH():
    """Function that determines the best IH administration durations and holliday
    durations when the order is MMd GF IH-> MMd GF IH + WMMd IH -> WMMd IH ->
    holiday -> MMd GF IH etc.It also determines the best MMd GF IH and WMMd IH
    strength."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimize the administration and holliday durations and the IH stregths
    # t_step_IH_strength = [GF IH t, W IH t, both IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.331, 3.349, 3.141, 3.714, 0.423, 0.329]
    result = minimize(minimal_tumour_nr_t_4_sit_equal_IH, t_step_IH_strength,
        args=(switch_dataframe_GF_WandGF_W_h, False,  nOC, nOB, nMMd, nMMr,
        growth_rates, growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
        matrix_GF_IH, matrix_GF_IH_comb), bounds = [(0, None), (0, None),
        (0, None), (0, None), (0, None), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> WMMd IH + MMd GF IH -> WMMd IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best WMMd IH + MMd GF IH add duration is {result.x[2]} generations
    The best holliday duration is {result.x[3]} generations
    The best MMd GF IH strength is {result.x[4]}
    The best WMMd IH strength is {result.x[5]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_GF_WandGF_W_h_IH.csv')

"""Optimise IH administration duration and holiday duration for WMMd IH->
IH combination -> holiday"""
def minimise_MM_W_comb_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> IH combination -> holiday -> WMMd IH
    etc. It also determines the best MMd GF IH and WMMd IH strength."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [W IH t, comb t, h t, comb GF IH s, W IH s, comb W IH s]
    t_step_IH_strength = [2.198, 2.988, 2.064, 0.212, 0.458, 0.183]
    result = minimize(minimal_tumour_nr_t_3_sit_W_IH, t_step_IH_strength,
        args=(switch_dataframe_W_comb_h, False,  nOC, nOB, nMMd, nMMr,
        growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
        matrix_no_GF_IH, matrix_GF_IH_comb), bounds = [(0, None), (0, None),
        (0, None), (0, 0.55), (0, None), (0, None), ], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> IH combination  -> holiday')
    print(f"""The best WMMd IH add duration is {result.x[0]} generations
    The best IH combination duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best WMMd IH strength when given alone is {result.x[4]}
    The best MMd GF IH strength when given as a combination is {result.x[3]}
    The best WMMd IH strength when given as a combination is {result.x[5]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_W_comb_h_IH.csv')


"""Optimise IH administration duration and holiday duration for MMd GF IH->
IH combination -> holiday"""
def minimise_MM_GF_comb_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH-> IH combination -> holiday -> MMd GF
    IH etc. It also determines the best MMd GF IH and WMMd IH strength."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, comb t, h t, GF IH s, comb GF IH s
    # comb W IH s]
    t_step_IH_strength = [2.203, 2.479, 3.175, 0.474, 0.198, 0.207]
    result = minimize(minimal_tumour_nr_t_3_sit_GF_IH, t_step_IH_strength,
      args=(switch_dataframe_GF_comb_h, False, nOC, nOB, nMMd, nMMr,
      growth_rates, growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
      matrix_GF_IH, matrix_GF_IH_comb), bounds = [(0, None), (0, None),
      (0, None), (0, 0.55), (0, 0.55), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> IH combination -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best IH combination duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength when given alone is {result.x[3]}
    The best MMd GF IH strength when given as a combination is {result.x[4]}
    The best WMMd IH strength when given as a combination is {result.x[5]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_GF_comb_h_IH.csv')

"""Optimise IH administration duration and holiday duration for WMMd IH -> MMd GF
IH -> holiday by different WMMd IH strengths"""
def minimise_MM_W_GF_h_changing_W_IH(growth_rates, growth_rates_IH, decay_rates,
                                                     decay_rates_IH, filename):
    """Function that determines the best IH administration durations and holiday
    durations for different WMMd IH values when the order is WMMd IH -> MMd GF
    IH -> holiday -> WMMd IH etc.

    Parameters:
    -----------
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    filename: String
        The name of the file in which the generated data is saved.

    Returns:
    --------
    df_W_GF_h_change_W: Dataframe
        Dataframe with the MM number for different WMMd IH strengths.
    """

    # Make a datframe
    column_names = ['W IH strength', 'MMd GF IH duration', 'WMMd IH duration',
                                                'Holiday duration', 'MM number']
    df_W_GF_h_change_W = pd.DataFrame(columns = column_names)

    for i in range(20):

        # Calculate and print the strength of the MMd GF IH
        W_IH = 0.2 + (round(i)/100)
        print(f'The WMMd IH stength is {W_IH}')

        # Set start values
        nOC = 20
        nOB = 30
        nMMd = 20
        nMMr = 5

        # Payoff matrix when no drugs are present
        matrix_no_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Payoff matrix when only GF inhibitor drugs are present
        matrix_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.2, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Optimise the administration and holiday durations
        # t_step_IH_strength = [GF IH t, W IH t, h t]
        t_step_IH_strength = [3.000, 2.001, 3.011]
        result = minimize(minimal_tumour_nr_t_3_situations, t_step_IH_strength,
                args=(switch_dataframe_W_GF_h, nOC, nOB, nMMd, nMMr, growth_rates,
                growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
                matrix_GF_IH, W_IH), bounds = [(0, None), (0, None), (0, None)],
                method='Nelder-Mead')

        # Print the results
        print('Optimising IH administration duration and holiday duration')
        print('Repeated order: WMMd IH -> MMd GF IH -> holiday')
        print(f"""The best MMd GF IH add duration is {result.x[0]} generations
        The best WMMd IH add duration is {result.x[1]} generations
        The best holiday duration is {result.x[2]} generations
        --> gives a MM number of {result.fun}""")

        # Add results to the dataframe
        new_row_df = pd.DataFrame([{'W IH strength': W_IH, 'MMd GF IH duration':\
                 result.x[0], 'WMMd IH duration':result.x[1], 'Holiday duration': \
                 result.x[2], 'MM number':result.fun}])
        df_W_GF_h_change_W = combine_dataframes(df_W_GF_h_change_W, new_row_df)

    # Save the data
    save_dataframe(df_W_GF_h_change_W, filename, r'..\data\data_model_nr_IH_inf')

    return(df_W_GF_h_change_W)


"""Optimise IH administration duration and holiday duration for WMMd IH -> MMd GF
IH -> holiday by different MMd GF IH strengths"""
def minimise_MM_W_GF_h_changing_GF_IH(growth_rates, growth_rates_IH, decay_rates,
                                                     decay_rates_IH, filename):
    """Function that determines the best IH administration durations and holiday
    durations for different MMd GF IH values when the order is WMMd IH -> MMd GF
    IH -> holiday -> WMMd IH etc.

    Parameters:
    -----------
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    filename: String
        The name of the file in which the generated data is saved.

    Returns:
    --------
    df_W_GF_h_change_GF: Dataframe
        Dataframe with the MM number for different MMd GF IH strengths.
    """

    # Make a datframe
    column_names = ['GF IH strength', 'MMd GF IH duration', 'WMMd IH duration',
                                                'Holiday duration', 'MM number']
    df_W_GF_h_change_GF = pd.DataFrame(columns = column_names)

    for i in range(14):

        # Calculate and print the strength of the MMd GF IH
        GF_IH = round(0.08 + (i/100), 2)
        print(f'The MMd GF IH stength is {GF_IH}')

        # Set start values
        nOC = 20
        nOB = 30
        nMMd = 20
        nMMr = 5

        # Payoff matrix when no drugs are present
        matrix_no_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Payoff matrix when only GF inhibitor drugs are present
        matrix_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6 -GF_IH, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        WMMd_inhibitor = 0.4

        # Optimise the administration and holiday durations
        # t_step_IH_strength = [GF IH t, W IH t, h t]
        t_step_IH_strength = [3.000, 2.001, 3.011]
        result = minimize(minimal_tumour_nr_t_3_situations, t_step_IH_strength,
                args=(switch_dataframe_W_GF_h, nOC, nOB, nMMd, nMMr, growth_rates,
                growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
                matrix_GF_IH,WMMd_inhibitor), bounds = [(0, None), (0, None),
                (0, None)], method='Nelder-Mead')

        # Print the results
        print('Optimising IH administration duration and holiday duration')
        print('Repeated order: WMMd IH -> MMd GF IH -> holiday')
        print(f"""The best MMd GF IH add duration is {result.x[0]} generations
        The best WMMd IH add duration is {result.x[1]} generations
        The best holiday duration is {result.x[2]} generations
        --> gives a MM number of {result.fun}""")

        # Add results to the dataframe
        new_row_df = pd.DataFrame([{'GF IH strength': GF_IH, 'MMd GF IH duration':\
                 result.x[0], 'WMMd IH duration':result.x[1], 'Holiday duration': \
                 result.x[2], 'MM number':result.fun}])
        df_W_GF_h_change_GF = combine_dataframes(df_W_GF_h_change_GF, new_row_df)

    # Save the data
    save_dataframe(df_W_GF_h_change_GF, filename, r'..\data\data_model_nr_IH_inf')

    return(df_W_GF_h_change_GF)

"""optimise IH administration duration and holiday duration for MMd GF IH -> WMMd
IH -> holiday by different WMMd IH strengths"""
def minimise_MM_GF_W_h_changing_W_IH(growth_rates, growth_rates_IH, decay_rates,
                                                     decay_rates_IH, filename):
    """Function that determines the best IH administration durations and holiday
    durations for different WMMd IH values when the order is MMd GF IH -> WMMd IH
    -> holiday -> MMd GF IH etc.

    Parameters:
    -----------
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    filename: String
        The name of the file in which the generated data is saved.

    Returns:
    --------
    df_GF_W_h_change_W: Dataframe
        Dataframe with the MM number for different WMMd IH strengths.
    """

    # Make a datframe
    column_names = ['W IH strength', 'MMd GF IH duration', 'WMMd IH duration',
                                                'Holiday duration', 'MM number']
    df_GF_W_h_change_W = pd.DataFrame(columns = column_names)

    for i in range(20):

        # Calculate and print the strength of the WMMd IH
        W_IH = 0.2 + (round(i)/100)
        print(f'The WMMd IH stength is {W_IH}')

        # Set start values
        nOC = 20
        nOB = 30
        nMMd = 20
        nMMr = 5

        # Payoff matrix when no drugs are present
        matrix_no_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Payoff matrix when only GF inhibitor drugs are present
        matrix_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.2, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Optimise the administration and holiday durations
        # t_step_IH_strength = [GF IH t, W IH t, h t]
        t_step_IH_strength = [3.000, 2.000, 3.000]
        result = minimize(minimal_tumour_nr_t_3_situations, t_step_IH_strength,
                args=(switch_dataframe_GF_W_h, nOC, nOB, nMMd, nMMr, growth_rates,
                growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
                matrix_GF_IH, W_IH), bounds = [(0, None), (0, None), (0, None)],
                method='Nelder-Mead')

        # Print the results
        print('Optimising IH administration duration and holiday duration')
        print('Repeated order: MMd GF IH -> WMMd IH -> holiday')
        print(f"""The best MMd GF IH add duration is {result.x[0]} generations
        The best WMMd IH add duration is {result.x[1]} generations
        The best holiday duration is {result.x[2]} generations
        --> gives a MM number of {result.fun}""")

        # Add results to the dataframe
        new_row_df = pd.DataFrame([{'W IH strength': W_IH, 'MMd GF IH duration':\
                 result.x[0], 'WMMd IH duration':result.x[1],
                 'Holiday duration': result.x[2], 'MM number':result.fun}])
        df_GF_W_h_change_W = combine_dataframes(df_GF_W_h_change_W, new_row_df)
    # Save the data
    save_dataframe(df_GF_W_h_change_W, filename, r'..\data\data_model_nr_IH_inf')

    return(df_GF_W_h_change_W)

"""optimise IH administration duration and holiday duration for MMd GF IH -> WMMd
IH -> holiday by different MMd GF IH strengths"""
def minimise_MM_GF_W_h_changing_GF_IH(growth_rates, growth_rates_IH, decay_rates,
                                                     decay_rates_IH, filename):
    """Function that determines the best IH administration durations and holiday
    durations for different MMd GF IH values when the order is MMd GF IH -> WMMd
    IH -> holiday -> MMd GF IH etc.

    Parameters:
    -----------
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    filename: String
        The name of the file in which the generated data is saved.

    Returns:
    --------
    df_GF_W_h_change_GF: Dataframe
        Dataframe with the MM number for different MMd GF IH strengths.
    """

    # Make a datframe
    column_names = ['GF IH strength', 'MMd GF IH duration', 'WMMd IH duration',
                                            'Holiday duration', 'MM number']
    df_GF_W_h_change_GF = pd.DataFrame(columns = column_names)

    for i in range(14):

        # Calculate and print the strength of the MMd GF IH
        GF_IH = round(0.08 + (i)/100, 2)
        print(f'The MMd GF IH stength is {GF_IH}')

        # Set start values
        nOC = 20
        nOB = 30
        nMMd = 20
        nMMr = 5

        # Payoff matrix when no drugs are present
        matrix_no_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Payoff matrix when only GF inhibitor drugs are present
        matrix_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6 - GF_IH, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        WMMd_inhibitor = 0.4

        # Optimise the administration and holiday durations
        # t_step_IH_strength = [GF IH t, W IH t, h t]
        t_step_IH_strength = [3.000, 2.000, 3.000]
        result = minimize(minimal_tumour_nr_t_3_situations, t_step_IH_strength,
                args=(switch_dataframe_GF_W_h, nOC, nOB, nMMd, nMMr, growth_rates,
                growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
                matrix_GF_IH, WMMd_inhibitor), bounds = [(0, None), (0, None),
                (0, None)], method='Nelder-Mead')

        # Print the results
        print('Optimising IH administration duration and holiday duration')
        print('Repeated order: MMd GF IH -> WMMd IH -> holiday')
        print(f"""The best MMd GF IH add duration is {result.x[0]} generations
        The best WMMd IH add duration is {result.x[1]} generations
        The best holiday duration is {result.x[2]} generations
        --> gives a MM number of {result.fun}""")

        # Add results to the dataframe
        new_row_df = pd.DataFrame([{'GF IH strength': GF_IH, 'MMd GF IH duration':\
            result.x[0], 'WMMd IH duration':result.x[1], 'Holiday duration': \
            result.x[2], 'MM number':result.fun}])
        df_GF_W_h_change_GF = combine_dataframes(df_GF_W_h_change_GF, new_row_df)

    # Save the data
    save_dataframe(df_GF_W_h_change_GF, filename, r'..\data\data_model_nr_IH_inf')

    return(df_GF_W_h_change_GF)


"""optimise IH administration duration, holiday duration and strength for
MMd GF IH -> WMMd IH -> holiday where the weight of the MMr relative to the MMd
can be specified """
def minimise_MM_GF_W_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH -> WMMd IH -> holiday -> MMd GF IH
    etc. It also determines the best MMd GF IH and WMMd IH strength. The weight
    of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.588, 2.778, 2.126, 0.351, 0.428]
    result = minimize(minimal_tumour_nr_t_3_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_GF_W_h, relative_weight_MMr, nOC, nOB, nMMd,
            nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.6), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> WMMd IH -> holiday.')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                        r'..\data\data_model_nr_IH_inf\optimise_GF_W_h_IH_w.csv')


"""Optimise IH administration duration, holiday duration and strength for
WMMd IH -> MMd GF IH -> holiday """
def minimise_MM_W_GF_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> MMd GF IH -> holiday -> WMMd IH etc.
    It also determines the best MMd GF IH and WMMd IH strength. The weight of the
    MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.200, 2.339, 2.286, 0.476,0.459]
    result = minimize(minimal_tumour_nr_t_3_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_W_GF_h, relative_weight_MMr, nOC, nOB, nMMd,
            nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.6), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a weighted MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_W_GF_h_IH_w.csv')


"""Optimise IH administration duration and holiday duration for WMMd IH->
IH combination -> MMd GF IH -> holiday where the weight of the MMr relative to the
MMd can be specified"""
def minimise_MM_W_comb_GF_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> IH combination -> MMd GF IH -> holiday
    -> WMMd IH etc.It also determines the best MMd GF IH and WMMd IH strength.
    The weight of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [3.765, 2.725, 2.09, 2.775, 0.411, 0.119, 0.439, 0.08]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
        args=(switch_dataframe_W_comb_GF_h, relative_weight_MMr, nOC, nOB, nMMd,
        nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
        matrix_no_GF_IH, matrix_GF_IH, matrix_GF_IH_comb), bounds = [(0, None),
        (0, None), (0, None), (0, None), (0.015, 0.6), (0, 0.6), (0, None),
        (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> IH combination -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best IH combination duration is {result.x[2]} generations
    The best holiday duration is {result.x[3]} generations
    The best MMd GF IH strength when given alone is {result.x[4]}
    The best WMMd IH strength when given alone is {result.x[6]}
    The best MMd GF IH strength when given as a combination is {result.x[5]}
    The best WMMd IH strength when given as a combination is {result.x[7]}
    --> gives a weighted MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
        r'..\data\data_model_nr_IH_inf\optimise_W_comb_GF_h_IH_w.csv')


"""Optimise IH administration duration and holiday duration for MMd GF IH->
IH combination -> WMMd IH -> holiday where the weight of the MMr relative to the
MMd can be specified"""
def minimise_MM_GF_comb_W_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH-> IH combination -> WMMd IH -> holiday
    -> MMd GF IH etc.It also determines the best MMd GF IH and WMMd IH
    strength. The weight of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [3.293, 3.578, 2.209, 3.215, 0.443, 0.095, 0.365, 0.081]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
        args=(switch_dataframe_GF_comb_W_h, relative_weight_MMr, nOC, nOB, nMMd,
        nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
        matrix_no_GF_IH, matrix_GF_IH, matrix_GF_IH_comb), bounds = [(0, None),
        (0, None), (0, None), (0, None), (0, 0.6), (0, 0.6), (0, None), (0, None)],
        method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> IH combination -> WMMd IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best IH combination duration is {result.x[2]} generations
    The best holiday duration is {result.x[3]} generations
    The best MMd GF IH strength when given alone is {result.x[4]}
    The best WMMd IH strength when given alone is {result.x[6]}
    The best MMd GF IH strength when given as a combination is {result.x[5]}
    The best WMMd IH strength when given as a combination is {result.x[7]}
    --> gives a weighted MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
        r'..\data\data_model_nr_IH_inf\optimise_GF_comb_W_h_IH_w.csv')

"""Optimise IH administration duration, holiday duration and strength for WMMd
IH -> holiday -> MMd GF IH -> holiday where the weight of the MMr relative to the
MMd can be specified"""
def minimise_MM_W_h_GF_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> holiday -> MMd GF IH -> holiday ->
    WMMd IH etc. It also determines the best MMd GF IH and WMMd IH strength.
    The weight of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.463, 3.976, 3.597, 0.468, 0.324]
    result = minimize(minimal_tumour_nr_t_3_4_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_W_h_GF_h, relative_weight_MMr, nOC, nOB, nMMd,
            nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.6), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> holiday -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
            r'..\data\data_model_nr_IH_inf_mutation\optimise_W_h_GF_h_IH_w.csv')

"""Optimise IH administration duration, holiday duration and strength for MMd GF
IH -> holiday -> WMMd IH -> holiday where the weight of the MMr relative to the
MMd can be specified """
def minimise_MM_GF_h_W_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH -> holiday -> WMMd IH -> holiday -> MMd
    GF IH etc. It also determines the best MMd GF IH and WMMd IH strength. The
    weight of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.628, 2.284, 2.12, 0.42, 0.311]
    result = minimize(minimal_tumour_nr_t_3_4_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_GF_h_W_h, relative_weight_MMr, nOC, nOB, nMMd,
            nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.55), (0, 0.6)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> holiday -> WMMd IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
            r'..\data\data_model_nr_IH_inf_mutation\optimise_GF_h_W_h_IH_w.csv')

"""Optimise IH administration duration and holiday duration for WMMd IH -> WMMd
IH + MMd GF IH -> MMd GF IH -> holiday where the weight of the MMr relative to
the MMd can be specified """
def minimise_MM_W_WandGF_GF_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holliday
    durations when the order is WMMd IH -> WMMd IH + MMd GF IH -> MMd GF IH ->
    holiday -> WMMd IH etc.It also determines the best MMd GF IH and WMMd IH
    strength. The weight of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.3, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.3, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimize the administration and holliday durations and the IH stregths
    # t_step_IH_strength = [GF IH t, W IH t, both IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.816, 3.322, 3.489, 2.026, 0.344, 0.494]
    result = minimize(minimal_tumour_nr_t_4_sit_equal_IH, t_step_IH_strength,
        args=(switch_dataframe_W_WandGF_GF_h, relative_weight_MMr, nOC, nOB, nMMd,
        nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
        matrix_no_GF_IH, matrix_GF_IH, matrix_GF_IH_comb), bounds = [(0, None),
        (0, None), (0, None), (0, None), (0, None), (0.0, None)],
        method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> WMMd IH + MMd GF IH -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best WMMd IH + MMd GF IH add duration is {result.x[2]} generations
    The best holliday duration is {result.x[3]} generations
    The best MMd GF IH strength is {result.x[4]}
    The best WMMd IH strength is {result.x[5]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
       r'..\data\data_model_nr_IH_inf\optimise_W_WandGF_GF_h_IH_w.csv')


"""Optimise IH administration duration and holiday duration for MMd GF IH-> MMd
GF IH + WMMd IH -> WMMd IH -> holiday where the weight of the MMr relative to
the MMd can be specified """
def minimise_MM_GF_GFandW_W_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holliday
    durations when the order is MMd GF IH-> MMd GF IH + WMMd IH -> WMMd IH ->
    holiday -> MMd GF IH etc.It also determines the best MMd GF IH and WMMd IH
    strength. The weight of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.3, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.3, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimize the administration and holliday durations and the IH stregths
    # t_step_IH_strength = [GF IH t, W IH t, both IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [3.006, 2.13, 3.152, 3.598, 0.344, 0.342]
    result = minimize(minimal_tumour_nr_t_4_sit_equal_IH, t_step_IH_strength,
        args=(switch_dataframe_GF_WandGF_W_h, relative_weight_MMr, nOC, nOB, nMMd,
        nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
        matrix_no_GF_IH, matrix_GF_IH, matrix_GF_IH_comb), bounds = [(0, None),
        (0, None), (0, None), (0, None), (0, None), (0, None)],
        method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> WMMd IH + MMd GF IH -> WMMd IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best WMMd IH + MMd GF IH add duration is {result.x[2]} generations
    The best holliday duration is {result.x[3]} generations
    The best MMd GF IH strength is {result.x[4]}
    The best WMMd IH strength is {result.x[5]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
       r'..\data\data_model_nr_IH_inf\optimise_GF_WandGF_W_h_IH_w.csv')

"""Optimise IH administration duration and holiday duration for WMMd IH->
IH combination -> holiday where the weight of the MMr relative to
the MMd can be specified """
def minimise_MM_W_comb_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> IH combination -> holiday -> WMMd IH
    etc. It also determines the best MMd GF IH and WMMd IH strength. The weight
    of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
    The weight of the MMr relative to that of the MMd.
    """

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
    [0.0, 0.4, 0.65, 0.55],
    [0.3, 0.0, -0.3, -0.3],
    [0.6, 0.0, 0.2, 0.0],
    [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
    [0.0, 0.4, 0.65, 0.55],
    [0.3, 0.0, -0.3, -0.3],
    [0.4, 0.0, 0.2, 0.0],
    [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [W IH t, comb t, h t, comb GF IH s, W IH s, comb W IH s]
    t_step_IH_strength = [2.362, 3.469, 2.047, 0.09, 0.365, 0.112]
    result = minimize(minimal_tumour_nr_t_3_sit_W_IH, t_step_IH_strength,
    args=(switch_dataframe_W_comb_h, relative_weight_MMr, nOC, nOB, nMMd,
    nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
    matrix_no_GF_IH, matrix_GF_IH_comb), bounds = [(0, None), (0, None),
    (0, None), (0, 0.55), (0, None), (0, None), ], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> IH combination  -> holiday')
    print(f"""The best WMMd IH add duration is {result.x[0]} generations
    The best IH combination duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best WMMd IH strength when given alone is {result.x[4]}
    The best MMd GF IH strength when given as a combination is {result.x[3]}
    The best WMMd IH strength when given as a combination is {result.x[5]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
         r'..\data\data_model_nr_IH_inf\optimise_W_comb_h_IH_w.csv')


"""Optimise IH administration duration and holiday duration for MMd GF IH->
IH combination -> holiday where the weight of the MMr relative to the MMd can be
specified """
def minimise_MM_GF_comb_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH-> IH combination -> holiday -> MMd GF
    IH etc. It also determines the best MMd GF IH and WMMd IH strength.The weight
    of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
     The weight of the MMr relative to that of the MMd.
    """

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
     [0.0, 0.4, 0.65, 0.55],
     [0.3, 0.0, -0.3, -0.3],
     [0.6, 0.0, 0.2, 0.0],
     [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
     [0.0, 0.4, 0.65, 0.55],
     [0.3, 0.0, -0.3, -0.3],
     [0.2, 0.0, 0.2, 0.0],
     [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
     [0.0, 0.4, 0.65, 0.55],
     [0.3, 0.0, -0.3, -0.3],
     [0.4, 0.0, 0.2, 0.0],
     [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, comb t, h t, GF IH s, comb GF IH s
    # comb W IH s]
    t_step_IH_strength = [3.39, 3.782, 2.038, 0.36, 0.084, 0.119]
    result = minimize(minimal_tumour_nr_t_3_sit_GF_IH, t_step_IH_strength,
     args=(switch_dataframe_GF_comb_h, relative_weight_MMr, nOC, nOB, nMMd,
     nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
     matrix_no_GF_IH, matrix_GF_IH, matrix_GF_IH_comb), bounds = [(0, None),
     (0, None), (0, None), (0, 0.55), (0, 0.55), (0, None)],
     method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> IH combination -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best IH combination duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength when given alone is {result.x[3]}
    The best MMd GF IH strength when given as a combination is {result.x[4]}
    The best WMMd IH strength when given as a combination is {result.x[5]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
         r'..\data\data_model_nr_IH_inf\optimise_GF_comb_h_IH_w.csv')

if __name__ == "__main__":
    main()
