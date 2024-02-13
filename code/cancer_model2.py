"""
Author:       Eva Nieuwenhuis
University:   UvA
Student id':  13717405
Description:  Code with the model that simulates the dynamics in the multiple myeloma
              (MM) microenvironment with three cell types: MM cells, osteoblasts (OBs)
              and osteoclasts (OCs). The model is a public goods game in the framework
              of evolutionary game theory with collective interactions and nonlinear
              benefits.
              The model is based on a in the paper of Sartakhti et al., 2018.

Sartakhti, J. S., Manshaei, M. H., & Archetti, M. (2018). Game Theory of Tumor–Stroma
Interactions in Multiple Myeloma: Effect of nonlinear benefits. Games, 9(2), 32.
https://doi.org/10.3390/g9020032
"""

import math
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import ternary
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy.integrate import odeint


def fitness_WOC(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix):
    """
    Function that calculates the fitness of an osteoclast in a population.

    Parameters:
    -----------
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMr: Float
        Frequency of the resistant MM cells.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix : Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WOC : Float
        Fitness of an OC.

    Example:
    -----------
    >>> fitness_WOC(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]))
    0.10859999999999997
    """
    # Extract the needed matrix values
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[0, 2]
    d = matrix[0, 3]

    # Calculate the fitness value
    WOC = (a*xOC*cOC + b*xOB*cOB + c*xMMd*cMMd + d* xMMr *cMMr)*(N - 1)/N - cOC
    return WOC

def fitness_WOB(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix):
    """
    Function that calculates the fitness of an osteoblast in a population.

    Parameters:
    -----------
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMr: Float
        Frequency of the resistant MM cells.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix : Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WOB : Float
        Fitness of an OB.

    Example:
    -----------
    >>> fitness_WOB(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1, 2.5, 2.1],
    ...    [1, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0, -0.2, 1.2]]))
    -0.020900000000000002
    """
    # Extract the necessary matrix values
    e = matrix[1, 0]
    f = matrix[1, 1]
    g = matrix[1, 2]
    h = matrix[1, 3]

    # Calculate the fitness value
    WOB = (e*xOC*cOC + f*xOB*cOB + g*xMMd*cMMd + h* xMMr*cMMr)*(N - 1)/N - cOB
    return WOB

def fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix):
    """
    Function that calculates the fitness of a drug-senstive MM cell in a population.

    Parameters:
    -----------
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMr: Float
        Frequency of the resistant MM cells.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix : Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WMMd : Float
        Fitness of a drug-sensitive MM cell.

    Example:
    -----------
    >>> fitness_WMMd(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]))
    0.05730000000000007
    """
    # Extract the necessary matrix values
    i = matrix[2, 0]
    j = matrix[2, 1]
    k = matrix[2, 2]
    l = matrix[2, 3]

    # Calculate the fitness value
    WMMd = (i*xOC*cOC + j*xOB*cOB + k*xMMd*cMMd + l* xMMr*cMMr)*(N - 1)/N - cMMd
    return WMMd

def fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix, drug_effect = 0):
    """
    Function that calculates the fitness of a drug-senstive MM cell in a population.

    Parameters:
    -----------
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMr: Float
        Frequency of the resistant MM cells.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    drug_effect: Float
        The effect of a drug on the drug-sensitive MM cells

    Returns:
    --------
    WMMd : Float
        Fitness of a drug-sensitive MM cell.

    Example:
    -----------
    >>> fitness_WMMd(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]), 0)
    0.05730000000000007
    """
    # Extract the necessary matrix values
    i = matrix[2, 0]
    j = matrix[2, 1]
    k = matrix[2, 2]
    l = matrix[2, 3]

    # Calculate the fitness value
    WMMd = (i*xOC*cOC + j*xOB*cOB + k*xMMd*cMMd + l* xMMr*cMMr - drug_effect * cMMd)*(N - 1)/N - cMMd
    return WMMd

def fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix):
    """
    Function that calculates the fitness of a resistant MM cell in a population.

    Parameters:
    -----------
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMr: Float
        Frequency of the resistant MM cells.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix : Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WMMr : Float
        Fitness of a resistant MM cell.

    Example:
    -----------
    >>> fitness_WMMr(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]))
    -0.23539999999999994
    """
    # Extract the necessary matrix values
    m = matrix[3, 0]
    n = matrix[3, 1]
    o = matrix[3, 2]
    p = matrix[3, 3]

    # Calculate the fitness value
    WMMr = (m*xOC*cOC + n*xOB*cOB + o*xMMd*cMMd + p* xMMr*cMMr)*(N - 1)/N - cMMr
    return WMMr

def model_dynamics(y, t, N, cOC, cOB, cMMd, cMMr, matrix):
    """Determines the frequenty dynamics in a population over time.

    Parameters:
    -----------
    y : List
        List with the values of xOC, xOB, xMMd and xMMr.
    t : List
        List with all the time points.
    N : Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    [xOC_change, xOB_change, xMMd_change, xMMr_change]: List
        List containing the changes in frequencies of xOC, xOB, xMMd and xMMr.
    """
    xOC, xOB, xMMd, xMMr= y

    # Determine the fitness values
    WOC = fitness_WOC(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
    WOB = fitness_WOB(xOC, xOB, xMMd, xMMr,  N, cOC, cOB, cMMd, cMMr, matrix)
    WMMd = fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
    WMMr = fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the average fitness
    W_average = xOC * WOC + xOB * WOB + xMMd * WMMd + xMMr * WMMr

    # Determine the new frequencies based on replicator dynamics
    xOC_change = xOC * (WOC - W_average)
    xOB_change = xOB * (WOB - W_average)
    xMMd_change = xMMd * (WMMd - W_average)
    xMMr_change = xMMr * (WMMr - W_average)

    return [xOC_change, xOB_change, xMMd_change, xMMr_change]

def model_dynamics(y, t, N, cOC, cOB, cMMd, cMMr, matrix, drug_effect = 0):
    """Determines the frequenty dynamics in a population over time.

    Parameters:
    -----------
    y : List
        List with the values of xOC, xOB, xMMd and xMMr.
    t : List
        List with all the time points.
    N : Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    drug_effect: Float
        The effect of a drug on the drug-sensitive MM cells

    Returns:
    --------
    [xOC_change, xOB_change, xMMd_change, xMMr_change]: List
        List containing the changes in frequencies of xOC, xOB, xMMd and xMMr.
    """
    xOC, xOB, xMMd, xMMr = y

    # Determine the fitness values
    WOC = fitness_WOC(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
    WOB = fitness_WOB(xOC, xOB, xMMd, xMMr,  N, cOC, cOB, cMMd, cMMr, matrix)
    WMMd = fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix, drug_effect)
    WMMr = fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the average fitness
    W_average = xOC * WOC + xOB * WOB + xMMd * WMMd + xMMr * WMMr

    # Determine the new frequencies based on replicator dynamics
    xOC_change = xOC * (WOC - W_average)
    xOB_change = xOB * (WOB - W_average)
    xMMd_change = xMMd * (WMMd - W_average)
    xMMr_change = xMMr * (WMMr - W_average)

    return [xOC_change, xOB_change, xMMd_change, xMMr_change]

def model_dynamics_change_N(y, t, N, cOC, cOB, cMMd, cMMr, matrix):
    """Determines the frequenty dynamics in a population over time.

    Parameters:
    -----------
    y : List
        List with the values of xOC, xOB, xMMd and xMMr.
    t : List
        List with all the time points.
    N : Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix : Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    [xOC_change, xOB_change, xMMd_change, xMMr_change]: List
        List containing the changes in frequencies of xOC, xOB, xMMd and xMMr.
    """
    xOC, xOB, xMMd, xMMr= y

    for i in range(2, 100, 2):
        if t > i:
            N += 5

    # Determine the fitness values
    WOC = fitness_WOC(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
    WOB = fitness_WOB(xOC, xOB, xMMd, xMMr,  N, cOC, cOB, cMMd, cMMr, matrix)
    WMMd = fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
    WMMr = fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the average fitness
    W_average = xOC * WOC + xOB * WOB + xMMd * WMMd + xMMr * WMMr

    # Determine the new frequencies based on replicator dynamics
    xOC_change = xOC * (WOC - W_average)
    xOB_change = xOB * (WOB - W_average)
    xMMd_change = xMMd * (WMMd - W_average)
    xMMr_change = xMMr * (WMMr - W_average)

    return [xOC_change, xOB_change, xMMd_change, xMMr_change]

def freq_to_fitness_values_change_N(dataframe_frequencies, N, cOC, cOB, cMMd, cMMr, matrix):
    """Function that determine the fitness values of the OCs, OBs, MMr and MMr
    based on there frequencies on every time point. It also calculates the
    average fitness.

    Parameters:
    -----------
    dataframe_frequencies: Dataframe
        Dataframe with the frequencies of the OBs, OCs MMd and MMr on every
        timepoint

    Returns:
    --------
    dataframe_fitness: Dataframe
        A dataframe with the fitness values of the OBs, OCs, MMd and MMr and
        the avreage fitness on every time point.
    """

    # Make lists
    WOC_list = []
    WOB_list = []
    WMMd_list = []
    WMMr_list = []
    W_average_list = []
    generation_list = []
    N_start = N

    # Iterate over each row
    for index, row in dataframe_frequencies.iterrows():
        N = N_start
        for i in range(2, 100, 2):
            if index > i:
                N += 5


        # Extract values of xOC, xOB, and xMM for the current row
        xOC = row['xOC']
        xOB = row['xOB']
        xMMd = row['xMMd']
        xMMr = row['xMMr']

        # Determine the fitness values
        WOC = fitness_WOC(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
        WOB = fitness_WOB(xOC, xOB, xMMd, xMMr,  N, cOC, cOB, cMMd, cMMr, matrix)
        WMMd = fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
        WMMr = fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)

        # Determine the average fitness
        W_average = xOC * WOC + xOB * WOB + xMMd * WMMd + xMMr * WMMr

        # Append the calculated values to the respective lists
        WOC_list.append(WOC)
        WOB_list.append(WOB)
        WMMd_list.append(WMMd)
        WMMr_list.append(WMMr)
        W_average_list.append(W_average)
        generation_list.append(index)

    # Create a new DataFrame with the calculated values
    dataframe_fitness = pd.DataFrame({'Generation': generation_list,
    'WOC': WOC_list, 'WOB': WOB_list, 'WMMd': WMMd_list, 'WMMr': WMMr_list, 'W_average': W_average_list})

    return(dataframe_fitness)

def freq_to_fitness_values(dataframe_frequencies, N, cOC, cOB, cMMd, cMMr, matrix):
    """Function that determine the fitness values of the OCs, OBs, MMr and MMr
    based on there frequencies on every time point. It also calculates the
    average fitness.

    Parameters:
    -----------
    dataframe_frequencies: Dataframe
        Dataframe with the frequencies of the OBs, OCs MMd and MMr on every
        timepoint

    Returns:
    --------
    dataframe_fitness: Dataframe
        A dataframe with the fitness values of the OBs, OCs, MMd and MMr and
        the avreage fitness on every time point.
    """

    # Make lists
    WOC_list = []
    WOB_list = []
    WMMd_list = []
    WMMr_list = []
    W_average_list = []
    generation_list = []

    # Iterate over each row
    for index, row in dataframe_frequencies.iterrows():
        # Extract values of xOC, xOB, and xMM for the current row
        xOC = row['xOC']
        xOB = row['xOB']
        xMMd = row['xMMd']
        xMMr = row['xMMr']

        # Determine the fitness values
        WOC = fitness_WOC(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
        WOB = fitness_WOB(xOC, xOB, xMMd, xMMr,  N, cOC, cOB, cMMd, cMMr, matrix)
        WMMd = fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
        WMMr = fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)

        # Determine the average fitness
        W_average = xOC * WOC + xOB * WOB + xMMd * WMMd + xMMr * WMMr

        # Append the calculated values to the respective lists
        WOC_list.append(WOC)
        WOB_list.append(WOB)
        WMMd_list.append(WMMd)
        WMMr_list.append(WMMr)
        W_average_list.append(W_average)
        generation_list.append(index)

    # Create a new DataFrame with the calculated values
    dataframe_fitness = pd.DataFrame({'Generation': generation_list,
    'WOC': WOC_list, 'WOB': WOB_list, 'WMMd': WMMd_list, 'WMMr': WMMr_list, 'W_average': W_average_list})

    return(dataframe_fitness)


"""
Example payoff matrix:
M = np.array([
       Goc Gob Gmmd Gmmr
    OC  [a,  b,  c,  d],
    OB  [e,  f,  g,  h],
    MMd [i,  j,  k,  l],
    MMr [m,  n,  o,  p]])
"""

# Do doc tests
import doctest
doctest.testmod()

def figure_freq_fitness_dynamics_change_N():
    """Function that makes figure of the OC, OB, MMd and MMr frequency and fitness
     values over the time wherby N changes over the time"""

    # Set start values
    N = 50
    cMMr = 1.4
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.3
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.2

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 1.8, 1.5],
        [1.0, 0.1, -0.3, -0.3],
        [1.8, 0, 0.2, -0.2],
        [1.5, 0, -0.2, 0.2]])

    t = np.linspace(0, 30, 30)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Set new start parameter value
    xOC = 0.3
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.2

    t = np.linspace(0, 30, 30)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics_change_N, y0, t, args=parameters)
    df_figure_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_fitness_first_line= freq_to_fitness_values(df_figure_first_line, N, cOC, cOB, cMMd, cMMr, matrix)
    df_fitness_second_line =freq_to_fitness_values_change_N(df_figure_second_line, N, cOC, cOB, cMMd, cMMr, matrix)

    # Create a figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # Plot first subplot
    df_figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0, 0])
    axs[0, 0].set_xlabel('Generations')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title('Frequenc dynamics for a scenario where cOB<cOC<cMMd<cMMr')
    axs[0, 0].legend()

    # Plot the second subplot
    df_fitness_first_line.plot(x='Generation', y=['WOC', 'WOB', 'WMMd', 'WMMr', 'W_average'],
                                                                        ax=axs[0, 1])
    axs[0, 1].set_title('Fitness for a scenario where cOB<cOC<cMMd<cMMr')
    axs[0, 1].set_xlabel('Generations')
    axs[0, 1].set_ylabel('Fitness')
    axs[0, 1].legend(['Fitness OC', 'Fitness OB', 'Fitness MMd', 'Fitness MMr', 'Average fitness'])

    # Plot the third subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Frequenc dynamics for a scenario where cOB<cOC<cMMd<cMMr, N changes')
    axs[1, 0].legend()

    # Plot the fourth subplot
    df_fitness_second_line.plot(x='Generation', y=['WOC', 'WOB', 'WMMd', 'WMMr', 'W_average'],
                                                                        ax=axs[1, 1])
    axs[1, 1].set_title('Fitness for a scenario where cOB<cOC<cMMd<cMMr, N changes')
    axs[1, 1].set_xlabel('Generations')
    axs[1, 1].set_ylabel('Fitness')
    axs[1, 1].legend(['Fitness OC', 'Fitness OB', 'Fitness MMd', 'Fitness MMr', 'Average fitness'])
    plt.tight_layout()
    plt.show()

# figure_freq_fitness_dynamics_change_N()


def figure_freq_fitness_dynamics():
    """Function that makes figure of the OC, OB, MMd and MMr frequency and fitness
     values over the time"""

    # Set start values
    N = 50
    cMMr = 1.4
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.3
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.2

    # Payoff matrix
    matrix = np.array([
        [0, 1, 1.5, 1.3],
        [1, 0, -0.3, -0.3],
        [1.5, 0, 0, -0.2],
        [1.3, 0, -0.2, 0]])

    t = np.linspace(0, 65, 65)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Set new start parameter values
    xOC = 0.1
    xOB = 0.2
    xMMd = 0.4
    xMMr = 0.3

    t = np.linspace(0, 65, 65)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_fitness_first_line= freq_to_fitness_values(df_figure_first_line, N, cOC, cOB, cMMd, cMMr, matrix)
    df_fitness_second_line =freq_to_fitness_values(df_figure_second_line, N, cOC, cOB, cMMd, cMMr, matrix)

    # Create a figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # Plot first subplot
    df_figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0, 0])
    axs[0, 0].set_xlabel('Generations')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title('Frequenc dynamics for a scenario where cOB<cOC<cMMd<cMMr')
    axs[0, 0].legend()

    # Plot the second subplot
    df_fitness_first_line.plot(x='Generation', y=['WOC', 'WOB', 'WMMd', 'WMMr', 'W_average'],
                                                                        ax=axs[0, 1])
    axs[0, 1].set_title('Fitness for a scenario where cOB<cOC<cMMd<cMMr')
    axs[0, 1].set_xlabel('Generations')
    axs[0, 1].set_ylabel('Fitness')
    axs[0, 1].legend(['Fitness OC', 'Fitness OB', 'Fitness MMd', 'Fitness MMr', 'Average fitness'])

    # Plot the third subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Frequenc dynamics for a scenario where cOB<cOC<cMMd<cMMr')
    axs[1, 0].legend()

    # Plot the fourth subplot
    df_fitness_second_line.plot(x='Generation', y=['WOC', 'WOB', 'WMMd', 'WMMr', 'W_average'],
                                                                        ax=axs[1, 1])
    axs[1, 1].set_title('Fitness for a scenario where cOB<cOC<cMMd<cMMr')
    axs[1, 1].set_xlabel('Generations')
    axs[1, 1].set_ylabel('Fitness')
    axs[1, 1].legend(['Fitness OC', 'Fitness OB', 'Fitness MMd', 'Fitness MMr', 'Average fitness'])
    plt.tight_layout()
    plt.show()

# figure_freq_fitness_dynamics()

def figure_freq_dynamics():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time.
    Two senearios with differents start frequencies """
    # Set start values
    N = 10
    cMMr = 1.4
    cMMd = 1.4
    cOB = 1.2
    cOC = 1
    xOC = 0.3
    xOB = 0.3
    xMMd = 0.3
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0, 1, 1.5, 1.3],
        [1, 0, -0.3, -0.3],
        [1.5, 0, 0, 0],
        [1.3, 0, 0, 0]])

    t = np.linspace(0, 40, 40)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Set new start parameter value
    xOC = 0.05
    xOB = 0.3
    xMMd = 0.4
    xMMr = 0.25

    t = np.linspace(0, 40, 40)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})


    # Create a figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where cOB<cOC<cMM<cMMd ')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where cOB<cOC<cMM<cMMd ')
    axs[1].legend()
    plt.tight_layout()
    plt.show()
#
# figure_freq_dynamics()

def figure_interaction_dynamics():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time.
    Two senearios with differents interaction values"""
    # Set start values
    N = 10
    cMMr = 0.5
    cMMd = 0.4
    cOB = 0.2
    cOC = 0.3
    xOC = 0.1
    xOB = 0.3
    xMMd = 0.4
    xMMr = 0.2

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.4, 1.8, 1.5],
        [1.1, 0.1, -0.3, -0.3],
        [1.8, 0, 0.2, -0.2],
        [1.5, 0, -0.2, 0.2]])

    t = np.linspace(0, 80, 80)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Set new start parameter value
    matrix = np.array([
        [0.1, 1.4, 2.0, 1.5],
        [0.6, 0.1, -0.3, -0.3],
        [2.0, 0, 0.3, -0.2],
        [1.5, 0, -0.2, 0.2]])


    t = np.linspace(0, 80, 80)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})


    # Create a figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where cOB<cOC<cMMr<cMMd ')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where cOB<cOC<cMMr<cMMd')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# figure_interaction_dynamics()

def figure_freq_dynamics_2():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time.
    Diffent start values result in another equilibrium"""
    # Set start values
    N = 10
    cMMr = 1.1
    cMMd = 1.0
    cOB = 0.6
    cOC = 0.8
    xOC = 0.4
    xOB = 0.4
    xMMd = 0.2
    xMMr = 0.0

    # Payoff matrix
    matrix = np.array([
        [0, 1, 2.0, 1.0],
        [1, 0, -0.3, -0.3],
        [2.0, 0, 0, 0],
        [1.0, 0, 0, 0]])

    t = np.linspace(0, 80,100)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Set new start parameter value
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.3
    xMMr = 0.0

    t = np.linspace(0, 80,100)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})


    # Create a figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where cOB<cOC<cMM<cMMd ')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where cOB<cOC<cMM<cMMd ')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# figure_freq_dynamics_2()

def figure_freq_dynamics_drugs():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time.
    It shows the effect of a drug that lowers the fitness of of the MMd cells """
    # Set start values
    N = 50
    cMMr = 1.1
    cMMd = 1.0
    cOB = 0.6
    cOC = 0.8
    xOC = 0.3
    xOB = 0.4
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 2.1],
        [1.0, 0.1, -0.3, -0.3],
        [2.2, 0, 0.2, -0.2],
        [2.1, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    drug_effect_1 = 1.5

    # Initial conditions
    t = np.linspace(40, 140, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, drug_effect_1)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_figure_first_line = pd.concat([df_1, df_2])

    # Set new start parameter value
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 2.1],
        [1.0, 0.1, -0.3, -0.3],
        [2.2, 0, 0.2, -0.2],
        [2.1, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    # Payoff matrix
    drug_effect_2 = 0.6

    # Initial conditions
    t = np.linspace(40, 140, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, drug_effect_2)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_figure_second_line = pd.concat([df_1, df_2])

    # Create a figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Dynamics when drugs are added (strenght {drug_effect_1})')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Dynamics when drugs are added (strength {drug_effect_2})')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# figure_freq_dynamics_drugs()

def figure_freq_dynamics_drug_strength():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time.
    It shows the effect of a drug that lowers the fitness of of the MMd cells """
    # Set start values
    N = 50
    cMMr = 1.2
    cMMd = 1.0
    cOB = 0.7
    cOC = 0.8
    xOC = 0.3
    xOB = 0.4
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.1, 2.0],
        [1.0, 0.1, -0.3, -0.2],
        [2.2, 0, 0.2, -0.5],
        [2.0, 0, -0.5, 0.2]])


    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    drug_effect_1 = 1.5

    # Initial conditions
    t = np.linspace(40, 90, 60)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, drug_effect_1)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})


    xOC = df_2['xOC'].iloc[-1]
    xOB = df_2['xOB'].iloc[-1]
    xMMd = df_2['xMMd'].iloc[-1]
    xMMr = df_2['xMMr'].iloc[-1]

    drug_effect_1 = 0.0

    # Initial conditions
    t = np.linspace(90, 140, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, drug_effect_1)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_3 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_figure_first_line = pd.concat([df_1, df_2, df_3])


    # Set new start parameter value
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.1, 2.0],
        [1.0, 0.1, -0.3, -0.2],
        [2.2, 0, 0.2, -0.5],
        [2.0, 0, -0.5, 0.2]])


    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    drug_effect_2 = 0.9

    # Initial conditions
    t = np.linspace(40, 90, 50)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, drug_effect_2)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_2['xOC'].iloc[-1]
    xOB = df_2['xOB'].iloc[-1]
    xMMd = df_2['xMMd'].iloc[-1]
    xMMr = df_2['xMMr'].iloc[-1]

    drug_effect_2 = 0.0

    # Initial conditions
    t = np.linspace(90, 140, 50)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, drug_effect_2)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_3 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_figure_second_line = pd.concat([df_1, df_2, df_3])

    # Set new start parameter value
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.1, 2.0],
        [1.0, 0.1, -0.3, -0.2],
        [2.2, 0, 0.2, -0.5],
        [2.0, 0, -0.5, 0.2]])


    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    drug_effect_3 = 0.6

    # Initial conditions
    t = np.linspace(40, 140, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, drug_effect_3)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_figure_third_line = pd.concat([df_1, df_2])

    # Create a figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot first line data in the first subplot
    df_figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Dynamics when drugs are added at G 40\n and stoped at G 90 (strenght 1.5)')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Dynamics when drugs are added at G 40 \n and stoped at G 90 (strength 0.9)')
    axs[1].legend(loc='upper left')
    plt.tight_layout()

    # Plot second line data in the third subplot
    df_figure_third_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[2])
    axs[2].set_xlabel('Generations')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title(f'Dynamics when drugs are added \n at G 40 (strength {drug_effect_3})')
    axs[2].legend()
    plt.tight_layout()
    plt.show()

figure_freq_dynamics_drug_strength()

def figure_freq_dynamics_GF_inhibition():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time.
    After some time the GF for the MM cells from the OC gets inhibited."""
    # Set start values
    N = 50
    cMMr = 1.1
    cMMd = 1.0
    cOB = 0.6
    cOC = 0.8
    xOC = 0.3
    xOB = 0.4
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 1.9],
        [1.0, 0.1, -0.3, -0.3],
        [2.2, 0, 0.2, -0.2],
        [1.9, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    matrix = np.array([
        [0.1, 1.6, 2.2, 1.9],
        [1.0, 0.1, -0.3, -0.3],
        [0.9, 0, 0.2, -0.2],
        [1.9, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(40, 140, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_figure_first_line = pd.concat([df_1, df_2])

    # Set new start parameter value
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 1.9],
        [1.0, 0.1, -0.3, -0.3],
        [2.2, 0, 0.2, -0.2],
        [1.9, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 1.9],
        [1.0, 0.1, -0.3, -0.3],
        [0.8, 0, 0.2, -0.2],
        [1.9, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(40, 140, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_figure_second_line = pd.concat([df_1, df_2])

    # Create a figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where GF from OC for MM get inhibited (works)')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where GF from OC for MM get inhibited (works not)')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# figure_freq_dynamics_GF_inhibition()

def figure_freq_dynamics_GF_inhibition_short():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time.
    After some time the GF for the MM cells from the OC gets inhibited."""
    # Set start values
    N = 50
    cMMr = 1.1
    cMMd = 1.0
    cOB = 0.6
    cOC = 0.8
    xOC = 0.3
    xOB = 0.4
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 1.9],
        [1.0, 0.1, -0.3, -0.3],
        [2.2, 0, 0.2, -0.2],
        [1.9, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 1.9],
        [1.0, 0.1, -0.3, -0.3],
        [0.8, 0, 0.2, -0.2],
        [1.9, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(40, 75, 35)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_2['xOC'].iloc[-1]
    xOB = df_2['xOB'].iloc[-1]
    xMMd = df_2['xMMd'].iloc[-1]
    xMMr = df_2['xMMr'].iloc[-1]

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 1.9],
        [1.0, 0.1, -0.3, -0.3],
        [2.2, 0, 0.2, -0.2],
        [1.9, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(75, 140, 65)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_3 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_figure_first_line = pd.concat([df_1, df_2, df_3])

    # Set new start parameter value
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 1.9],
        [1.0, 0.1, -0.3, -0.3],
        [2.2, 0, 0.2, -0.2],
        [1.9, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 1.9],
        [1.0, 0.1, -0.3, -0.3],
        [0.8, 0, 0.2, -0.2],
        [1.9, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(40, 75, 35)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_2['xOC'].iloc[-1]
    xOB = df_2['xOB'].iloc[-1]
    xMMd = df_2['xMMd'].iloc[-1]
    xMMr = df_2['xMMr'].iloc[-1]

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 1.9],
        [1.0, 0.1, -0.3, -0.3],
        [2.2, 0, 0.2, -0.2],
        [1.9, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(75, 120, 45)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_3 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_3['xOC'].iloc[-1]
    xOB = df_3['xOB'].iloc[-1]
    xMMd = df_3['xMMd'].iloc[-1]
    xMMr = df_3['xMMr'].iloc[-1]

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.6, 2.2, 1.9],
        [1.0, 0.1, -0.3, -0.3],
        [0.8, 0, 0.2, -0.2],
        [1.9, 0, -0.2, 0.2]])

    # Initial conditions
    t = np.linspace(120, 180, 60)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_4 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})


    df_figure_second_line = pd.concat([df_1, df_2, df_3, df_4])


    # Create a figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where GF for MM from OC get inhibited')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where GF for MM from OC get inhibited')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# figure_freq_dynamics_GF_inhibition_short()



def figure_freq_dynamics_decrease_MMd():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time.
    Where with transplantation big part of the MM cells get removed  """
    # Set start values
    N = 50
    cMMr = 1.2
    cMMd = 1.0
    cOB = 0.6
    cOC = 0.8
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.4
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0, 1, 1.85, 1.0],
        [1, 0, -0.3, -0.3],
        [1.85, 0, 0, 0],
        [1.0, 0, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = 0.65
    xOB = 0.25
    xMMd = 0.1
    xMMr = 0.0

    # Initial conditions
    t = np.linspace(40, 120, 120)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_figure_first_line = pd.concat([df_1, df_2])

    # Set new start parameter value
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.3
    xMMr = 0.2

    # Initial conditions
    t = np.linspace(0, 55, 65)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = 0.63
    xOB = 0.22
    xMMd = 0.15
    xMMr = 0.0

    # Initial conditions
    t = np.linspace(55, 140, 140)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_figure_second_line = pd.concat([df_1, df_2])


    # Create a figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where MMd gets reduced (works)')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where MMd gets reduced (works not)')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# figure_freq_dynamics_decrease_MMd()

def figure_freq_dynamics_resistance():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time.
    After some time the resiatnce cells will grow"""
    # Set start values
    N = 50
    cMMr = 0.6
    cMMd = 0.5
    cOB = 0.3
    cOC = 0.4
    xOC = 0.3
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.2

    # Payoff matrix
    matrix = np.array([
        [0.1, 1.4, 2.0, 1.8],
        [1.0, 0.1, -0.3, -0.3],
        [2.0, 0, 0.2, -0.3],
        [1.8, 0, -0.3, 0.2]])


    # Initial conditions
    t = np.linspace(0, 30, 30)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = 0.60
    xOB = 0.3
    xMMd = 0.08
    xMMr = 0.02

    # Initial conditions
    t = np.linspace(30,140, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = 0.60
    xOB = 0.3
    xMMd = 0.08
    xMMr = 0.02

    # Initial conditions
    t = np.linspace(140, 240, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_3 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_figure_first_line = pd.concat([df_1, df_2, df_3])

    # Plot first line data in the first subplot
    df_figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
         label=['Frequency OC', 'Frequency OB', 'Frequency MMd', 'Frequency MMr'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where MMr develops')
    plt.legend()
    plt.show()

# figure_freq_dynamics_resistance()
