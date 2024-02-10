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
    xOC: float
        Frequency of OCs.
    xOB: float
        Frequency of OBs.
    xMMr: float
        Frequency of the resistant MM cells.
    xMMd: float
        Frequency of the drug-sensitive MM cells.
    N: int
        Number of individuals within the interaction range.
    cOC: float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: float
        Cost parameter resistant MM cells.
    cMMd: float
        Cost parameter drug-sensitive MM cells.
    matrix : numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WOC : float
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
    xOC: float
        Frequency of OCs.
    xOB: float
        Frequency of OBs.
    xMMr: float
        Frequency of the resistant MM cells.
    xMMd: float
        Frequency of the drug-sensitive MM cells.
    N: int
        Number of individuals within the interaction range.
    cOC: float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: float
        Cost parameter resistant MM cells.
    cMMd: float
        Cost parameter drug-sensitive MM cells.
    matrix : numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WOB : float
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
    xOC: float
        Frequency of OCs.
    xOB: float
        Frequency of OBs.
    xMMr: float
        Frequency of the resistant MM cells.
    xMMd: float
        Frequency of the drug-sensitive MM cells.
    N: int
        Number of individuals within the interaction range.
    cOC: float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: float
        Cost parameter resistant MM cells.
    cMMd: float
        Cost parameter drug-sensitive MM cells.
    matrix : numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WMMd : float
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

def fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix):
    """
    Function that calculates the fitness of a resistant MM cell in a population.

    Parameters:
    -----------
    xOC: float
        Frequency of OCs.
    xOB: float
        Frequency of OBs.
    xMMr: float
        Frequency of the resistant MM cells.
    xMMd: float
        Frequency of the drug-sensitive MM cells.
    N: int
        Number of individuals within the interaction range.
    cOC: float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: float
        Cost parameter resistant MM cells.
    cMMd: float
        Cost parameter drug-sensitive MM cells.
    matrix : numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WMMr : float
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
    cOC: float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: float
        Cost parameter resistant MM cells.
    cMMd: float
        Cost parameter drug-sensitive MM cells.
    matrix : numpy.ndarray
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

# Do doc tests
import doctest
doctest.testmod()


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

    t = np.linspace(0, 40, 40)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Set new start parameter value
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

    t = np.linspace(0, 40, 40)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_fitness_first_line= freq_to_fitness_values(df_figure_second_line, N, cOC, cOB, cMMd, cMMr, matrix)
    df_fitness_second_line =freq_to_fitness_values(df_figure_first_line, N, cOC, cOB, cMMd, cMMr, matrix)

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

figure_freq_fitness_dynamics()


def figure_freq_dynamics():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time"""
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
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

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
    axs[0].set_title('Dynamics for a scenario where cOB<cOC<cMMr/cMMd ')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where cOB<cOC<cMMr/cMMd')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# figure_freq_dynamics()

""""Different start conditions different results"""
def figure_freq_dynamics_2():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time"""
    # Set start values
    N = 10
    cMMr = 1.0
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
    axs[0].set_title('Dynamics for a scenario where cOB<cOC<cMMr/cMMd')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where cOB<cOC<cMMr/cMMd ')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# figure_freq_dynamics_2()

def figure_freq_dynamics_decrease_MMd():
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time"""
    # Set start values
    N = 50
    cMMr = 1.0
    cMMd = 1.0
    cOB = 0.6
    cOC = 0.8
    xOC = 0.3
    xOB = 0.3
    xMMd = 0.4
    xMMr = 0.0

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
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.3
    xMMr = 0.0

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
    """Function that makes figure of the xOC, xOB, xMMd and xMMr values over the time"""
    # Set start values
    N = 50
    cMMr = 0.8
    cMMd = 1.0
    cOB = 0.6
    cOC = 0.8
    xOC = 0.3
    xOB = 0.3
    xMMd = 0.4
    xMMr = 0.0

    # Payoff matrix
    matrix = np.array([
        [0, 1, 1.85, 1.85],
        [1, 0, -0.3, -0.4],
        [1.85, 0, 0, 0],
        [1.85, 0, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 30, 30)
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
    t = np.linspace(40,140, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = 0.3
    xOB = 0.65
    xMMd = 0.0
    xMMr = 0.05

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

figure_freq_dynamics_resistance()

"""
t = np.linspace(0, 50, 50)

# Initial conditions
y0 = [xOC, xOB, xMM]
parameters = (N, c1, c2, c3, matrix)

# determine the ODE solutions
y = odeint(model_dynamics, y0, t, args=parameters)
df_figure_2_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
'xOB': y[:, 1], 'xMM': y[:, 2]})

Example payoff matrix:
M = np.array([
       Goc Gob Gmm
    OC [a, b, c],
    OB [d, e, f],
    MM [g, h, i]])
"""
def figure_8A():
    """Function that makes figure 8A in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 10
    cOC = 1
    cOB = 1.2
    cMMr = 0
    cMMd = 1.4
    xOC = 0.8
    xOB = 0.2
    xMMr = 0.0
    xMMd = 0

    # Payoff matrix
    matrix = np.array([
        [0, 1, 1.1, 0],
        [1, 0, -0.3, 0],
        [1.1, 0, 0, 0],
        [0, 0, 0, 0]])

    # t = np.linspace(0, 10,10)
    #
    # # Initial conditions
    # y0 = [xOC, xOB, xMMd, cMMr]
    # parameters = (N, cOC, cOB, cMMd, cMMr, matrix)
    #
    # # determine the ODE solutions
    # y = odeint(model_dynamics, y0, t, args=parameters)
    # df_figure_8A_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    # 'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Set new start parameter value
    xOC = 0.2
    xOB = 0.5
    xMMd = 0.3
    xMMr = 0.0

    t = np.linspace(0, 10,10)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, cMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_figure_8A_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # # Save the data as csv file
    # save_data(df_figure_8A_first_line, 'df_figure_8A_first_line.csv',
    #                                 r'..\data\reproduced_data_Sartakhti_linear')
    # save_data(df_figure_8A_second_line, 'df_figure_8A_second_line.csv',
    #                                 r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    # df_figure_8A_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMMd', 'xMMr'],
    #     label = ['Frequency OC', 'Frequency OB', 'Frequency MMd', 'Frequency MMr'])
    # plt.xlabel('Generations')
    # plt.ylabel('Frequency')
    # plt.title('Dynamics for a scenario where c2<c1<c3 (figure 8A)')
    # plt.legend()
    # plt.show()

    df_figure_8A_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMMd', 'xMMr'],
        label = ['Frequency OC', 'Frequency OB', 'Frequency MMd', 'Frequency MMr'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (figure 8A)')
    plt.legend()
    plt.show()

    # # Make a ternary plot
    # fig1 = px.line_ternary(df_figure_8A_first_line, a='xOC', b='xOB', c='xMM')
    # fig2 = px.line_ternary(df_figure_8A_second_line, a='xOC', b='xOB', c='xMM')
    # fig1.update_layout(
    #     ternary=dict(
    #         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
    #         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
    #         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    #
    # # Add both lines to one ternary plot
    # for trace in fig2.data:
    #     fig1.add_trace(trace)
    # fig1.data[0].update(line=dict(color='red'))
    # fig1.data[1].update(line=dict(color='blue'))
    # fig1.update_layout(title_text= 'Dynamics for a scenario where c2<c1<c3 (figure 8A)')
    # # save_ternary(fig1, 'Ternary_plot_figure_8A',
    # #                     r'..\visualisation\reproduced_results_Sartakhti_linear')
    # fig1.show()
# Set start values
N = 10
cOC = 1
cOB = 1.2
cMMr = 1.4
cMMd = 0
xOC = 0.2
xOB = 0.5
xMMd = 0.3
xMMr = 0.0

# Payoff matrix
matrix = np.array([
    [0, 1, 1.1, 0],
    [1, 0, -0.3, 0],
    [1.1, 0, 0, 0],
    [0, 0, 0, 0]])

WOC = fitness_WOC(xOC, xOB, xMMr, xMMd, N, cOC, cOB, cMMr, cMMd, matrix)
WOB = fitness_WOB(xOC, xOB, xMMr, xMMd, N, cOC, cOB, cMMr, cMMd, matrix)
WMMd = fitness_WMMd(xOC, xOB, xMMr, xMMd, N, cOC, cOB, cMMr, cMMd, matrix)
WMMr = fitness_WMMr(xOC, xOB, xMMr, xMMd, N, cOC, cOB, cMMr, cMMd, matrix)

# Determine the average fitness
W_average = xOC * WOC + xOB * WOB + xMMr * WMMr + xMMd * WMMd

# Determine the new frequencies based on replicator dynamics
xOC_change = xOC * (WOC - W_average)
xOB_change = xOB * (WOB - W_average)
xMMd_change = xMMd * (WMMd - W_average)
xMMr_change = xMMr * (WMMr - W_average)

print(xOC_change, xOB_change, xMMd_change, xMMr_change)


figure_8A()
#
# """Figure 10A"""
# def figure_10A():
#     """Function that makes figure 10A in the paper of Sartakhti et al., 2016."""
#     # Set start values
#     N = 10
#     c3 = 1
#     c2 = 1.2
#     c1 = 1.2
#     xOC = 0.1
#     xOB = 0.1
#     xMM = 0.8
#
#     # Payoff matrix
#     matrix = np.array([
#         [0.3, 1, 2],
#         [1, 1.4, 1.5],
#         [-0.5, -0.9, 1.2]])
#
#     generations = 50
#
#     # Make a dataframe
#     column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
#     df_figure_10A_first_line = pd.DataFrame(columns=column_names)
#
#     # Determine the frequentie value a number of times
#     for generation in range(generations):
#         WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#         WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#         WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#
#         # Determine the average fittness
#         W_average = xOC * WOC + xOB * WOB + xMM * WMM
#
#         # Determine the new frequencies based of replicator dynamics
#         xOC_change = xOC * (WOC - W_average) # (15)
#         xOB_change = xOB * (WOB - W_average) # (16)
#         xMM_change = xMM * (WMM - W_average) # (17)
#
#         # Add row to dataframe (first add row and the update because then also the
#         # beginning values get added to the dataframe at generation =0)
#         new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                         'xMM': xMM, 'W_average': W_average}])
#         df_figure_10A_first_line = pd.concat([df_figure_10A_first_line, new_row],
#                                                                 ignore_index=True)
#
#         # Update the xOC, xOB, xMM values
#         xOC = max(0, xOC + xOC_change)
#         xOB = max(0, xOB + xOB_change)
#         xMM = max(0, xMM + xMM_change)
#
#     # Set new start values
#     xOC = 0.2
#     xOB = 0.2
#     xMM = 0.6
#
#     # Make a datadrame
#     column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
#     df_figure_10A_second_line = pd.DataFrame(columns=column_names)
#
#     # Determine the frequentie value a number of times
#     for generation in range(generations):
#         WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#         WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#         WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#
#         # Determine the average fittness
#         W_average = xOC * WOC + xOB * WOB + xMM * WMM
#
#         # Determine the new frequencies based of replicator dynamics
#         xOC_change = xOC * (WOC - W_average) # (15)
#         xOB_change = xOB * (WOB - W_average) # (16)
#         xMM_change = xMM * (WMM - W_average) # (17)
#
#         # Add row to dataframe (first add row and the update because then also the
#         # beginning values get added to the dataframe at generation =0)
#         new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                                 'xMM': xMM, 'W_average': W_average}])
#         df_figure_10A_second_line = pd.concat([df_figure_10A_second_line, new_row],
#                                                                 ignore_index=True)
#
#         # Update the xOC, xOB, xMM values
#         xOC = max(0, xOC + xOC_change)
#         xOB = max(0, xOB + xOB_change)
#         xMM = max(0, xMM + xMM_change)
#
#     # Make a plot
#     df_figure_10A_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
#     plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
#     plt.xlabel('Generations')
#     plt.ylabel('Fitness/ Frequency')
#     plt.title('Dynamics for a scenario where c3<c1<c2 (figure 10A)')
#     plt.legend()
#     plt.show()
#
#     df_figure_10A_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
#     plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
#     plt.xlabel('Generations')
#     plt.ylabel('Fitness/ Frequency')
#     plt.title('Dynamics for a scenario where c3<c1<c2 (figure 10A)')
#     plt.legend()
#     plt.show()
#
#     # Make a ternary plot
#     fig1 = px.line_ternary(df_figure_10A_first_line, a='xOC', b='xOB', c='xMM')
#     fig2 = px.line_ternary(df_figure_10A_second_line, a='xOC', b='xOB', c='xMM')
#
#     fig1.update_layout(
#         ternary=dict(
#             aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#             baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#             caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
#     # Add both lines to one ternary plot
#     for trace in fig2.data:
#         fig1.add_trace(trace)
#     fig1.data[0].update(line=dict(color='red'))
#     fig1.data[1].update(line=dict(color='blue'))
#     fig1.update_layout(title_text= 'Dynamics for a scenario where c3<c1<c2 (figure 10A)')
#     save_ternary(fig1, 'Ternary_plot_figure_10A',
#                         r'..\visualisation\reproduced_results_Sartakhti_linear')
#     fig1.show()
#
# # figure_10A()
#
# def figure_effect_growth_factor_inhibition():
#     """Function that simulates the effect of growth inhibitor resistance"""
#     # Set start values
#     N = 10
#     c3 = 1.4
#     c2 = 1
#     c1 = 1.2
#     xOC = 0.3
#     xOB = 0.25
#     xMM = 0.45
#
#     # Payoff matrix
#     matrix = np.array([
#         [0.1, 1, 1.5],
#         [1.2, 0.1, -0.3],
#         [1.5, 0.9, 0.1]])
#
#     generations = 100
#
#     # Make a dataframe
#     column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
#     df_figure_11_first_line = pd.DataFrame(columns=column_names)
#
#     # Determine the frequentie value a number of times
#     for generation in range(generations):
#         generations = generation *1
#
#         #Reduce effect of OC GF on MM cells
#         if generation == 30:
#             matrix = np.array([
#                 [0.3, 1, 1.5],
#                 [1.2, 0.1, -0.3],
#                 [1.2, 0.9, 0.2]])
#
#         WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#         WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#         WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#
#         # Determine the average fittness
#         W_average = xOC * WOC + xOB * WOB + xMM * WMM
#
#         # Determine the new frequencies based of replicator dynamics
#         xOC_change = xOC * (WOC - W_average) # (15)
#         xOB_change = xOB * (WOB - W_average) # (16)
#         xMM_change = xMM * (WMM - W_average) # (17)
#
#
#         # Add row to dataframe (first add row and the update because then also the
#         # beginning values get added to the dataframe at generation =0)
#         new_row = pd.DataFrame([{'Generation':generations, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#         df_figure_11_first_line = pd.concat([df_figure_11_first_line, new_row],
#                                                                 ignore_index=True)
#
#         # Update the xOC, xOB, xMM values
#         xOC = max(0, xOC + xOC_change)
#         xOB = max(0, xOB + xOB_change)
#         xMM = max(0, xMM + xMM_change)
#
#     # Make a plot
#     df_figure_11_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'])
#     plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
#     plt.xlabel('Generations')
#     plt.ylabel('Frequency')
#     plt.title('Effect GF inhibition')
#     plt.legend()
#     plt.show()
#
#     # Make a ternary plot
#     fig1 = px.line_ternary(df_figure_11_first_line, a='xOC', b='xOB', c='xMM')
#
#     fig1.update_layout(
#         ternary=dict(
#             aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#             baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#             caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
#     # Add both lines to one ternary plot
#     fig1.update_layout(title_text= 'Effect GF inhibition')
#
#     fig1.show()
#
# # figure_effect_growth_factor_inhibition()
#
# def figure_effect_resistentie():
#     """Function that simulates the effect of growth inhibitor resistance"""
#     # Set start values
#     N = 10
#     c3 = 1.4
#     c2 = 1
#     c1 = 1.2
#     xOC = 0.3
#     xOB = 0.25
#     xMM = 0.45
#
#     # Payoff matrix
#     matrix = np.array([
#         [0.3, 1, 1.5],
#         [1.2, 0.1, -0.3],
#         [1.5, 0.9, 0.2]])
#
#     generations = 110
#
#     # Make a dataframe
#     column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
#     df_figure_11_first_line = pd.DataFrame(columns=column_names)
#
#     # Determine the frequentie value a number of times
#     for generation in range(generations):
#         generations = generation *1
#
#         # Reduce effect of OC GF on MM cells
#         if generation == 15:
#             matrix = np.array([
#                 [0.3, 1, 1.5],
#                 [1.2, 0.1, -0.3],
#                 [1.2, 0.9, 0.2]])
#
#         # Development resistance
#         if generation == 75:
#             matrix = np.array([
#                 [0.3, 1, 1.5],
#                 [1.2, 0.1, -0.3],
#                 [1.5, 0.9, 0.2]])
#
#         WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#         WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#         WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)
#
#         # Determine the average fittness
#         W_average = xOC * WOC + xOB * WOB + xMM * WMM
#
#
#         # Determine the new frequencies based of replicator dynamics
#         xOC_change = xOC * (WOC - W_average) # (15)
#         xOB_change = xOB * (WOB - W_average) # (16)
#         xMM_change = xMM * (WMM - W_average) # (17)
#
#
#         # Add row to dataframe (first add row and the update because then also the
#         # beginning values get added to the dataframe at generation =0)
#         new_row = pd.DataFrame([{'Generation':generations, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#         df_figure_11_first_line = pd.concat([df_figure_11_first_line, new_row],
#                                                                 ignore_index=True)
#
#         # Update the xOC, xOB, xMM values
#         xOC = max(0, xOC + xOC_change)
#         xOB = max(0, xOB + xOB_change)
#         xMM = max(0, xMM + xMM_change)
#
#     # Make a plot
#     df_figure_11_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
#     plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
#     plt.xlabel('Generations')
#     plt.ylabel('Frequency')
#     plt.title('Effect reducing MM cells (figure 11)')
#     plt.legend()
#     plt.show()
#
#     # Make a ternary plot
#     fig1 = px.line_ternary(df_figure_11_first_line, a='xOC', b='xOB', c='xMM')
#
#     fig1.update_layout(
#         ternary=dict(
#             aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#             baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#             caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
#     # Add both lines to one ternary plot
#     fig1.update_layout(title_text= 'Effect reducing MM cells (figure 11)')
#
#     fig1.show()
#
# figure_effect_resistentie()
