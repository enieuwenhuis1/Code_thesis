"""
Author:       Eva Nieuwenhuis
University:   UvA
Student id':  13717405
Description:  Code with the model that simulates the dynamics in the multiple myeloma
              (MM) microenvironment with four cell types: drug-sensitive MM cells
              (MMd), resistant MM cells (MMr), osteoblasts (OBs) and osteoclasts
              (OCs). The model is a public goods game in the framework of evolutionary
              game theory with collective interactions and linear benefits.
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
import csv
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import doctest

doctest.testmod()

def fitness_WOC(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix):
    """
    Function that calculates the fitness of an osteoclast in a population.

    Parameters:
    -----------
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    xMMr: Float
        Frequency of the resistant MM cells.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    cMMr: Float
        Cost parameter resistant MM cells.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WOC: Float
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
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    xMMr: Float
        Frequency of the resistant MM cells.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    cMMr: Float
        Cost parameter resistant MM cells.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WOB: Float
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

def fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix,
                                                            WMMd_inhibitor = 0):
    """
    Function that calculates the fitness of a drug-senstive MM cell in a population.

    Parameters:
    -----------
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    xMMr: Float
        Frequency of the resistant MM cells.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    cMMr: Float
        Cost parameter resistant MM cells.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor: Float
        The effect of a drug on the drug-sensitive MM cells.

    Returns:
    --------
    WMMd: Float
        Fitness of a MMd.

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
    WMMd = (i*xOC*cOC + j*xOB*cOB + k*xMMd*cMMd + l* xMMr*cMMr - WMMd_inhibitor \
                                                        * cMMd)*(N - 1)/N - cMMd
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
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    cMMr: Float
        Cost parameter resistant MM cells.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WMMr: Float
        Fitness of a MMr.

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

def model_dynamics(y, t, N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor = 0):
    """Function that determines the frequenty dynamics in a population over time.

    Parameters:
    -----------
    y: List
        List with the values of xOC, xOB, xMMd and xMMr.
    t: Numpy.ndarray
        Array with all the time points.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    cMMr: Float
        Cost parameter resistant MM cells.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor: Float
        The effect of a drug on the drug-sensitive MM cells.

    Returns:
    --------
    [xOC_change, xOB_change, xMMd_change, xMMr_change]: List
        List containing the changes in frequencies of xOC, xOB, xMMd and xMMr.

    Example:
    -----------
    >>> model_dynamics([0.4, 0.2, 0.3, 0.1], 1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]))
    [0.030275999999999983, -0.010762000000000006, 0.0073170000000000145, -0.026830999999999994]
    """
    xOC, xOB, xMMd, xMMr = y

    # Determine the fitness values
    WOC = fitness_WOC(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
    WOB = fitness_WOB(xOC, xOB, xMMd, xMMr,  N, cOC, cOB, cMMd, cMMr, matrix)
    WMMd = fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix,
                                                                WMMd_inhibitor)
    WMMr = fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the average fitness
    W_average = xOC * WOC + xOB * WOB + xMMd * WMMd + xMMr * WMMr

    # Determine the new frequencies based on replicator dynamics
    xOC_change = xOC * (WOC - W_average)
    xOB_change = xOB * (WOB - W_average)
    xMMd_change = xMMd * (WMMd - W_average)
    xMMr_change = xMMr * (WMMr - W_average)

    # Make floats of the arrays
    xOC_change = float(xOC_change)
    xOB_change = float(xOB_change)
    xMMd_change = float(xMMd_change)
    xMMr_change = float(xMMr_change)

    return [xOC_change, xOB_change, xMMd_change, xMMr_change]

def model_dynamics_change_N(y, t, N, cOC, cOB, cMMd, cMMr, matrix):
    """Function that determines the frequency dynamics in a population over time
    when N changes.

    Parameters:
    -----------
    y: List
        List with the values of xOC, xOB, xMMd and xMMr.
    t: Numpy.ndarray
        Array with all the time points.
    N: Int
        Number of cells in the difussion range.
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

    Returns:
    --------
    [xOC_change, xOB_change, xMMd_change, xMMr_change]: List
        List containing the changes in frequencies of xOC, xOB, xMMd and xMMr.
    """
    xOC, xOB, xMMd, xMMr= y

    # Change N every 2 generations
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

def freq_to_fitness_values_change_N(dataframe_frequencies, N, cOC, cOB, cMMd,
                                                                    cMMr, matrix):
    """Function that determines the fitness values of the OCs, OBs, MMd and MMr
    based on their frequencies on every time point. It also calculates the average
    fitness. The number of cells in the popultaion changes.

    Parameters:
    -----------
    dataframe_frequencies: Dataframe
        Dataframe with the frequencies of the OBs, OCs MMd and MMr on every
        timepoint.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    cMMr: Float
        Cost parameter resistant MM cells.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    dataframe_fitness: Dataframe
        A dataframe with the fitness values of the OBs, OCs, MMd and MMr and
        the average fitness on every time point.
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

        # Change N every 2 generations
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

        # Add the calculated fitness values to the respective lists
        WOC_list.append(WOC)
        WOB_list.append(WOB)
        WMMd_list.append(WMMd)
        WMMr_list.append(WMMr)
        W_average_list.append(W_average)
        generation_list.append(index)

    # Create a dataframe with the calculated fitness values
    dataframe_fitness = pd.DataFrame({'Generation': generation_list,
                            'WOC': WOC_list, 'WOB': WOB_list, 'WMMd': WMMd_list,
                             'WMMr': WMMr_list, 'W_average': W_average_list})

    return(dataframe_fitness)

def freq_to_fitness_values(dataframe_frequencies, N, cOC, cOB, cMMd, cMMr, matrix,
                                                            WMMd_inhibitor = 0):
    """Function that determines the fitness values of the OCs, OBs, MMd and MMr
    based on their frequencies on every time point. It also calculates the average
    fitness.

    Parameters:
    -----------
    dataframe_frequencies: Dataframe
        Dataframe with the frequencies of the OBs, OCs MMd and MMr on every
        timepoint.
    N: Int
        Number of cells in the difussion range.
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
    WMMd_inhibitor: Float
        The effect of a drug on the drug-sensitive MM cells

    Returns:
    --------
    dataframe_fitness: Dataframe
        A dataframe with the fitness values of the OBs, OCs, MMd and MMr and
        the average fitness on every time point.
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
        WMMd = fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix,
                                                                WMMd_inhibitor)
        WMMr = fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)

        # Determine the average fitness
        W_average = xOC * WOC + xOB * WOB + xMMd * WMMd + xMMr * WMMr

        # Add the calculated fitness values to the respective lists
        WOC_list.append(WOC)
        WOB_list.append(WOB)
        WMMd_list.append(WMMd)
        WMMr_list.append(WMMr)
        W_average_list.append(W_average)
        generation_list.append(index)

    # Create a datafrane with the calculated fitness values
    dataframe_fitness = pd.DataFrame({'Generation': generation_list,
                            'WOC': WOC_list, 'WOB': WOB_list, 'WMMd': WMMd_list,
                                'WMMr': WMMr_list, 'W_average': W_average_list})

    return(dataframe_fitness)

def save_dataframe(data_frame, file_name, folder_path):
    """ Function that saves a dataframe as csv file.

    Parameters:
    -----------
    data_frame: DataFrame
        The dataframe containing the collected data.
    file_name: String
        The name of the csv file.
    folder_path: String
        Path to the folder where the data will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    data_frame.to_csv(file_path, index=False)

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

"""
Example payoff matrix:
M = np.array([
       Goc Gob Gmmd Gmmr
    OC  [a,  b,  c,  d],
    OB  [e,  f,  g,  h],
    MMd [i,  j,  k,  l],
    MMr [m,  n,  o,  p]])
"""

# Do a doc tests
import doctest
doctest.testmod()


"""Determine the best b_OC_MMd value
-----------------------------------------------------------------------------"""
def mimimal_tumour_freq_b_OC_MMd(b_OC_MMd, xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd,
                                                    cMMr, matrix, t, b_OC_MMd_array):
    """Function that determines the fraction of the population being MM for a
    specific b_OC_MMd value.

    Parameters:
    -----------
    b_OC_MMd: Float
        Interaction value that gives the effect of the GFs of OCs on MMd.
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    xMMr: Float
        Frequency of the resistant MM cells.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    cMMr: Float
        Cost parameter resistant MM cells.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    t: Numpy.ndarray
        Array with all the time points.
    b_OC_MMd_array: Float
        If True b_OC_MMd is an array and if False b_OC_MMd is a float.

    Returns:
    --------
    last_MM_frequency: Float
        The total MM frequency.
    """
    # Change b_OC_MMd to a float if it is an array
    if b_OC_MMd_array == True:
        b_OC_MMd = b_OC_MMd[0]

    # Change the b_OC_MM value to the specified value
    matrix[2, 0]= b_OC_MMd

    # Set the initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

    # Determine the total MM frequency
    last_MM_frequency = df['total_MM'].iloc[-1]

    return float(last_MM_frequency)

def Figure_optimal_b_OC_MMd():
    """ Function that makes a Figure that shows the total MM fraction for different
    b_OC_MMd values. It also determines the b_OC_MMd value causing the lowest total
    MM fraction"""

    # Set start values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.63, 0.2]])

    t = np.linspace(0, 100, 100)
    b_OC_MMd_start = 0.8

    # Perform the optimization
    result = minimize(mimimal_tumour_freq_b_OC_MMd, b_OC_MMd_start, args = (xOC, xOB,
            xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix, t, True), bounds=[(0, 3)])

    # Retrieve the optimal value
    optimal_b_OC_MMd= result.x
    print("Optimal value for b_OC_MMd:", float(optimal_b_OC_MMd[0]),
                                                ',gives tumour frequency:', result.fun)

    # Make a dictionary
    dict_freq_tumour_GF = {}

    # Loop over all the b_OC_MMd values
    for b_OC_MMd in range(3000):
        b_OC_MMd = b_OC_MMd/1000

        # Determine the total MM frequency
        freq_tumour = mimimal_tumour_freq_b_OC_MMd(b_OC_MMd, xOC, xOB, xMMd, xMMr,
                                        N, cOC, cOB, cMMd, cMMr, matrix, t, False)
        dict_freq_tumour_GF[b_OC_MMd] = freq_tumour

    # Save the data
    save_dictionary(dict_freq_tumour_GF,
                         r'..\data\data_own_model\dict_cell_freq_b_OC_MMd.csv')

    # Make a list of the keys and one of the values
    b_OC_MMd_values = list(dict_freq_tumour_GF.keys())
    MM_frequencies = list(dict_freq_tumour_GF.values())

    # Create the plot
    plt.plot(b_OC_MMd_values, MM_frequencies, linestyle='-')
    plt.xlabel('Total MM frequency')
    plt.ylabel(r'$b_{OC, MMd}$ value')
    plt.title(r'MM frequency for different $b_{OC, MMd}$ values')
    plt.grid(True)
    save_Figure(plt, 'line_plot_cell_freq_change_b_OC_MMd',
                                        r'..\visualisation\results_own_model')
    plt.show()

# Set initial values
N = 50
cMMr = 1.3
cMMd = 1.2
cOB = 0.8
cOC = 1
xOC = 0.4
xOB = 0.3
xMMd = 0.2
xMMr = 0.1

# Payoff matrix
matrix = np.array([
    [0.0, 1.6, 2.2, 1.9],
    [1.0, 0.0, -0.5, -0.5],
    [2.2, 0, 0.2, 0.0],
    [1.9, 0, -0.63, 0.2]])

t = np.linspace(0, 100, 100)
b_OC_MMd_start = 0.8

# Perform the optimization
result = minimize(mimimal_tumour_freq_b_OC_MMd, b_OC_MMd_start, args = (xOC, xOB,
        xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix, t, True), bounds=[(0, 3)])

# Retrieve the optimal value
optimal_b_OC_MMd= result.x
print("Optimal value for b_OC_MMd:", float(optimal_b_OC_MMd[0]),
                                            ',gives tumour frequency:', result.fun)



"""Determine the best drug effect value for high and low cOB and cOC values
--------------------------------------------------------------------------------"""
def mimimal_tumour_freq_dev(WMMd_inhibitor, xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd,
                                            cMMr, matrix, t, WMMd_inhibitor_array):
    """Function that determines the fraction of the population being MM for a
    specific wMMd drug inhibitor value.

    Parameters:
    -----------
    WMMd_inhibitor: Float
        Streght of the drugs that inhibits the cMMd.
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    xMMr: Float
        Frequency of the resistant MM cells.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    cMMr: Float
        Cost parameter resistant MM cells.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    t: Numpy.ndarray
        Array with all the time points.
    WMMd_inhibitor_array: Float
        If True WMMd_inhibitor is an array and if False WMMd_inhibitor is a float.

    Returns:
    --------
    last_MM_frequency: Float
        The total MM frequency.
    """
    # Determine if WMMd_inhibitor is an array
    if WMMd_inhibitor_array == True:
        WMMd_inhibitor = WMMd_inhibitor[0]

    # Set initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

    # Determine the total MM frequency
    last_MM_frequency = df['total_MM'].iloc[-1]

    return float(last_MM_frequency)

def Figure_drug_cost():
    """ Function that shows the effect of different OB and OC cost values for
    different WMMd drug inhibitor values. It also determines the WMMd IH value
    causing the lowest total MM fraction."""

    # Set start values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.9
    cOC = 1.0
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.0, 1.6, 2.0, 1.8],
        [1.0, 0.0, -0.5, -0.5],
        [2.0, 0, 0.2, 0.0],
        [1.8, 0, -1.1, 0.2]])

    t = np.linspace(0, 100, 100)
    dev_start = 0.3

    # Perform the optimization
    result_high = minimize(mimimal_tumour_freq_dev, dev_start, args = (xOC, xOB, xMMd,
                                xMMr, N, cOC, cOB, cMMd, cMMr, matrix, t, True),
                                bounds=[(0, 0.8)], method='Nelder-Mead')

    # Retrieve the optimal value
    optimal_dev_high = result_high.x
    print("Optimal value for drug effect:", float(optimal_dev_high[0]),
                                       ', gives tumour frequency:', result_high.fun)

    # Make a dictionary
    dict_freq_tumour_high_c = {}

    # Loop over the different WMMd_inhibitor values
    for WMMd_inhibitor in range(3000):
        WMMd_inhibitor = WMMd_inhibitor/1000
        freq_tumour = mimimal_tumour_freq_dev(WMMd_inhibitor, xOC, xOB, xMMd, xMMr,
                                        N, cOC, cOB, cMMd, cMMr, matrix, t, False)
        dict_freq_tumour_high_c[WMMd_inhibitor] = freq_tumour

    # Save the data
    save_dictionary(dict_freq_tumour_high_c,
                    r'..\data\data_own_model\dict_cell_freq_tumour_high_c.csv')

    # Make lists of the keys and the values
    keys_high_c = list(dict_freq_tumour_high_c.keys())
    values_high_c = list(dict_freq_tumour_high_c.values())

    # Set new cOC and cOB values and make a dictionary
    cOB = 0.8
    cOC = 0.9
    dict_freq_tumour_low_c = {}

    # Perform the optimization
    result_low = minimize(mimimal_tumour_freq_dev, dev_start, args = (xOC, xOB, xMMd,
                    xMMr, N, cOC, cOB, cMMd, cMMr, matrix, t, True), bounds=[(0, 3)])

    # Retrieve the optimal value
    optimal_dev_low= result_low.x
    print("Optimal value for drug effect:", float(optimal_dev_low[0]),
                                         ', gives tumour frequency:', result_low.fun)

    # Loop over the different WMMd_inhibitor values
    for WMMd_inhibitor in range(3000):
        WMMd_inhibitor = WMMd_inhibitor/1000
        freq_tumour = mimimal_tumour_freq_dev(WMMd_inhibitor, xOC, xOB, xMMd, xMMr,
                                        N, cOC, cOB, cMMd, cMMr, matrix, t, False)
        dict_freq_tumour_low_c[WMMd_inhibitor] = freq_tumour

    # Save the data
    save_dictionary(dict_freq_tumour_low_c,
                    r'..\data\data_own_model\dict_cell_freq_tumour_low_c.csv')

    # Make lists of the keys and the values
    keys_low_c = list(dict_freq_tumour_low_c.keys())
    values_low_c = list(dict_freq_tumour_low_c.values())

    # Create a Figure
    plt.Figure(figsize=(12, 6))

    # Subplot one
    plt.subplot(1, 2, 1)
    plt.plot(keys_high_c, values_high_c, color='purple')
    plt.title("""MM frequency for various MMd inhibitor strengths
     at cOB = 0.9 and cOC = 1.0""")
    plt.xlabel('MMd inhibtor strength')
    plt.ylabel('MM frequency')
    plt.grid(True)

    # Subplot two
    plt.subplot(1, 2, 2)
    plt.plot(keys_low_c, values_low_c, color='blue')
    plt.title("""MM frequency for various MMd inhibitor strengths
    at cOB = 0.8 and cOC = 0.9""")
    plt.xlabel('MMd inhibtor strength')
    plt.ylabel('MM frequency')

    plt.tight_layout()
    plt.grid(True)
    save_Figure(plt, 'line_plot_cell_freq_change_WMMd_high_low_c',
                                     r'..\visualisation\results_own_model')
    plt.show()

# Set start values
N = 50
cMMr = 1.3
cMMd = 1.2
cOB = 0.9
cOC = 1.0
xOC = 0.4
xOB = 0.3
xMMd = 0.2
xMMr = 0.1

# Payoff matrix
matrix = np.array([
    [0.0, 1.6, 2.0, 1.8],
    [1.0, 0.0, -0.5, -0.5],
    [2.0, 0, 0.2, 0.0],
    [1.8, 0, -1.1, 0.2]])

t = np.linspace(0, 100, 100)
dev_start = 0.3

# Perform the optimization
result_high = minimize(mimimal_tumour_freq_dev, dev_start, args = (xOC, xOB, xMMd,
                            xMMr, N, cOC, cOB, cMMd, cMMr, matrix, t, True),
                            bounds=[(0, 0.8)], method='Nelder-Mead')

# Retrieve the optimal value
optimal_dev_high = result_high.x
print("Optimal value for drug effect:", float(optimal_dev_high[0]),
                                   ', gives tumour frequency:', result_high.fun)

# Set new cOB and cOC values
cOB = 0.8
cOC = 0.9

# Perform the optimization
result_low = minimize(mimimal_tumour_freq_dev, dev_start, args = (xOC, xOB, xMMd,
                xMMr, N, cOC, cOB, cMMd, cMMr, matrix, t, True), bounds=[(0, 3)])

# Retrieve the optimal value
optimal_dev_low= result_low.x
print("Optimal value for drug effect:", float(optimal_dev_low[0]),
                                     ', gives tumour frequency:', result_low.fun)

# Make a Figure
# Figure_drug_cost()


"""-------------------Effet AT therapy-------------------------------"""
def switch_dataframe(n_switches, t_steps_drug, t_steps_no_drug, xOC, xOB, xMMd, xMMr,
    N, cOC, cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the xOC, xOB, xMMd and xMMr values over
    time for a given time of drug holiday and administration periods.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: Int
        The number of generations drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMr: Float
        Frequency of the resistant MM cells.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix_no_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when no drugs are administrated.
    matrix_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when drugs are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the drug-sensitive MM cells.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the xOC, xOB, xMMd and xMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 15
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                    'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a numver of switches
    for i in range(n_switches):

        # If x = 0 make sure the MMd is inhibited
        if x == 0:

            # Determine the start frequency values
            xOC = df_total_switch['xOC'].iloc[-1]
            xOB = df_total_switch['xOB'].iloc[-1]
            xMMd = df_total_switch['xMMd'].iloc[-1]
            xMMr = df_total_switch['xMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_drugs

            t = np.linspace(time, time + t_steps_drug, t_steps_drug)
            y0 = [xOC, xOB, xMMd, xMMr]
            parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

            # Add dataframe tot total dataframe
            df_total_switch = pd.concat([df_total_switch, df])
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 1
            time += t_steps_drug

        # If x = 1 make sure the MMd is not inhibited
        else:
            # Determine the start frequency values
            xOC = df_total_switch['xOC'].iloc[-1]
            xOB = df_total_switch['xOB'].iloc[-1]
            xMMd = df_total_switch['xMMd'].iloc[-1]
            xMMr = df_total_switch['xMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_drugs

            t = np.linspace(time, time + t_steps_no_drug , t_steps_no_drug)
            y0 = [xOC, xOB, xMMd, xMMr]
            parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

            # Add dataframe tot total dataframe
            df_total_switch = pd.concat([df_total_switch, df])
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 0
            time += t_steps_no_drug

    return df_total_switch

def pronto_switch_dataframe(n_switches, t_steps_drug, t_steps_no_drug, xOC, xOB,
                            xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix_no_drugs,
                            matrix_drugs, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the xOC, xOB, xMMd and xMMr values over
    time for a given time of drug holiday and administration periods. It starts
    immediately with switching.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: Int
        The number of generations drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMr: Float
        Frequency of the resistant MM cells.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix_no_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when no drugs are administrated.
    matrix_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when drugs are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the drug-sensitive MM cells.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the xOC, xOB, xMMd and xMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    t = np.linspace(0, t_steps_no_drug, t_steps_no_drug*2)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                    'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps_no_drug

    # Perform a numver of switches
    for i in range(n_switches):

        # If x = 0 make sure the MMd is inhibited
        if x == 0:

            # Determine the start frequency values
            xOC = df_total_switch['xOC'].iloc[-1]
            xOB = df_total_switch['xOB'].iloc[-1]
            xMMd = df_total_switch['xMMd'].iloc[-1]
            xMMr = df_total_switch['xMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_drugs

            t = np.linspace(time, time + t_steps_drug, t_steps_drug)
            y0 = [xOC, xOB, xMMd, xMMr]
            parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

            # Add dataframe tot total dataframe
            df_total_switch = pd.concat([df_total_switch, df])
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 1
            time += t_steps_drug

        # If x = 1 make sure the MMd is not inhibited
        else:
            # Determine the start frequency values
            xOC = df_total_switch['xOC'].iloc[-1]
            xOB = df_total_switch['xOB'].iloc[-1]
            xMMd = df_total_switch['xMMd'].iloc[-1]
            xMMr = df_total_switch['xMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_drugs

            t = np.linspace(time, time + t_steps_no_drug , t_steps_no_drug)
            y0 = [xOC, xOB, xMMd, xMMr]
            parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

            # Add dataframe tot total dataframe
            df_total_switch = pd.concat([df_total_switch, df])
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 0
            time += t_steps_no_drug

    return df_total_switch

def Figure_3_senarios_MMd_GF_IH(n_switches, t_steps_drug):
    """ Function that makes a Figure that shows the effect of drug holidays.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the number of time steps drugs are administared and the breaks
        are for the different Figures.
    """
    # Set start parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.3
    xOB = 0.2
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrices
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.7605, 0.2]])

    matrix_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [0.7, 0, 0.2, 0.0],
        [1.9, 0, -0.7605, 0.2]])

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_1 = pronto_switch_dataframe(n_switches, t_steps_drug[0],
                        t_steps_drug[0], xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd,
                        cMMr, matrix_no_drugs, matrix_drugs)
    df_total_switch_2 = pronto_switch_dataframe(n_switches, t_steps_drug[1],
                        t_steps_drug[1], xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd,
                        cMMr, matrix_no_drugs, matrix_drugs)
    df_total_switch_3 = pronto_switch_dataframe(n_switches, t_steps_drug[2],
                        t_steps_drug[2], xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd,
                         cMMr, matrix_no_drugs, matrix_drugs)

    t = np.linspace(0, 20, 20)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

    # Determine the current frequencies
    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    t = np.linspace(20, 100, 80)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

    # Combine the dataframes
    df_total = pd.concat([df_1, df_2])

    # Save the data
    save_dataframe(df_total_switch_1, 'df_cell_freq_G6_MMd_GF_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_total_switch_2, 'df_cell_freq_G8_MMd_GF_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_total_switch_3, 'df_cell_freq_G12_MMd_GF_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_total, 'df_cell_freq_MMd_GF_inhibit_continuously.csv',
                                                     r'..\data\data_own_model')

    # Create a Figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Plot the data without drug holidays in the first plot
    df_total.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[ \
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[0, 0])
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel('MM frequency')
    axs[0, 0].set_title(f'Dynamics when MMd GF inhibitors are administered continuously')
    axs[0, 0].legend(loc = 'upper right')
    axs[0, 0].grid(True)

    # Plot the data with drug holidays in the second plot
    df_total_switch_1.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[0, 1])
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel('MM frequency')
    axs[0, 1].set_title(f'Dynamics when MMd GF inhibitors are administered every 6 generations')
    axs[0, 1].legend(loc = 'upper right')
    axs[0, 1].grid(True)

    # Plot the data with drug holidays in the third plot
    df_total_switch_2.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations')
    axs[1, 0].set_ylabel('MM frequency')
    axs[1, 0].set_title(f'Dynamics when MMd GF inhibitors are administered every 8 generations')
    axs[1, 0].legend(loc = 'upper right')
    axs[1, 0].grid(True)
    plt.grid(True)

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_3.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[1, 1])
    axs[1, 1].set_xlabel('Generations')
    axs[1, 1].set_ylabel('MM frequency')
    axs[1, 1].set_title(f'Dynamics when MMd GF inhibitors are administered every 10 generations')
    axs[1, 1].legend(loc = 'upper right')
    axs[1, 1].grid(True)
    save_Figure(plt, 'line_plot_cell_freq_MMd_GF_inhibit',
                                     r'..\visualisation\results_own_model')
    plt.show()

# Create a Figure that shows the effect of drug holidays
# list_t_steps_drug = [6, 8, 10]
# Figure_3_senarios_MMd_GF_IH(10, list_t_steps_drug)

# Define your matrix
matrix = np.array([
    [0.0, 1.1, 1.4, 1.2],
    [0.6, 0.0, -0.3, -0.3],
    [1.4, 0, 0.1, 0.0],
    [1.2, 0, -0.4, 0.1]])

# Calculate eigenvalues
eigenvalues = np.linalg.eigvals(matrix)

print("Eigenvalues:", eigenvalues)



def Figure_freq_fitness_dynamics():
    """Function that makes Figure of the OC, OB, MMd and MMr frequency and fitness
     values over the time"""

    # Set start values
    # Set initial parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrix
    matrix_no_drugs = np.array([
        [0.0, 1.2, 1.6, 1.5],
        [0.4, 0.0, -0.3, -0.3],
        [1.6, 0.0, 0.1, 0.0],
        [1.5, 0.0, -0.6, 0.1]])

    t = np.linspace(0, 40, 40)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1_MMd_GF_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Determine the current frequencies
    xOC = df_1_MMd_GF_inhibition['xOC'].iloc[-1]
    xOB = df_1_MMd_GF_inhibition['xOB'].iloc[-1]
    xMMd = df_1_MMd_GF_inhibition['xMMd'].iloc[-1]
    xMMr = df_1_MMd_GF_inhibition['xMMr'].iloc[-1]

    # Payoff matrix
    matrix_drugs = np.array([
        [0.0, 1.6, 2.2, 2.1],
        [0.9, 0.0, -0.4, -0.4],
        [1.0, 0, 0.2, 0],
        [2.1, 0, -0.7, 0.2]])

    # Initial conditions
    t = np.linspace(40, 100, 60)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2_MMd_GF_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Combine the dataframes
    df_MMd_GF_inhibition = pd.concat([df_1_MMd_GF_inhibition, df_2_MMd_GF_inhibition])

    # Set new start parameter values
    xOC = 0.15
    xOB = 0.4
    xMMd = 0.25
    xMMr = 0.2
    t = np.linspace(0, 25, 25)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1_WMMd_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Determine the current frequencies
    xOC = df_1_WMMd_inhibition['xOC'].iloc[-1]
    xOB = df_1_WMMd_inhibition['xOB'].iloc[-1]
    xMMd = df_1_WMMd_inhibition['xMMd'].iloc[-1]
    xMMr = df_1_WMMd_inhibition['xMMr'].iloc[-1]

    # Initial conditions
    t = np.linspace(25, 100, 75)
    WMMd_inhibitor =  1.2
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2_WMMd_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Combine the dataframes
    df_WMMd_inhibition = pd.concat([df_1_WMMd_inhibition, df_2_WMMd_inhibition])

    # Make dataframes for the fitness values
    df_fitness_WMMd_inhibition_1 = freq_to_fitness_values(df_1_WMMd_inhibition, N,
                                          cOC, cOB, cMMd, cMMr, matrix_no_drugs)
    df_fitness_WMMd_inhibition_2 = freq_to_fitness_values(df_2_WMMd_inhibition, N,
                         cOC, cOB, cMMd, cMMr, matrix_no_drugs, WMMd_inhibitor)
    df_fitness_WMMd_inhibition = pd.concat([df_fitness_WMMd_inhibition_1,
                                df_fitness_WMMd_inhibition_2], ignore_index=True)

    df_fitness_MMd_GF_inhibition_1 = freq_to_fitness_values(df_1_MMd_GF_inhibition,
                                       N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)
    df_fitness_MMd_GF_inhibition_2 = freq_to_fitness_values(df_2_MMd_GF_inhibition,
                                          N, cOC, cOB, cMMd, cMMr, matrix_drugs)
    df_fitness_MMd_GF_inhibition = pd.concat([df_fitness_MMd_GF_inhibition_1,
                              df_fitness_MMd_GF_inhibition_2], ignore_index=True)

    # Save the data
    save_dataframe(df_WMMd_inhibition, 'df_cell_freq_cWMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_MMd_GF_inhibition, 'df_cell_freq_MMd_GF_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_fitness_WMMd_inhibition, 'df_fitness_WMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_fitness_MMd_GF_inhibition, 'df_fitness_MMd_GF_inhibit.csv',
                                                     r'..\data\data_own_model')

    # Create a Figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    # Plot first subplot
    df_WMMd_inhibition.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                        label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0, 0])
    axs[0, 0].set_xlabel('Generations')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title('Frequency dynamics when a WMMd inhibitor is administerd')
    axs[0, 0].legend(loc = 'upper right')
    axs[0, 0].grid(True)

    # Plot the second subplot
    df_fitness_WMMd_inhibition.plot(y=['WOC', 'WOB', 'WMMd', 'WMMr', 'W_average'],
                                label = ['Fitness OC', 'Fitness OB', 'Fitness MMd',
                                  'Fitness MMr', 'Average fitness'],  ax=axs[0, 1])
    axs[0, 1].set_title('Fitness dynamics when a WMMd inhibitor is administerd')
    axs[0, 1].set_xlabel('Generations')
    axs[0, 1].set_ylabel('Fitness')
    axs[0, 1].legend(['Fitness OC', 'Fitness OB', 'Fitness MMd', 'Fitness MMr',
                                                            'Average fitness'])
    axs[0, 1].legend(loc = 'upper right')
    axs[0, 1].grid(True)

    # Plot the third subplot
    df_MMd_GF_inhibition.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                        label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                                'Frequency MMr'], ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Frequency dynamics when a MMd GF inhibitor is administerd')
    axs[1, 0].legend(loc = 'upper right')
    axs[1, 0].grid(True)

    # Plot the fourth subplot
    df_fitness_MMd_GF_inhibition.plot(y=['WOC', 'WOB', 'WMMd', 'WMMr', 'W_average'],
                              label = ['Fitness OC', 'Fitness OB', 'Fitness MMd',
                                 'Fitness MMr', 'Average fitness'],  ax=axs[1, 1])
    axs[1, 1].set_title('Fitness dynamics when a MMd GF inhibitor is administerd')
    axs[1, 1].set_xlabel('Generations')
    axs[1, 1].set_ylabel('Fitness')
    axs[1, 1].legend(loc = 'upper right')
    axs[1, 1].grid(True)
    plt.tight_layout()
    save_Figure(plt, 'line_plot_cell_freq_fitness_drugs',
                                         r'..\visualisation\results_own_model')
    plt.show()

# Figure_freq_fitness_dynamics()















""" ---------------Optimizing time drug holiday-----------------"""
def mimimal_tumour_freq_t_steps(t_steps_drug, t_steps_no_drug, xOC, xOB, xMMd, xMMr,
        N, cOC, cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the xOC, xOB, xMMd and xMMr values over
    time for a given time of a drug holiday.

    Parameters:
    -----------
    t_steps_drug: Int
        The number of generations drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMr: Float
        Frequency of the resistant MM cells.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix_no_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when no drugs are administrated.
    matrix_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when drugs are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the drug-sensitive MM cells.

    Returns:
    --------
    average_MM_frequencies: float
        The average total MM frequency in the last period.
    """
    # Deteremine the number of switches
    time_step = (t_steps_drug + t_steps_no_drug) / 2
    n_switches = int((110 // time_step) -1)

    # Create a dataframe of the frequencies
    df = switch_dataframe(n_switches, t_steps_drug, t_steps_no_drug, xOC, xOB,
                                 xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                                 matrix_no_drugs, matrix_drugs, WMMd_inhibitor)

    # df.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'])
    # plt.show()
    # Determine the average MM frequency
    last_MM_frequencies = df['total_MM'].tail(int(time_step *2))
    average_MM_frequencies = last_MM_frequencies.sum() / (int(time_step*2))

    return float(average_MM_frequencies)

def Figure_3d_MM_fraction_best_IH_holiday():
    """ Figure that makes three 3D plot that shows the average MM frequency for
    different holiday and administration periods of only MMd GF inhibitor, only
    WMMd inhibitor or both."""

    # Set initial parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrix when no drugs are present
    matrix_no_drugs = np.array([
        [0.0, 1.2, 1.6, 1.5],
        [0.4, 0.0, -0.3, -0.3],
        [1.6, 0.0, 0.1, 0.0],
        [1.5, 0.0, -0.6, 0.1]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_drugs = np.array([
        [0.0, 1.2, 1.6, 1.5],
        [0.4, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.1, 0.0],
        [1.5, 0.0, -0.6, 0.1]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_drugs_comb = np.array([
        [0.0, 1.2, 1.6, 1.5],
        [0.4, 0.0, -0.3, -0.3],
        [1.0, 0.0, 0.1, 0.0],
        [1.5, 0.0, -0.9, 0.1]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.4

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.8

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
    df_holliday_GF_IH = pd.DataFrame(columns=column_names)

    # Loop over all the t_step values for drug administration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug,
                            t_steps_no_drug, xOC, xOB, xMMd, xMMr, N, cOC,
                            cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': int(t_steps_no_drug),
                                            'Generations drug': int(t_steps_drug),
                                             'MM fraction': float(freq_tumour)}])
            df_holliday_GF_IH = pd.concat([df_holliday_GF_IH, new_row_df],
                                                            ignore_index=True)

    # Save the data
    save_dataframe(df_holliday_GF_IH, 'df_cell_freq_best_MMd_GH_IH_holiday.csv',
                                                     r'..\data\data_own_model')

    # Find the drug administration and holiday period causing the lowest MM fraction
    min_index_GF_IH = df_holliday_GF_IH['MM fraction'].idxmin()
    g_no_drug_min_GF_IH = df_holliday_GF_IH.loc[min_index_GF_IH,
                                                           'Generations no drug']
    g_drug_min_GF_IH = df_holliday_GF_IH.loc[min_index_GF_IH, 'Generations drug']
    frac_min_GF_IH = df_holliday_GF_IH.loc[min_index_GF_IH, 'MM fraction']

    print(f"""Lowest MM fraction: {frac_min_GF_IH}-> MMd GF IH holidays are
            {g_no_drug_min_GF_IH} generations and MMd GF IH administrations
            are {g_drug_min_GF_IH} generations""")

    # Avoid errors because of the wrong datatype
    df_holliday_GF_IH['Generations no drug'] = pd.to_numeric(df_holliday_GF_IH[\
                                        'Generations no drug'], errors='coerce')
    df_holliday_GF_IH['Generations drug'] = pd.to_numeric(df_holliday_GF_IH[\
                                        'Generations drug'],errors='coerce')
    df_holliday_GF_IH['MM fraction'] = pd.to_numeric(df_holliday_GF_IH[\
                                        'MM fraction'], errors='coerce')

    # Make a meshgrid for the plot
    X_GF_IH = df_holliday_GF_IH['Generations no drug'].unique()
    Y_GF_IH = df_holliday_GF_IH['Generations drug'].unique()
    X_GF_IH, Y_GF_IH = np.meshgrid(X_GF_IH, Y_GF_IH)
    Z_GF_IH = np.zeros((20, 20))

    # Fill the 2D array with the MM frequency values by looping over each row
    for index, row in df_holliday_GF_IH.iterrows():
        i = int(row.iloc[0]) - 2
        j = int(row.iloc[1]) - 2
        Z_GF_IH[j, i] = row.iloc[2]

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
    df_holliday_W_IH = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug, t_steps_no_drug,
                                xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                                matrix_no_drugs, matrix_no_drugs, WMMd_inhibitor)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': int(t_steps_no_drug),
                                            'Generations drug': int(t_steps_drug),
                                             'MM fraction': float(freq_tumour)}])
            df_holliday_W_IH = pd.concat([df_holliday_W_IH, new_row_df],
                                                                ignore_index=True)

    # Save the data
    save_dataframe(df_holliday_W_IH, 'df_cell_freq_best_WMMd_IH_holiday.csv',
                                                     r'..\data\data_own_model')

    # Find the drug administration and holiday period causing the lowest MM fraction
    min_index_W_IH = df_holliday_W_IH['MM fraction'].idxmin()
    g_no_drug_min_W_IH = df_holliday_W_IH.loc[min_index_W_IH,'Generations no drug']
    g_drug_min_W_IH = df_holliday_W_IH.loc[min_index_W_IH, 'Generations drug']
    frac_min_W_IH = df_holliday_W_IH.loc[min_index_W_IH, 'MM fraction']

    print(f"""Lowest MM fraction: {frac_min_W_IH} -> WMMd IH holidays are
                                    {g_no_drug_min_W_IH} generations and WMMd IH
                            administrations are {g_drug_min_W_IH} generations""")

    # Avoid errors because of the wrong datatype
    df_holliday_W_IH['Generations no drug'] = pd.to_numeric(df_holliday_W_IH[\
                                    'Generations no drug'], errors='coerce')
    df_holliday_W_IH['Generations drug'] = pd.to_numeric(df_holliday_W_IH[\
                                            'Generations drug'], errors='coerce')
    df_holliday_W_IH['MM fraction'] = pd.to_numeric(df_holliday_W_IH[\
                                                'MM fraction'], errors='coerce')

    # Make a meshgrid for the plot
    X_W_IH = df_holliday_W_IH['Generations no drug'].unique()
    Y_W_IH = df_holliday_W_IH['Generations drug'].unique()
    X_W_IH, Y_W_IH = np.meshgrid(X_W_IH, Y_W_IH)
    Z_W_IH = np.zeros((20, 20))

    # Fill the 2D array with the MM frequency values by looping over each row
    for index, row in df_holliday_W_IH.iterrows():
        i = int(row.iloc[0]) -2
        j = int(row.iloc[1]) -2
        Z_W_IH[j, i] = row.iloc[2]

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
    df_holliday_comb = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug, t_steps_no_drug,
                            xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                            matrix_no_drugs, matrix_drugs_comb, WMMd_inhibitor_comb)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': int(t_steps_no_drug),
                                            'Generations drug': int(t_steps_drug),
                                            'MM fraction': float(freq_tumour)}])
            df_holliday_comb = pd.concat([df_holliday_comb, new_row_df],
                                                                ignore_index=True)

    # Save the data
    save_dataframe(df_holliday_comb, 'df_cell_freq_best_MMd_IH_holiday.csv',
                                                     r'..\data\data_own_model')

    # Find the drug administration and holiday period causing the lowest MM fraction
    min_index_comb = df_holliday_comb['MM fraction'].idxmin()
    g_no_drug_min_comb = df_holliday_comb.loc[min_index_comb, 'Generations no drug']
    g_drug_min_comb = df_holliday_comb.loc[min_index_comb, 'Generations drug']
    frac_min_comb = df_holliday_comb.loc[min_index_comb, 'MM fraction']

    print(f"""Lowest MM fraction: {frac_min_comb}-> MMd IH holidays are
                    {g_no_drug_min_comb} generations and MMd IH administrations
                    are {g_drug_min_comb} generations""")

    # Avoid errors because of the wrong datatype
    df_holliday_comb['Generations no drug'] = pd.to_numeric(df_holliday_comb[\
                                        'Generations no drug'], errors='coerce')
    df_holliday_comb['Generations drug'] = pd.to_numeric(df_holliday_comb[\
                                            'Generations drug'], errors='coerce')
    df_holliday_comb['MM fraction'] = pd.to_numeric(df_holliday_comb[\
                                            'MM fraction'], errors='coerce')

    # Make a meshgrid for the plot
    X_comb = df_holliday_comb['Generations no drug'].unique()
    Y_comb = df_holliday_comb['Generations drug'].unique()
    X_comb, Y_comb = np.meshgrid(X_comb, Y_comb)
    Z_comb = np.zeros((20, 20))

    # Fill the 2D array with the MM frequency values by looping over each row
    for index, row in df_holliday_comb.iterrows():
        i = int(row.iloc[0]) - 2
        j = int(row.iloc[1]) - 2
        Z_comb[j, i] = row.iloc[2]

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
            ax.set_zlabel('MM fraction')
            ax.set_title(r'A) $W_{MMd}$ inhibitor', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 28, azim = -127)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('MM fraction')

        elif i == 2:
            surf = ax.plot_surface(X_GF_IH, Y_GF_IH, Z_GF_IH, cmap = 'coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IH')
            ax.set_ylabel('Generations IH')
            ax.set_zlabel('MM fraction')
            ax.set_title('B)  MMd GF inhibitor', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 32, azim = -128)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')

            color_bar.set_label('MM fraction')

        elif i == 3:
            surf = ax.plot_surface(X_comb, Y_comb, Z_comb, cmap = 'coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IHs')
            ax.set_ylabel('Generations IHs')
            ax.set_zlabel('MM fraction')
            ax.set_title('C)  $W_{MMd}$ inhibitor and MMd GF inhibitor', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 40, azim = -142)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('MM fraction')

        else:
            # Hide the emply subplot
            ax.axis('off')

    # Add a color bar
    save_Figure(fig, '3d_plot_MM_frac_best_IH_h_a_periods',
                                            r'..\visualisation\results_own_model')
    plt.show()

# Figure_3d_MM_fraction_best_IH_holiday()

















""" ---------------Optimizing time drug holiday-----------------"""
def mimimal_tumour_freq_t_steps(t_steps_drug, t_steps_no_drug, xOC, xOB, xMMd, xMMr,
        N, cOC, cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the xOC, xOB, xMMd and xMMr values over
    time for a given time of a drug holiday.

    Parameters:
    -----------
    t_steps_drug: Int
        The number of generations drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMr: Float
        Frequency of the resistant MM cells.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix_no_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when no drugs are administrated.
    matrix_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when drugs are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the drug-sensitive MM cells.

    Returns:
    --------
    average_MM_frequencies: float
        The average total MM frequency in the last period.
    """
    # Deteremine the number of switches
    time_step = (t_steps_drug + t_steps_no_drug) / 2
    n_switches = int((110 // time_step) -1)

    # Create a dataframe of the frequencies
    df = switch_dataframe(n_switches, t_steps_drug, t_steps_no_drug, xOC, xOB,
                                 xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                                 matrix_no_drugs, matrix_drugs, WMMd_inhibitor)

    # df.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'])
    # plt.show()
    # Determine the average MM frequency
    last_MM_frequencies = df['total_MM'].tail(int(time_step *2))
    average_MM_frequencies = last_MM_frequencies.sum() / (int(time_step*2))

    return float(average_MM_frequencies)

def Figure_3d_MM_fraction_best_IH_holiday():
    """ Figure that makes three 3D plot that shows the average MM frequency for
    different holiday and administration periods of only MMd GF inhibitor, only
    WMMd inhibitor or both."""

    # Set initial parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrix when no drugs are present
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.77, 0.2]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [0.7, 0, 0.2, 0.0],
        [1.9, 0, -0.77, 0.2]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_drugs_comb = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [1.4, 0, 0.2, 0.0],
        [1.9, 0, -1.1, 0.2]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.7

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 1.0

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
    df_holliday_GF_IH = pd.DataFrame(columns=column_names)

    # Loop over all the t_step values for drug administration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug,
                            t_steps_no_drug, xOC, xOB, xMMd, xMMr, N, cOC,
                            cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': int(t_steps_no_drug),
                                            'Generations drug': int(t_steps_drug),
                                             'MM fraction': float(freq_tumour)}])
            df_holliday_GF_IH = pd.concat([df_holliday_GF_IH, new_row_df],
                                                            ignore_index=True)

    # Save the data
    save_dataframe(df_holliday_GF_IH, 'df_cell_freq_best_MMd_GH_IH_holiday.csv',
                                                     r'..\data\data_own_model')

    # Find the drug administration and holiday period causing the lowest MM fraction
    min_index_GF_IH = df_holliday_GF_IH['MM fraction'].idxmin()
    g_no_drug_min_GF_IH = df_holliday_GF_IH.loc[min_index_GF_IH,
                                                           'Generations no drug']
    g_drug_min_GF_IH = df_holliday_GF_IH.loc[min_index_GF_IH, 'Generations drug']
    frac_min_GF_IH = df_holliday_GF_IH.loc[min_index_GF_IH, 'MM fraction']

    print(f"""Lowest MM fraction: {frac_min_GF_IH}-> MMd GF IH holidays are
            {g_no_drug_min_GF_IH} generations and MMd GF IH administrations
            are {g_drug_min_GF_IH} generations""")

    # Avoid errors because of the wrong datatype
    df_holliday_GF_IH['Generations no drug'] = pd.to_numeric(df_holliday_GF_IH[\
                                        'Generations no drug'], errors='coerce')
    df_holliday_GF_IH['Generations drug'] = pd.to_numeric(df_holliday_GF_IH[\
                                        'Generations drug'],errors='coerce')
    df_holliday_GF_IH['MM fraction'] = pd.to_numeric(df_holliday_GF_IH[\
                                        'MM fraction'], errors='coerce')

    # Make a meshgrid for the plot
    X_GF_IH = df_holliday_GF_IH['Generations no drug'].unique()
    Y_GF_IH = df_holliday_GF_IH['Generations drug'].unique()
    X_GF_IH, Y_GF_IH = np.meshgrid(X_GF_IH, Y_GF_IH)
    Z_GF_IH = np.zeros((20, 20))

    # Fill the 2D array with the MM frequency values by looping over each row
    for index, row in df_holliday_GF_IH.iterrows():
        i = int(row.iloc[0]) - 2
        j = int(row.iloc[1]) - 2
        Z_GF_IH[j, i] = row.iloc[2]

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
    df_holliday_W_IH = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug, t_steps_no_drug,
                                xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                                matrix_no_drugs, matrix_no_drugs, WMMd_inhibitor)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': int(t_steps_no_drug),
                                            'Generations drug': int(t_steps_drug),
                                             'MM fraction': float(freq_tumour)}])
            df_holliday_W_IH = pd.concat([df_holliday_W_IH, new_row_df],
                                                                ignore_index=True)

    # Save the data
    save_dataframe(df_holliday_W_IH, 'df_cell_freq_best_WMMd_IH_holiday.csv',
                                                     r'..\data\data_own_model')

    # Find the drug administration and holiday period causing the lowest MM fraction
    min_index_W_IH = df_holliday_W_IH['MM fraction'].idxmin()
    g_no_drug_min_W_IH = df_holliday_W_IH.loc[min_index_W_IH,'Generations no drug']
    g_drug_min_W_IH = df_holliday_W_IH.loc[min_index_W_IH, 'Generations drug']
    frac_min_W_IH = df_holliday_W_IH.loc[min_index_W_IH, 'MM fraction']

    print(f"""Lowest MM fraction: {frac_min_W_IH} -> WMMd IH holidays are
                                    {g_no_drug_min_W_IH} generations and WMMd IH
                            administrations are {g_drug_min_W_IH} generations""")

    # Avoid errors because of the wrong datatype
    df_holliday_W_IH['Generations no drug'] = pd.to_numeric(df_holliday_W_IH[\
                                    'Generations no drug'], errors='coerce')
    df_holliday_W_IH['Generations drug'] = pd.to_numeric(df_holliday_W_IH[\
                                            'Generations drug'], errors='coerce')
    df_holliday_W_IH['MM fraction'] = pd.to_numeric(df_holliday_W_IH[\
                                                'MM fraction'], errors='coerce')

    # Make a meshgrid for the plot
    X_W_IH = df_holliday_W_IH['Generations no drug'].unique()
    Y_W_IH = df_holliday_W_IH['Generations drug'].unique()
    X_W_IH, Y_W_IH = np.meshgrid(X_W_IH, Y_W_IH)
    Z_W_IH = np.zeros((20, 20))

    # Fill the 2D array with the MM frequency values by looping over each row
    for index, row in df_holliday_W_IH.iterrows():
        i = int(row.iloc[0]) -2
        j = int(row.iloc[1]) -2
        Z_W_IH[j, i] = row.iloc[2]

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
    df_holliday_comb = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug, t_steps_no_drug,
                            xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                            matrix_no_drugs, matrix_drugs_comb, WMMd_inhibitor_comb)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': int(t_steps_no_drug),
                                            'Generations drug': int(t_steps_drug),
                                            'MM fraction': float(freq_tumour)}])
            df_holliday_comb = pd.concat([df_holliday_comb, new_row_df],
                                                                ignore_index=True)

    # Save the data
    save_dataframe(df_holliday_comb, 'df_cell_freq_best_MMd_IH_holiday.csv',
                                                     r'..\data\data_own_model')

    # Find the drug administration and holiday period causing the lowest MM fraction
    min_index_comb = df_holliday_comb['MM fraction'].idxmin()
    g_no_drug_min_comb = df_holliday_comb.loc[min_index_comb, 'Generations no drug']
    g_drug_min_comb = df_holliday_comb.loc[min_index_comb, 'Generations drug']
    frac_min_comb = df_holliday_comb.loc[min_index_comb, 'MM fraction']

    print(f"""Lowest MM fraction: {frac_min_comb}-> MMd IH holidays are
                    {g_no_drug_min_comb} generations and MMd IH administrations
                    are {g_drug_min_comb} generations""")

    # Avoid errors because of the wrong datatype
    df_holliday_comb['Generations no drug'] = pd.to_numeric(df_holliday_comb[\
                                        'Generations no drug'], errors='coerce')
    df_holliday_comb['Generations drug'] = pd.to_numeric(df_holliday_comb[\
                                            'Generations drug'], errors='coerce')
    df_holliday_comb['MM fraction'] = pd.to_numeric(df_holliday_comb[\
                                            'MM fraction'], errors='coerce')

    # Make a meshgrid for the plot
    X_comb = df_holliday_comb['Generations no drug'].unique()
    Y_comb = df_holliday_comb['Generations drug'].unique()
    X_comb, Y_comb = np.meshgrid(X_comb, Y_comb)
    Z_comb = np.zeros((20, 20))

    # Fill the 2D array with the MM frequency values by looping over each row
    for index, row in df_holliday_comb.iterrows():
        i = int(row.iloc[0]) - 2
        j = int(row.iloc[1]) - 2
        Z_comb[j, i] = row.iloc[2]

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
            ax.set_zlabel('MM fraction')
            ax.set_title(r'A) $W_{MMd}$ inhibitor', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 28, azim = -127)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('MM fraction')

        elif i == 2:
            surf = ax.plot_surface(X_GF_IH, Y_GF_IH, Z_GF_IH, cmap = 'coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IH')
            ax.set_ylabel('Generations IH')
            ax.set_zlabel('MM fraction')
            ax.set_title('B)  MMd GF inhibitor', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 32, azim = -128)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')

            color_bar.set_label('MM fraction')

        elif i == 3:
            surf = ax.plot_surface(X_comb, Y_comb, Z_comb, cmap = 'coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IHs')
            ax.set_ylabel('Generations IHs')
            ax.set_zlabel('MM fraction')
            ax.set_title('C)  $W_{MMd}$ inhibitor and MMd GF inhibitor', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 40, azim = -142)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('MM fraction')

        else:
            # Hide the emply subplot
            ax.axis('off')

    # Add a color bar
    save_Figure(fig, '3d_plot_MM_frac_best_IH_h_a_periods',
                                            r'..\visualisation\results_own_model')
    plt.show()

# Figure_3d_MM_fraction_best_IH_holiday()


def Figure_MMd_IH_strength_3D():
    """ 3D plot that shows the average MM frequency for different MMd GF inhibitor
    and WMMd inhibitor strengths."""
    # Set initial parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrix when no drugs are pressent
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.8, 0.2]])

    # Payoff matrix when GF inhibitor drugs are pressent
    matrix_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [1.5, 0, 0.2, 0.0],
        [1.9, 0, -0.8, 0.2]])

    # Administration and holiday periods
    t_steps_drug = 5
    t_steps_no_drug = 5

    # Make a dataframe
    column_names = ['Strength WMMd IH', 'Strength MMd GF IH', 'MM fraction']
    df_holliday = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for strength_WMMd_IH in range(0, 21):

        # Drug inhibitor effect
        WMMd_inhibitor = strength_WMMd_IH / 10
        for strength_MMd_GF_IH in range(0, 21):

            # Change effect of GF of OC on MMd
            matrix_drugs[2, 0] = 2.2 - round((strength_MMd_GF_IH / 10), 1)

            # Change how fast the MMr will be stronger than the MMd
            matrix_drugs[3, 2] = -0.77 - round((WMMd_inhibitor +  round((strength_MMd_GF_IH / 10), 1))/ 20, 3)

            freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug, t_steps_no_drug,
                                    xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                                    matrix_no_drugs, matrix_drugs, WMMd_inhibitor)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Strength WMMd IH': round(strength_WMMd_IH/ 10, 1),
                                    'Strength MMd GF IH': round(strength_MMd_GF_IH/ 10, 1),
                                            'MM fraction': freq_tumour}])

            df_holliday = pd.concat([df_holliday, new_row_df], ignore_index=True)

    # Save the data
    save_dataframe(df_holliday, 'df_cell_freq_best_MMd_IH_strength.csv',
                                                     r'..\data\data_own_model')


    # Find the drug administration and holiday period causing the lowest MM fraction
    min_index = df_holliday['MM fraction'].idxmin()
    strength_WMMd_min = df_holliday.loc[min_index, 'Strength WMMd IH']
    strength_MMd_GF_min = df_holliday.loc[min_index, 'Strength MMd GF IH']
    frac_min = df_holliday.loc[min_index, 'MM fraction']

    print(f"""Lowest MM fraction: {frac_min}-> MMd GF IH strength is
        {strength_MMd_GF_min} and WMMd IH strength is {strength_WMMd_min}""")

    # Avoid errors because of the wrong datatype
    df_holliday['Strength WMMd IH'] = pd.to_numeric(df_holliday[\
                                        'Strength WMMd IH'], errors='coerce')
    df_holliday['Strength MMd GF IH'] = pd.to_numeric(df_holliday['Strength MMd GF IH'],
                                                                errors='coerce')
    df_holliday['MM fraction'] = pd.to_numeric(df_holliday['MM fraction'],
                                                                errors='coerce')

    # Make a meshgrid for the plot
    X = df_holliday['Strength WMMd IH'].unique()
    Y = df_holliday['Strength MMd GF IH'].unique()
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((21, 21))

    # Fill the 2D array with the MM frequency values by looping over each row
    for index, row in df_holliday.iterrows():
        i = int(row.iloc[0]*10)
        j = int(row.iloc[1]*10)
        Z[j, i] = row.iloc[2]

    # Make a 3D Figure
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = '3d')
    surf = ax.plot_surface(X, Y, Z, cmap = 'coolwarm')

    # Add labels
    ax.set_xlabel('Strength WMMd IH')
    ax.set_ylabel('Strength MMd GF IH')
    ax.set_zlabel('MM fraction')
    ax.set_title("""Average MM fraction with varing WMMd inhibitor and MMd
    GF inhibitor strengths""")

    # Turn to the right angle
    ax.view_init(elev = 38, azim = -66)

    # Add a color bar
    color_bar = fig.colorbar(surf, shrink = 0.6, location= 'left')
    color_bar.set_label('MM fraction')

    save_Figure(fig, '3d_plot_MM_frac_best_IH_strength',
                                            r'..\visualisation\results_own_model')
    plt.show()

# Figure_MMd_IH_strength_3D()





def Figure_MMd_IH_holiday_3D():
    """ 3D plot that shows the average MM frequency for different MMd GF inhibitor
    and WMMd inhibitor holiday and administration periods."""
    # Set initial parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrix when no drugs are pressent
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.77, 0.2]])

    # Payoff matrix when GF inhibitor drugs are pressent
    matrix_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [1.5, 0, 0.2, 0.0],
        [1.9, 0, -1.0, 0.2]])

    # Drug inhibitor effect
    WMMd_inhibitor = 0.6

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
    df_holliday = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug, t_steps_no_drug,
                                    xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                                    matrix_no_drugs, matrix_drugs, WMMd_inhibitor)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': int(t_steps_no_drug),
            'Generations drug': int(t_steps_drug), 'MM fraction': float(freq_tumour)}])
            df_holliday = pd.concat([df_holliday, new_row_df], ignore_index=True)

    # Save the data
    save_dataframe(df_holliday, 'df_cell_freq_best_MMd_IH_holiday.csv',
                                                     r'..\data\data_own_model')

    # Find the drug administration and holiday period causing the lowest MM fraction
    min_index = df_holliday['MM fraction'].idxmin()
    g_no_drug_min = df_holliday.loc[min_index, 'Generations no drug']
    g_drug_min = df_holliday.loc[min_index, 'Generations drug']
    frac_min = df_holliday.loc[min_index, 'MM fraction']

    print(f"""Lowest MM fraction: {frac_min}-> MMd IH holidays are {g_no_drug_min}
    generations and MMd IH administrations are {g_drug_min} generations""")

    # Avoid errors because of the wrong datatype
    df_holliday['Generations no drug'] = pd.to_numeric(df_holliday[\
                                        'Generations no drug'], errors='coerce')
    df_holliday['Generations drug'] = pd.to_numeric(df_holliday['Generations drug'],
                                                                errors='coerce')
    df_holliday['MM fraction'] = pd.to_numeric(df_holliday['MM fraction'],
                                                                errors='coerce')

    # Make a meshgrid for the plot
    X = df_holliday['Generations no drug'].unique()
    Y = df_holliday['Generations drug'].unique()
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((20, 20))

    # Fill the 2D array with the MM frequency values by looping over each row
    for index, row in df_holliday.iterrows():
        i = int(row.iloc[0]) - 2
        j = int(row.iloc[1]) - 2
        Z[j, i] = row.iloc[2]

    # Make a 3D Figure
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = '3d')
    surf = ax.plot_surface(X, Y, Z, cmap = 'coolwarm')

    # Add labels
    ax.set_xlabel('Generations no MMd inhibitors')
    ax.set_ylabel('Generations MMd inhibitors')
    ax.set_zlabel('MM fraction')
    ax.set_title("""Average MM fraction with varing WMMd inhibitor and MMd
    GF inhibitor administration and holiday periods""")

    # Turn to the right angle
    ax.view_init(elev = 26, azim = 24)

    # Add a color bar
    color_bar = fig.colorbar(surf, shrink = 0.6)
    color_bar.set_label('The MM fraction')

    save_Figure(fig, '3d_plot_cell_freq_best_MMd_IH_holiday',
                                            r'..\visualisation\results_own_model')
    plt.show()

# Figure_MMd_IH_holiday_3D()



# def mimimal_tumour_freq_t_steps(t_steps_drug, t_steps_no_drug, xOC, xOB, xMMd, xMMr,
#         N, cOC, cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs, WMMd_inhibitor = 0):
#     """ Function that makes a dataframe of the xOC, xOB, xMMd and xMMr values over
#     time for a given time of a drug holiday.
#
#     Parameters:
#     -----------
#     t_steps_drug: Int
#         The number of generations drugs are administared.
#     t_steps_no_drug: Int
#         The number of generations drugs are not administared.
#     xOC: Float
#         Frequency of OCs.
#     xOB: Float
#         Frequency of OBs.
#     xMMr: Float
#         Frequency of the resistant MM cells.
#     xMMd: Float
#         Frequency of the drug-sensitive MM cells.
#     N: Int
#         Number of cells in the difussion range.
#     cOC: Float
#         Cost parameter OCs.
#     cOB: float
#         Cost parameter OBs.
#     cMMr: Float
#         Cost parameter resistant MM cells.
#     cMMd: Float
#         Cost parameter drug-sensitive MM cells.
#     matrix_no_drugs: Numpy.ndarray
#         4x4 matrix containing the interaction factors when no drugs are administrated.
#     matrix_drugs: Numpy.ndarray
#         4x4 matrix containing the interaction factors when drugs are administrated.
#     WMMd_inhibitor: Float
#         The effect of a drug on the drug-sensitive MM cells.
#
#     Returns:
#     --------
#     average_MM_frequencies: float
#         The average total MM frequency in the last period.
#     """
#     # Deteremine the number of switches
#     time_step = (t_steps_drug + t_steps_no_drug) / 2
#     n_switches = int((110 // time_step) -1)
#
#     # Create a dataframe of the frequencies
#     df = switch_dataframe(n_switches, t_steps_drug, t_steps_no_drug, xOC, xOB,
#                                  xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
#                                  matrix_no_drugs, matrix_drugs, WMMd_inhibitor)
#
#     df.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'])
#     plt.show()
#     # Determine the average MM frequency
#     last_MM_frequencies = df['total_MM'].tail(int(time_step *2))
#     average_MM_frequencies = last_MM_frequencies.sum() / (int(time_step*2))
#
#     return float(average_MM_frequencies)

def mimimal_tumour_freq_t_steps(t_steps_drug, t_steps_no_drug, xOC, xOB, xMMd, xMMr,
        N, cOC, cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the xOC, xOB, xMMd and xMMr values over
    time for a given time of a drug holiday.

    Parameters:
    -----------
    t_steps_drug: Int
        The number of generations drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMr: Float
        Frequency of the resistant MM cells.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter resistant MM cells.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    matrix_no_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when no drugs are administrated.
    matrix_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when drugs are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the drug-sensitive MM cells.

    Returns:
    --------
    average_MM_frequencies: float
        The average total MM frequency in the last period.
    """
    # Deteremine the number of switches
    time_step = (t_steps_drug + t_steps_no_drug) / 2
    n_switches = int((110 // time_step) -1)

    # Create a dataframe of the frequencies
    df = switch_dataframe(n_switches, t_steps_drug, t_steps_no_drug, xOC, xOB,
                                 xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                                 matrix_no_drugs, matrix_drugs, WMMd_inhibitor)

    # df.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'])
    # plt.show()
    # Determine the average MM frequency
    last_MM_frequencies = df['total_MM'].tail(int(time_step *2))
    average_MM_frequencies = last_MM_frequencies.sum() / (int(time_step*2))

    return float(average_MM_frequencies)


# Define your matrix
matrix = np.array([
        [0.0, 1.6, 2.1, 1.9],
        [1.1, 0.0, -0.5, -0.5],
        [2.3, 0, 0.2, 0.0],
        [1.9, 0, -0.7, 0.2]])

# Calculate eigenvalues
# eigenvalues = np.linalg.eigvals(matrix)
#
# print("Eigenvalues matrix are :", eigenvalues)

#
# import numpy as np
#
# # Original matrix
# original_matrix = np.array([
#         [0.0, 1.6, 2.2, 1.9],
#         [1.0, 0.0, -0.5, -0.5],
#         [2.2, 0, 0.2, 0.0],
#         [1.9, 0, -0.8, 0.2]])
#
# # Calculate the eigenvalues of the original matrix
# original_eigenvalues = np.linalg.eigvals(original_matrix)
#
# # Choose a factor by which to scale the identity matrix
# scaling_factor = 0.6 # You can adjust this factor as needed
#
# # Create the scaled identity matrix
# identity_matrix = np.identity(original_matrix.shape[0])
# scaled_identity_matrix = scaling_factor * identity_matrix
#
# # Add the scaled identity matrix to the original matrix
# modified_matrix = original_matrix + scaled_identity_matrix
# print(modified_matrix)
# # Calculate the eigenvalues of the modified matrix
# modified_eigenvalues = np.linalg.eigvals(modified_matrix)
#
# print("Original Eigenvalues:", original_eigenvalues)
# print("Modified Eigenvalues:", modified_eigenvalues)

"""
higher
matrix_no_drugs = np.array([
        [0.0, 1.6, 2.0, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.0, 0, 1.0, 0.0],
        [1.9, 0, -0.4, 1.0]])
H3 A10
"""

def Figure_freq_fitness_dynamics():
    """Function that makes Figure of the OC, OB, MMd and MMr frequency and fitness
     values over the time"""

    # Set start values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.15
    xOB = 0.4
    xMMd = 0.25
    xMMr = 0.2

    # Payoff matrix when no drugs are present
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.1, 1.9],
        [0.6, 0.0, -0.5, -0.5],
        [2.1, 0.0, 0.2, 0.0],
        [1.6, 0.0, -0.7, 0.2]])




    t = np.linspace(0, 25, 25)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1_MMd_GF_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Determine the current frequencies
    xOC = df_1_MMd_GF_inhibition['xOC'].iloc[-1]
    xOB = df_1_MMd_GF_inhibition['xOB'].iloc[-1]
    xMMd = df_1_MMd_GF_inhibition['xMMd'].iloc[-1]
    xMMr = df_1_MMd_GF_inhibition['xMMr'].iloc[-1]

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_drugs = np.array([
        [0.0, 1.6, 2.1, 1.9],
        [0.6, 0.0, -0.5, -0.5],
        [0.7, 0.0, 0.2, 0.0],
        [1.6, 0.0, -0.7, 0.2]])

    eigenvalues = np.linalg.eigvals(matrix_drugs)

    print("Eigenvalues matrix :)))) :", eigenvalues)

    # Initial conditions
    t = np.linspace(25, 100, 75)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2_MMd_GF_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Combine the dataframes
    df_MMd_GF_inhibition = pd.concat([df_1_MMd_GF_inhibition, df_2_MMd_GF_inhibition])

    # Set new start parameter values
    xOC = 0.15
    xOB = 0.4
    xMMd = 0.25
    xMMr = 0.2
    t = np.linspace(0, 25, 25)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1_WMMd_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Determine the current frequencies
    xOC = df_1_WMMd_inhibition['xOC'].iloc[-1]
    xOB = df_1_WMMd_inhibition['xOB'].iloc[-1]
    xMMd = df_1_WMMd_inhibition['xMMd'].iloc[-1]
    xMMr = df_1_WMMd_inhibition['xMMr'].iloc[-1]

    # Initial conditions
    t = np.linspace(25, 100, 75)
    WMMd_inhibitor =  1.2
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2_WMMd_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Combine the dataframes
    df_WMMd_inhibition = pd.concat([df_1_WMMd_inhibition, df_2_WMMd_inhibition])

    # Make dataframes for the fitness values
    df_fitness_WMMd_inhibition_1 = freq_to_fitness_values(df_1_WMMd_inhibition, N,
                                          cOC, cOB, cMMd, cMMr, matrix_no_drugs)
    df_fitness_WMMd_inhibition_2 = freq_to_fitness_values(df_2_WMMd_inhibition, N,
                         cOC, cOB, cMMd, cMMr, matrix_no_drugs, WMMd_inhibitor)
    df_fitness_WMMd_inhibition = pd.concat([df_fitness_WMMd_inhibition_1,
                                df_fitness_WMMd_inhibition_2], ignore_index=True)

    df_fitness_MMd_GF_inhibition_1 = freq_to_fitness_values(df_1_MMd_GF_inhibition,
                                       N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)
    df_fitness_MMd_GF_inhibition_2 = freq_to_fitness_values(df_2_MMd_GF_inhibition,
                                          N, cOC, cOB, cMMd, cMMr, matrix_drugs)
    df_fitness_MMd_GF_inhibition = pd.concat([df_fitness_MMd_GF_inhibition_1,
                              df_fitness_MMd_GF_inhibition_2], ignore_index=True)

    # Save the data
    save_dataframe(df_WMMd_inhibition, 'df_cell_freq_cWMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_MMd_GF_inhibition, 'df_cell_freq_MMd_GF_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_fitness_WMMd_inhibition, 'df_fitness_WMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_fitness_MMd_GF_inhibition, 'df_fitness_MMd_GF_inhibit.csv',
                                                     r'..\data\data_own_model')

    # Create a Figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    # Plot first subplot
    df_WMMd_inhibition.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                        label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0, 0])
    axs[0, 0].set_xlabel('Generations')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title('Frequency dynamics when a WMMd inhibitor is administerd')
    axs[0, 0].legend(loc = 'upper right')
    axs[0, 0].grid(True)

    # Plot the second subplot
    df_fitness_WMMd_inhibition.plot(y=['WOC', 'WOB', 'WMMd', 'WMMr', 'W_average'],
                                label = ['Fitness OC', 'Fitness OB', 'Fitness MMd',
                                  'Fitness MMr', 'Average fitness'],  ax=axs[0, 1])
    axs[0, 1].set_title('Fitness dynamics when a WMMd inhibitor is administerd')
    axs[0, 1].set_xlabel('Generations')
    axs[0, 1].set_ylabel('Fitness')
    axs[0, 1].legend(['Fitness OC', 'Fitness OB', 'Fitness MMd', 'Fitness MMr',
                                                            'Average fitness'])
    axs[0, 1].legend(loc = 'upper right')
    axs[0, 1].grid(True)

    # Plot the third subplot
    df_MMd_GF_inhibition.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                        label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                                'Frequency MMr'], ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Frequency dynamics when a MMd GF inhibitor is administerd')
    axs[1, 0].legend(loc = 'upper right')
    axs[1, 0].grid(True)

    # Plot the fourth subplot
    df_fitness_MMd_GF_inhibition.plot(y=['WOC', 'WOB', 'WMMd', 'WMMr', 'W_average'],
                              label = ['Fitness OC', 'Fitness OB', 'Fitness MMd',
                                 'Fitness MMr', 'Average fitness'],  ax=axs[1, 1])
    axs[1, 1].set_title('Fitness dynamics when a MMd GF inhibitor is administerd')
    axs[1, 1].set_xlabel('Generations')
    axs[1, 1].set_ylabel('Fitness')
    axs[1, 1].legend(loc = 'upper right')
    axs[1, 1].grid(True)
    plt.tight_layout()
    save_Figure(plt, 'line_plot_cell_freq_fitness_drugs',
                                         r'..\visualisation\results_own_model')
    plt.show()

Figure_freq_fitness_dynamics()
print(hi)
"""
changing boc,mmd
# Payoff matrix when no drugs are present
matrix_no_drugs = np.array([
    [0.0, 1.6, 2.2, 1.9],
    [1.0, 0.0, -0.5, -0.5],
    [2.2, 0.0, 0.2, 0.0],
    [1.9, 0.0, -0.8, 0.2]])


# Payoff matrix when only GF inhibitor drugs are present
matrix_drugs = np.array([
    [0.0, 1.6, 2.2, 1.9],
    [1.0, 0.0, -0.5, -0.5],
    [1.2,  0, 0.2, 0],
    [1.9, 0, -0.8, 0.2]])
------------------------------------------------------------------------

MM self stimulation
# Payoff matrix when no drugs are present
matrix_no_drugs = np.array([
    [0.0, 1.6, 2.2, 1.9],
    [1.0, 0.0, -0.5, -0.5],
    [2.2, 0.0, 1.1, 0.0],
    [1.9, 0.0, -0.8, 1.1]])


# Payoff matrix when only GF inhibitor drugs are present
matrix_drugs = np.array([
    [0.0, 1.6, 2.2, 1.9],
    [1.0, 0.0, -0.5, -0.5],
    [0.7, 0.0, 1.1, 0.0],
    [1.9, 0, -0.8, 1.1]])
--------------------------------------------------------------------

IH MMr
    # Payoff matrix when no drugs are present
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.2, 0.0, 0.2, 0.0],
        [1.9, 0.0, -0.7, 0.2]])


    # Payoff matrix when only GF inhibitor drugs are present
    matrix_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [0.7,  0, 0.2, 0],
        [1.9, 0, -0.7, 0.2]])


OC stimuleren resistance cells
        # Payoff matrix when no drugs are present
        matrix_no_drugs = np.array([
            [0.0, 1.6, 2.2, 2.0],
            [0.6, 0.0, -0.5, -0.5],
            [2.2, 0.0, 0.2, 0.0],
            [2.1, 0.0, -0.8, 0.2]])


        # Payoff matrix when only GF inhibitor drugs are present
        matrix_drugs = np.array([
            [0.0, 1.6, 2.2, 2.0],
            [0.6, 0.0, -0.5, -0.5],
            [0.7, 0.0, 0.2, 0.0],
            [2.1, 0, -0.8, 0.2]])
"""

def Dataframe_bOCMMd_eigenvalues():
    """ Function that makes a table of the eigenvalues of the interaction matrix,
    MM fraction and the best holiday (H) and administration (A) period for different
    bOC,MMr values."""

    # Make a dataframe
    column_names = ['bOC,MMd', 'Eigenvalue 1', 'Eigenvalue 2', 'Eigenvalue 3',
                        'Eigenvalue 4', 'period H', 'period A', 'MM fraction']
    df_eigenvalues = pd.DataFrame(columns=column_names)

    for i in range(7):
        # Set initial parameter values
        N = 50
        cMMr = 1.3
        cMMd = 1.2
        cOB = 0.8
        cOC = 1
        xOC = 0.2
        xOB = 0.3
        xMMd = 0.2
        xMMr = 0.3

        # Payoff matrix when no drugs are present
        matrix_no_drugs = np.array([
            [0.0, 1.6, 2.2, 1.9],
            [1.0, 0.0, -0.5, -0.5],
            [2.2, 0.0, 0.2, 0.0],
            [1.9, 0.0, -0.8, 0.2]])

        # Payoff matrix when only GF inhibitor drugs are present
        matrix_drugs = np.array([
            [0.0, 1.6, 2.2, 1.9],
            [1.0, 0.0, -0.5, -0.5],
            [0.7,  0, 0.2, 0],
            [1.9, 0, -0.8, 0.2]])

        # Change the interaction value
        matrix_drugs[2, 0] = round(0.6 + (i/10), 1)

        # Determine the eigenvalues
        eigenvalues = np.linalg.eigvals(matrix_drugs)

        # Make a dataframe
        column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
        df_holliday = pd.DataFrame(columns=column_names)

        # Loop over al the t_step values for drug dministration and drug holidays
        for t_steps_no_drug in range(2, 22):

            for t_steps_drug in range(2, 22):
                # print(t_steps_no_drug, t_steps_drug)
                freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug,
                                t_steps_no_drug, xOC, xOB, xMMd, xMMr, N, cOC,
                                cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs)

                # Add results to the dataframe
                new_row_df = pd.DataFrame([{'Generations no drug': \
                    int(t_steps_no_drug), 'Generations drug': int(t_steps_drug),
                    'MM fraction': float(freq_tumour)}])
                df_holliday = pd.concat([df_holliday, new_row_df], ignore_index=True)

        # Find the drug administration and holiday period causing the lowest MM
        # fraction
        min_index = df_holliday['MM fraction'].idxmin()
        g_no_drug_min = df_holliday.loc[min_index, 'Generations no drug']
        g_drug_min = df_holliday.loc[min_index, 'Generations drug']
        frac_min = df_holliday.loc[min_index, 'MM fraction']

        new_row_df = pd.DataFrame([{'bOC,MMd': round(0.6 + (i/10), 1),
                'Eigenvalue 1': eigenvalues[0], 'Eigenvalue 2': eigenvalues[1],
                'Eigenvalue 3': eigenvalues[2], 'Eigenvalue 4': eigenvalues[3],
                'period H': g_no_drug_min, 'period A': g_drug_min,
                'MM fraction': frac_min}])

        df_eigenvalues = pd.concat([df_eigenvalues, new_row_df], ignore_index=True)

    # Save the data
    save_dataframe(df_eigenvalues, 'df_eigenvalues_bOCMMd.csv',
                                             r'..\data\data_own_model_fractions')

Dataframe_bOCMMd_eigenvalues()


def Dataframe_bMMdMMd_bMMrMMr_eigenvalues():
    """ Function that makes a table of the eigenvalues of the interaction matrix,
    MM fraction and the best holiday (H) and administration (A) period for different
    bOC,OC and bOB,OB values."""

    # Make a dataframe
    column_names = ['bMMd,MMd & bMMr,MMr', 'Eigenvalue 1', 'Eigenvalue 2',
        'Eigenvalue 3', 'Eigenvalue 4', 'period H', 'period A', 'MM fraction']
    df_eigenvalues = pd.DataFrame(columns=column_names)

    for i in range(6):
        # Set initial parameter values
        N = 50
        cMMr = 1.3
        cMMd = 1.2
        cOB = 0.8
        cOC = 1
        xOC = 0.2
        xOB = 0.3
        xMMd = 0.2
        xMMr = 0.3

        # Payoff matrix when no drugs are present
        matrix_no_drugs = np.array([
            [0.0, 1.6, 2.2, 1.9],
            [1.0, 0.0, -0.5, -0.5],
            [2.2, 0.0, 1.1, 0.0],
            [1.9, 0.0, -0.8, 1.1]])

        # Payoff matrix when only GF inhibitor drugs are present
        matrix_drugs = np.array([
            [0.0, 1.6, 2.2, 1.9],
            [1.0, 0.0, -0.5, -0.5],
            [0.7, 0.0, 1.1, 0.0],
            [1.9, 0, -0.8, 1.1]])

        # Change the interaction values
        matrix_drugs[2, 2] = 0.1 + round(i/5,1)
        matrix_drugs[3, 3] = 0.1 + round(i/5,1)
        matrix_no_drugs[2, 2] = 0.1 + round(i/5,1)
        matrix_no_drugs[3, 3] = 0.1 + round(i/5,1)

        # Determine the eigenvalues
        eigenvalues = np.linalg.eigvals(matrix_no_drugs)

        # Make a dataframe
        column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
        df_holliday = pd.DataFrame(columns=column_names)

        # Loop over al the t_step values for drug dministration and drug holidays
        for t_steps_no_drug in range(2, 22):

            for t_steps_drug in range(2, 22):
                # print(t_steps_no_drug, t_steps_drug)
                freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug,
                                t_steps_no_drug, xOC, xOB, xMMd, xMMr, N, cOC,
                                cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs)

                # Add results to the dataframe
                new_row_df = pd.DataFrame([{'Generations no drug': \
                    int(t_steps_no_drug), 'Generations drug': int(t_steps_drug),
                    'MM fraction': float(freq_tumour)}])
                df_holliday = pd.concat([df_holliday, new_row_df], ignore_index=True)

        # Find the drug administration and holiday period causing the lowest MM
        # fraction
        min_index = df_holliday['MM fraction'].idxmin()
        g_no_drug_min = df_holliday.loc[min_index, 'Generations no drug']
        g_drug_min = df_holliday.loc[min_index, 'Generations drug']
        frac_min = df_holliday.loc[min_index, 'MM fraction']

        new_row_df = pd.DataFrame([{'bMMd,MMd & bMMr,MMr':  0.1 + round(i/5,1),
                'Eigenvalue 1': eigenvalues[0], 'Eigenvalue 2': eigenvalues[1],
                'Eigenvalue 3': eigenvalues[2], 'Eigenvalue 4': eigenvalues[3],
                'period H': g_no_drug_min, 'period A': g_drug_min,
                'MM fraction': frac_min}])

        df_eigenvalues = pd.concat([df_eigenvalues, new_row_df], ignore_index=True)

    # Save the data
    save_dataframe(df_eigenvalues, 'df_eigenvalues_bMMdMMd_bMMrMMr.csv',
                                            r'..\data\data_own_model_fractions')


Dataframe_bOCOC_bOBOB_eigenvalues()

def Dataframe_bOCMMr_eigenvalues():
    """ Function that makes a table of the eigenvalues of the interaction matrix,
    MM fraction and the best holiday (H) and administration (A) period for different
    bOC,MMr values."""

    # Make a dataframe
    column_names = ['bOC,MMr', 'Eigenvalue 1', 'Eigenvalue 2', 'Eigenvalue 3',
                        'Eigenvalue 4', 'period H', 'period A', 'MM fraction']
    df_eigenvalues = pd.DataFrame(columns=column_names)

    for i in range(5):
        # Set initial parameter values
        N = 50
        cMMr = 1.3
        cMMd = 1.2
        cOB = 0.8
        cOC = 1
        xOC = 0.2
        xOB = 0.3
        xMMd = 0.2
        xMMr = 0.3

        # Payoff matrix when no drugs are present
        matrix_no_drugs = np.array([
            [0.0, 1.6, 2.2, 2.0],
            [0.4, 0.0, -0.5, -0.5],
            [2.2, 0.0, 0.2, 0.0],
            [2.1, 0.0, -0.8, 0.2]])


        # Payoff matrix when only GF inhibitor drugs are present
        matrix_drugs = np.array([
            [0.0, 1.6, 2.2, 2.0],
            [0.4, 0.0, -0.5, -0.5],
            [0.7, 0.0, 0.2, 0.0],
            [2.1, 0, -0.8, 0.2]])

        # Change the interaction values
        matrix_no_drugs[3, 0] = 1.7 + round(i/10, 1)
        matrix_drugs[3, 0] = 1.7 + round(i/10, 1)

        # Determine the eigenvalues
        eigenvalues = np.linalg.eigvals(matrix_no_drugs)

        # Make a dataframe
        column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
        df_holliday = pd.DataFrame(columns=column_names)

        # Loop over al the t_step values for drug dministration and drug holidays
        for t_steps_no_drug in range(2, 22):

            for t_steps_drug in range(2, 22):
                # print(t_steps_no_drug, t_steps_drug)
                freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug,
                                t_steps_no_drug, xOC, xOB, xMMd, xMMr, N, cOC,
                                cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs)

                # Add results to the dataframe
                new_row_df = pd.DataFrame([{'Generations no drug': \
                    int(t_steps_no_drug), 'Generations drug': int(t_steps_drug),
                    'MM fraction': float(freq_tumour)}])
                df_holliday = pd.concat([df_holliday, new_row_df], ignore_index=True)

        # Find the drug administration and holiday period causing the lowest MM
        # fraction
        min_index = df_holliday['MM fraction'].idxmin()
        g_no_drug_min = df_holliday.loc[min_index, 'Generations no drug']
        g_drug_min = df_holliday.loc[min_index, 'Generations drug']
        frac_min = df_holliday.loc[min_index, 'MM fraction']

        new_row_df = pd.DataFrame([{'bOC,MMr': 1.7 + round(i/10, 1),
                'Eigenvalue 1': eigenvalues[0], 'Eigenvalue 2': eigenvalues[1],
                'Eigenvalue 3': eigenvalues[2], 'Eigenvalue 4': eigenvalues[3],
                'period H': g_no_drug_min, 'period A': g_drug_min,
                'MM fraction': frac_min}])

        df_eigenvalues = pd.concat([df_eigenvalues, new_row_df], ignore_index=True)

    # Save the data
    save_dataframe(df_eigenvalues, 'df_eigenvalues_bOCMMr.csv',
                                             r'..\data\data_own_model_fractions')

Dataframe_bOCMMr_eigenvalues()

def Figure_WMMd_IH_holiday_3D():
    """ Function that makes a table of the eigenvalues of the interaction matrix
    and the best holiday (H) and administration (A) period for different bOC,MMr
    values."""

    # Set initial parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrix when no drugs are pressent
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.77, 0.2]])

    # Drug inhibitor effect
    WMMd_inhibitor = 1.0

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
    df_holliday = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            freq_tumour = mimimal_tumour_freq_t_steps(t_steps_drug, t_steps_no_drug,
                                xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                                matrix_no_drugs, matrix_no_drugs, WMMd_inhibitor)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': int(t_steps_no_drug),
            'Generations drug': int(t_steps_drug), 'MM fraction': float(freq_tumour)}])
            df_holliday = pd.concat([df_holliday, new_row_df], ignore_index=True)

    # Save the data
    save_dataframe(df_holliday, 'df_cell_freq_best_WMMd_IH_holiday.csv',
                                                     r'..\data\data_own_model')

    # Find the drug administration and holiday period causing the lowest MM fraction
    min_index = df_holliday['MM fraction'].idxmin()
    g_no_drug_min = df_holliday.loc[min_index, 'Generations no drug']
    g_drug_min = df_holliday.loc[min_index, 'Generations drug']
    frac_min = df_holliday.loc[min_index, 'MM fraction']

    print(f"""Lowest MM fraction: {frac_min} -> WMMd IH holidays are {g_no_drug_min}
    generations and WMMd IH administrations are {g_drug_min} generations""")

    # Avoid errors because of the wrong datatype
    df_holliday['Generations no drug'] = pd.to_numeric(df_holliday[\
                                    'Generations no drug'], errors='coerce')
    df_holliday['Generations drug'] = pd.to_numeric(df_holliday['Generations drug'],
                                                                errors='coerce')
    df_holliday['MM fraction'] = pd.to_numeric(df_holliday['MM fraction'],
                                                            errors='coerce')

    # Make a meshgrid for the plot
    X = df_holliday['Generations no drug'].unique()
    Y = df_holliday['Generations drug'].unique()
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((20, 20))

    # Fill the 2D array with the MM frequency values by looping over each row
    for index, row in df_holliday.iterrows():
        i = int(row.iloc[0]) -2
        j = int(row.iloc[1]) -2
        Z[j, i] = row.iloc[2]

    # Make a 3D Figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')

    # Add labels
    ax.set_xlabel('Generations no WMMd IH')
    ax.set_ylabel('Generations WMMd IH')
    ax.set_zlabel('MM fraction')
    ax.set_title("""Average MM fraction with varing WMMd inhibitor
    administration and holiday periods""")

    # Turn to the right angle
    ax.view_init(elev = 32, azim = 20)

    # Add a color bar
    color_bar = fig.colorbar(surf, shrink = 0.6)
    color_bar.set_label('The MM fraction')

    save_Figure(fig, '3d_plot_cell_freq_best_WMMd_IH_holiday',
                                        r'..\visualisation\results_own_model')
    plt.show()

Figure_WMMd_IH_holiday_3D()


def Figure_MMd_GF_IH_holiday():
    """ Figure that shows the average MM frequency for different MMd GF inhibitor
    holiday and administration periods."""
    # Set initial parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrix when no drugs are pressent
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.77, 0.2]])

    # Payoff matrix when GF inhibitor drugs are pressent
    matrix_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [0.5, 0, 0.2, 0.0],
        [1.9, 0, -0.77, 0.2]])

    t = np.linspace(0, 100, 100)

    # Make a dictionary
    dict_freq_tumour_t_step = {}

    # Loop over al the t_step values
    for t_steps in range(2, 16):
        freq_tumour = mimimal_tumour_freq_t_steps(t_steps, t_steps, xOC, xOB, xMMd,
                    xMMr, N, cOC, cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs)
        dict_freq_tumour_t_step[t_steps] = freq_tumour

    # Save the data
    save_dictionary(dict_freq_tumour_t_step,
            r'..\data\data_own_model\dict_cell_freq_best_MMd_GH_IH_holiday.csv')

    # Convert the keys and values of the dictionary to lists
    keys_t_step = list(dict_freq_tumour_t_step.keys())
    values_MM_frequency = list(dict_freq_tumour_t_step.values())

    # Create a plot
    plt.plot(keys_t_step, values_MM_frequency, color = 'purple', linestyle='--',
                                                                    marker = 'o')
    plt.xlabel('Duration adminstartion and holiday periods')
    plt.ylabel('Average MM frequency')
    plt.title("""Average MM frequency with varing MMd GF inhibitor
    administration and holiday periods""")
    plt.grid(True)
    save_Figure(plt, 'line_plot_cell_freq_best_MMd_GF_IH_holiday',
                                            r'..\visualisation\results_own_model')
    plt.show()

def Figure_WMMd_IH_holiday():
    """ Figure that shows the average MM frequency for different WMMd inhibitor
    holiday and administration periods."""
    # Set initial parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrix when no drugs are pressent
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.785, 0.2]])

    # Drug inhibitor effect
    WMMd_inhibitor = 1.1

    # Make a dictionary
    dict_freq_tumour_t_step = {}

    # Loop over al the t_step values
    for t_steps in range(2, 18):
        freq_tumour = mimimal_tumour_freq_t_steps(t_steps, t_steps, xOC, xOB, xMMd,
                                    xMMr, N, cOC, cOB, cMMd, cMMr, matrix_no_drugs,
                                    matrix_no_drugs, WMMd_inhibitor)
        dict_freq_tumour_t_step[t_steps] = freq_tumour

    # Save the data
    save_dictionary(dict_freq_tumour_t_step,
                r'..\data\data_own_model\dict_cell_freq_best_WMMd_IH_holiday.csv')

    # Convert the keys and values of the dictionary to lists
    keys_t_step = list(dict_freq_tumour_t_step.keys())
    values_MM_frequency = list(dict_freq_tumour_t_step.values())

    # Create a plot
    plt.plot(keys_t_step, values_MM_frequency, color = 'purple', linestyle='--',
                                                                    marker = 'o')
    plt.xlabel('Duration adminstartion and holiday periods')
    plt.ylabel('Average MM frequency')
    plt.title("""Average MM frequency with varing WMMd inhibitor
    administration and holiday periods""")
    plt.grid(True)
    save_Figure(plt, 'line_plot_cell_freq_WMMd_IH_drug_holiday',
                                            r'..\visualisation\results_own_model')
    plt.show()

# Set initial parameter values
N = 50
cMMr = 1.3
cMMd = 1.2
cOB = 0.8
cOC = 1
xOC = 0.2
xOB = 0.3
xMMd = 0.2
xMMr = 0.3

# Payoff matrix when no drugs are pressent
matrix_no_drugs = np.array([
    [0.0, 1.6, 2.2, 1.9],
    [1.0, 0.0, -0.5, -0.5],
    [2.2, 0, 0.2, 0.0],
    [1.9, 0, -0.77, 0.2]])

# Payoff matrix when GF inhibitor drugs are pressent
matrix_drugs = np.array([
    [0.0, 1.6, 2.2, 1.9],
    [1.0, 0.0, -0.5, -0.5],
    [0.5, 0, 0.2, 0.0],
    [1.9, 0, -0.77, 0.2]])

# Make a dictionary
dict_freq_MM_t_step_MMd_GF = {}

# Loop over al the t_step values
for t_steps in range(2, 16):
    freq_tumour = mimimal_tumour_freq_t_steps(t_steps, t_steps, xOC, xOB, xMMd, xMMr, N,
                          cOC, cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs)
    dict_freq_MM_t_step_MMd_GF[t_steps] = freq_tumour

# Determine the optimized values
min_key = min(dict_freq_MM_t_step_MMd_GF, key=dict_freq_MM_t_step_MMd_GF.get)
min_value = min(dict_freq_MM_t_step_MMd_GF.values())
print(f'The MMd GF inhibitor holiday time: {min_key},',
                                f'giving a average MM frequency of {min_value}')

# # Make a Figure
# Figure_MMd_GF_IH_holiday()

WMMd_inhibitor = 1.1

# Make a dictionary
dict_freq_MM_t_step_WMMd = {}

# Loop over al the t_step values
for t_steps in range(2, 18):
    freq_tumour = mimimal_tumour_freq_t_steps(t_steps, t_steps, xOC, xOB, xMMd,
                                xMMr, N, cOC, cOB, cMMd, cMMr, matrix_no_drugs,
                                matrix_no_drugs, WMMd_inhibitor)
    dict_freq_MM_t_step_WMMd[t_steps] = freq_tumour

# Determine the optimized values
min_key = min(dict_freq_MM_t_step_WMMd, key=dict_freq_MM_t_step_WMMd.get)
min_value = min(dict_freq_MM_t_step_WMMd.values())
print(f'The WMMd inhibitor holiday time: {min_key},',f'giving a average MM frequency of {min_value}')

# Make a Figure
# Figure_WMMd_IH_holiday()


""" Best point to stop the giving of GF inhibition drugs"""
def one_time_switch_dataframe(t_steps, xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd,
                        cMMr, matrix_no_drugs, matrix_drugs, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the xOC, xOB, xMMd and xMMr values over
    time for a given period that drugs are once administerd.

    Parameters:
    -----------
    t_steps: Int
        The number of time steps drugs are administared and the breaks are for
        the different Figures.
    xOC: Float
        Frequency of OCs.
    xOB: Float
        Frequency of OBs.
    xMMr: Float
        Frequency of the resistant MM cells.
    xMMd: Float
        Frequency of the drug-sensitive MM cells.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter drug-sensitive MM cells.
    cMMr: Float
        Cost parameter resistant MM cells.
    matrix_no_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when no drugs are administrated.
    matrix_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when drugs are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the drug-sensitive MM cells.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the xOC, xOB, xMMd and xMMr values over time.
    """
    # set start values and make a dataframe
    x = 0
    time = 0
    t = np.linspace(0, 30, 30)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)
    df_total_switch = pd.DataFrame()

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += 30

    # Perform a numver of switches
    for i in range(2):

        # If x = 0 make sure the MMd is inhibited
        if x == 0:

            # Determine the start frequency values
            xOC = df_total_switch['xOC'].iloc[-1]
            xOB = df_total_switch['xOB'].iloc[-1]
            xMMd = df_total_switch['xMMd'].iloc[-1]
            xMMr = df_total_switch['xMMr'].iloc[-1]

            t = np.linspace(time, time + t_steps , t_steps)
            y0 = [xOC, xOB, xMMd, xMMr]
            parameters = (N, cOC, cOB, cMMd, cMMr, matrix_drugs, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

            # Add dataframe tot total dataframe
            df_total_switch = pd.concat([df_total_switch, df])
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 1
            time += t_steps

        # If x = 1 make sure the MMd is not inhibited
        else:
            # Determine the start frequency values
            xOC = df_total_switch['xOC'].iloc[-1]
            xOB = df_total_switch['xOB'].iloc[-1]
            xMMd = df_total_switch['xMMd'].iloc[-1]
            xMMr = df_total_switch['xMMr'].iloc[-1]

            t = np.linspace(time, 100, 100- time)
            y0 = [xOC, xOB, xMMd, xMMr]
            parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

            # Add dataframe tot total dataframe
            df_total_switch = pd.concat([df_total_switch, df])
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 0
            time += t_steps

    return df_total_switch

# Set initial values
N = 50
cMMr = 1.3
cMMd = 1.2
cOB = 0.8
cOC = 1
xOC = 0.2
xOB = 0.3
xMMd = 0.2
xMMr = 0.3

# Payoff matrix when no drugs are present
matrix_no_drugs = np.array([
    [0.0, 1.6, 2.2, 2.0],
    [1.0, 0.0, -0.5, -0.5],
    [1.8, 0, 0.2, 0.0],
    [2.0, 0, -0.785, 0.2]])

# Payoff matrix when GF inhibitor drugs are present
matrix_drugs = np.array([
    [0.0, 1.6, 2.2, 2.0],
    [1.0, 0.0, -0.5, -0.5],
    [0.7, 0, 0.2, 0.0],
    [2.0, 0, -0.785, 0.2]])

WMMd_inhibitor = 1.0

# # Loop over the different t step values
# for t_steps in range(2, 30):
#     df = one_time_switch_dataframe(t_steps, xOC, xOB, xMMd, xMMr, N, cOC, cOB,
#                                     cMMd, cMMr, matrix_no_drugs, matrix_drugs)
#
#     # If the xMMr is larger than xMMd the drug administartion has to stop
#     if df['xMMd'].iloc[-1] < df['xMMr'].iloc[-1]:
#         t_step_final_MMd_GF_IH = t_steps -1
#         break
#
# print(f"""The generation at which giving drugs that inhibts the GF from
# OC for MMd has to stop:{t_step_final_MMd_GF_IH}""")
#
#
# # Loop over the different t step values
# for t_steps in range(2, 30):
#     df = one_time_switch_dataframe(t_steps, xOC, xOB, xMMd, xMMr, N, cOC, cOB,
#                 cMMd, cMMr, matrix_no_drugs, matrix_no_drugs, WMMd_inhibitor)
#
#     # If the xMMr is larger than xMMd the drug administartion has to stop
#     if df['xMMd'].iloc[-1] < df['xMMr'].iloc[-1]:
#         t_step_final_WMMd_IH = t_steps -1
#         break
#
# print(f"""The generation at which giving drugs that inhibts the MMd fitness has
# to stop:{t_step_final_WMMd_IH}""")


""" The effect of the two different drugs """
def Figure_effect_two_drugs():
    """Function that compares the effect of the two types of drugs"""
    # Set new start parameter value
    N = 50
    cMMr = 1.1
    cMMd = 1.0
    cOB = 0.6
    cOC = 0.8
    xOC = 0.3
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.2

    # Payoff matrix
    matrix = np.array([
        [0.0, 1.6, 2.2, 2.1],
        [1.0, 0.0, -0.4, -0.4],
        [2.2, 0, 0.2, 0],
        [2.1, 0, -0.7, 0.2]])

    # Initial conditions
    t = np.linspace(0, 100, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_no_drug = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Set new start parameter value
    xOC = 0.3
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.2

    # Set initial conditions
    t = np.linspace(0, 25, 25)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Determine the current frequencies
    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    WMMd_inhibitor_1 = 1.0

    # Initial conditions
    t = np.linspace(25, 100, 75)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor_1)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_WMMd_inhibitor = pd.concat([df_1, df_2])

    # Set new start parameter value
    xOC = 0.3
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.2

    # Initial conditions
    t = np.linspace(0, 25, 25)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Determine the current frequencies
    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    # payoff matrix
    matrix = np.array([
        [0.0, 1.6, 2.2, 2.1],
        [1.0, 0.0, -0.4, -0.4],
        [1.0, 0, 0.2, 0],
        [2.1, 0, -0.7, 0.2]])

    # Initial conditions
    t = np.linspace(25, 100, 75)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_GF_inhibitor = pd.concat([df_1, df_2])

    # Set new start parameter value
    xOC = 0.3
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.2

    # Payoff matrix
    matrix = np.array([
        [0.0, 1.6, 2.2, 2.1],
        [1.0, 0.0, -0.4, -0.4],
        [2.2, 0, 0.2, 0],
        [2.1, 0, -0.7, 0.2]])

    # Initial conditions
    t = np.linspace(0, 25, 25)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                                'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Determine the current frequencies
    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    # Payoff matrix
    matrix = np.array([
        [0.0, 1.6, 2.2, 2.1],
        [1.0, 0.0, -0.4, -0.4],
        [1.0, 0, 0.2, 0],
        [2.1, 0, -0.9, 0.2]])

    # Initial conditions
    t = np.linspace(25, 100, 75)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor_1)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})
    df_GF_and_WMMd_inhibitor = pd.concat([df_1, df_2])

    # Save the data
    save_dataframe(df_no_drug, 'df_no_drug.csv', r'..\data\data_own_model')
    save_dataframe(df_WMMd_inhibitor, 'df_WMMd_inhibitor.csv',
                                                    r'..\data\data_own_model')
    save_dataframe(df_GF_inhibitor,'df_GF_inhibitor.csv', r'..\data\data_own_model')
    save_dataframe(df_GF_and_WMMd_inhibitor, 'df_GF_and_WMMd_inhibitor.csv',
                                                    r'..\data\data_own_model')

    # Create a Figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot the no drug data in the first subplot
    df_no_drug.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                        label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                                'Frequency MMr'], ax=axs[0, 0])
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title(f'Dynamics when no drugs are administerd')
    axs[0, 0].legend(loc = 'upper right')
    axs[0, 0].grid(True)

    # Plot the WMMd inhibitor data in the second subplot
    df_WMMd_inhibitor.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                         label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                                'Frequency MMr'], ax=axs[0, 1])
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_title(f'Dynamics when MMd fitness inhibitors are administered')
    axs[0, 1].legend(loc = 'upper right')
    axs[0, 1].grid(True)

    # Plot the MMd GF inhibitor data in the third subplot
    df_GF_inhibitor.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                         label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                                'Frequency MMr'], ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title(f"Dynamics when MMd GF inhibitors are administerd")
    axs[1, 0].legend(loc = 'upper right')
    axs[1, 0].grid(True)

    # Plot the MMd GF inhibitor and WMMd inhibitor data in the fourth subplot
    df_GF_and_WMMd_inhibitor.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                        label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                                'Frequency MMr'], ax=axs[1, 1])
    axs[1, 1].set_xlabel('Generations')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title(f"Dynamics when MMd fitness inhibitors and MMd GF inhibitors are administerd")
    axs[1, 1].legend(loc = 'upper right')
    axs[1, 1].grid(True)

    plt.tight_layout()
    save_Figure(plt, 'line_plot_cell_freq_drug_effect',
                                            r'..\visualisation\results_own_model')
    plt.show()

# Figure_effect_two_drugs()

"""-------------------Effet AT therapy-------------------------------"""
def Figure_3_senarios_WMMd_IH(n_switches, t_steps_drug):
    """ Function that makes a Figure that shows the effect of the time of a
    drug holiday and administration period.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the number of time steps drugs are administared and the breaks
        are for the different Figures.
    """
    # Set start parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.3
    xOB = 0.2
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrix
    matrix = np.array([
        [0.0, 1.6, 2.2, 1.8],
        [1.0, 0.0, -0.5, -0.5],
        [2.4, 0, 0.2, 0.0],
        [1.8, 0, -0.754, 0.2]])

    WMMd_inhibitor = 1.33

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_1 = pronto_switch_dataframe(n_switches, t_steps_drug[0],
                            t_steps_drug[0], xOC, xOB, xMMd, xMMr, N, cOC, cOB,
                            cMMd, cMMr, matrix, matrix, WMMd_inhibitor)
    df_total_switch_2 = pronto_switch_dataframe(n_switches, t_steps_drug[1],
                            t_steps_drug[1], xOC, xOB, xMMd, xMMr, N, cOC, cOB,
                            cMMd, cMMr, matrix, matrix, WMMd_inhibitor)
    df_total_switch_3 = pronto_switch_dataframe(n_switches, t_steps_drug[2],
                            t_steps_drug[2], xOC, xOB, xMMd, xMMr, N, cOC, cOB,
                            cMMd, cMMr, matrix, matrix, WMMd_inhibitor)

    t = np.linspace(0, 20, 20)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

    # Determine the current frequencies
    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    t = np.linspace(20, 100, 80)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

    # Combine the dataframes
    df_total = pd.concat([df_1, df_2])

    # Save the data
    save_dataframe(df_total_switch_1, 'df_cell_freq_G8_WMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_total_switch_2, 'df_cell_freq_G10_WMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_total_switch_3, 'df_cell_freq_G12_WMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_total, 'df_cell_freq_WMMd_inhibit_continuously.csv',
                                                     r'..\data\data_own_model')

    # Create a Figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Plot the data without drug holidays in the first plot
    df_total.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[ \
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[0, 0])
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel('MM frequency')
    axs[0, 0].set_title(f'Dynamics when drugs are administered continuously')
    axs[0, 0].legend(loc = 'upper right')
    axs[0, 0].grid(True)

    # Plot the data with drug holidays in the second plot
    df_total_switch_1.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[0, 1])
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel('MM frequency')
    axs[0, 1].set_title(f'Dynamics when WMMd inhibitors are administered every 8 generations')
    axs[0, 1].legend(loc = 'upper right')
    axs[0, 1].grid(True)

    # Plot the data with drug holidays in the third plot
    df_total_switch_2.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations')
    axs[1, 0].set_ylabel('MM frequency')
    axs[1, 0].set_title(f'Dynamics when WMMd inhibitors are administered every 10 generations')
    axs[1, 0].legend(loc = 'upper right')
    axs[1, 0].grid(True)
    plt.grid(True)

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_3.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[1, 1])
    axs[1, 1].set_xlabel('Generations')
    axs[1, 1].set_ylabel('MM frequency')
    axs[1, 1].set_title(f'Dynamics when WMMd inhibitors are administered every 12 generations')
    axs[1, 1].legend(loc = 'upper right')
    axs[1, 1].grid(True)
    save_Figure(plt, 'line_plot_cell_freq_WMMd_inhibit',
                                          r'..\visualisation\results_own_model')
    plt.show()

# Create a Figure that shows the effect of drug holidays
list_t_steps_drug = [8, 10, 12]
Figure_3_senarios_WMMd_IH(10, list_t_steps_drug)



""" ---------- Fitness when drugs are added -------------------------- """
def Figure_freq_fitness_dynamics():
    """Function that makes Figure of the OC, OB, MMd and MMr frequency and fitness
     values over the time"""

    # Set start values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.15
    xOB = 0.4
    xMMd = 0.25
    xMMr = 0.2

    # Payoff matrix
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.2, 1.8],
        [0.9, 0.0, -0.5, -0.5],
        [2.2, 0.0, 0.2, 0.0],
        [1.8, 0.0, -0.75, 0.2]])

    t = np.linspace(0, 25, 25)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1_MMd_GF_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Determine the current frequencies
    xOC = df_1_MMd_GF_inhibition['xOC'].iloc[-1]
    xOB = df_1_MMd_GF_inhibition['xOB'].iloc[-1]
    xMMd = df_1_MMd_GF_inhibition['xMMd'].iloc[-1]
    xMMr = df_1_MMd_GF_inhibition['xMMr'].iloc[-1]

    # Payoff matrix
    matrix_drugs = np.array([
        [0.0, 1.6, 2.2, 2.1],
        [0.9, 0.0, -0.4, -0.4],
        [1.0, 0, 0.2, 0],
        [2.1, 0, -0.7, 0.2]])

    # Initial conditions
    t = np.linspace(25, 100, 75)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2_MMd_GF_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Combine the dataframes
    df_MMd_GF_inhibition = pd.concat([df_1_MMd_GF_inhibition, df_2_MMd_GF_inhibition])

    # Set new start parameter values
    xOC = 0.15
    xOB = 0.4
    xMMd = 0.25
    xMMr = 0.2
    t = np.linspace(0, 25, 25)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1_WMMd_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Determine the current frequencies
    xOC = df_1_WMMd_inhibition['xOC'].iloc[-1]
    xOB = df_1_WMMd_inhibition['xOB'].iloc[-1]
    xMMd = df_1_WMMd_inhibition['xMMd'].iloc[-1]
    xMMr = df_1_WMMd_inhibition['xMMr'].iloc[-1]

    # Initial conditions
    t = np.linspace(25, 100, 75)
    WMMd_inhibitor =  1.2
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2_WMMd_inhibition = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
                            'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Combine the dataframes
    df_WMMd_inhibition = pd.concat([df_1_WMMd_inhibition, df_2_WMMd_inhibition])

    # Make dataframes for the fitness values
    df_fitness_WMMd_inhibition_1 = freq_to_fitness_values(df_1_WMMd_inhibition, N,
                                          cOC, cOB, cMMd, cMMr, matrix_no_drugs)
    df_fitness_WMMd_inhibition_2 = freq_to_fitness_values(df_2_WMMd_inhibition, N,
                         cOC, cOB, cMMd, cMMr, matrix_no_drugs, WMMd_inhibitor)
    df_fitness_WMMd_inhibition = pd.concat([df_fitness_WMMd_inhibition_1,
                                df_fitness_WMMd_inhibition_2], ignore_index=True)

    df_fitness_MMd_GF_inhibition_1 = freq_to_fitness_values(df_1_MMd_GF_inhibition,
                                       N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)
    df_fitness_MMd_GF_inhibition_2 = freq_to_fitness_values(df_2_MMd_GF_inhibition,
                                          N, cOC, cOB, cMMd, cMMr, matrix_drugs)
    df_fitness_MMd_GF_inhibition = pd.concat([df_fitness_MMd_GF_inhibition_1,
                              df_fitness_MMd_GF_inhibition_2], ignore_index=True)

    # Save the data
    save_dataframe(df_WMMd_inhibition, 'df_cell_freq_cWMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_MMd_GF_inhibition, 'df_cell_freq_MMd_GF_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_fitness_WMMd_inhibition, 'df_fitness_WMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_fitness_MMd_GF_inhibition, 'df_fitness_MMd_GF_inhibit.csv',
                                                     r'..\data\data_own_model')

    # Create a Figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    # Plot first subplot
    df_WMMd_inhibition.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                        label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0, 0])
    axs[0, 0].set_xlabel('Generations')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title('Frequency dynamics when a WMMd inhibitor is administerd')
    axs[0, 0].legend(loc = 'upper right')
    axs[0, 0].grid(True)

    # Plot the second subplot
    df_fitness_WMMd_inhibition.plot(y=['WOC', 'WOB', 'WMMd', 'WMMr', 'W_average'],
                                label = ['Fitness OC', 'Fitness OB', 'Fitness MMd',
                                  'Fitness MMr', 'Average fitness'],  ax=axs[0, 1])
    axs[0, 1].set_title('Fitness dynamics when a WMMd inhibitor is administerd')
    axs[0, 1].set_xlabel('Generations')
    axs[0, 1].set_ylabel('Fitness')
    axs[0, 1].legend(['Fitness OC', 'Fitness OB', 'Fitness MMd', 'Fitness MMr',
                                                            'Average fitness'])
    axs[0, 1].legend(loc = 'upper right')
    axs[0, 1].grid(True)

    # Plot the third subplot
    df_MMd_GF_inhibition.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                        label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                                'Frequency MMr'], ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Frequency dynamics when a MMd GF inhibitor is administerd')
    axs[1, 0].legend(loc = 'upper right')
    axs[1, 0].grid(True)

    # Plot the fourth subplot
    df_fitness_MMd_GF_inhibition.plot(y=['WOC', 'WOB', 'WMMd', 'WMMr', 'W_average'],
                              label = ['Fitness OC', 'Fitness OB', 'Fitness MMd',
                                 'Fitness MMr', 'Average fitness'],  ax=axs[1, 1])
    axs[1, 1].set_title('Fitness dynamics when a MMd GF inhibitor is administerd')
    axs[1, 1].set_xlabel('Generations')
    axs[1, 1].set_ylabel('Fitness')
    axs[1, 1].legend(loc = 'upper right')
    axs[1, 1].grid(True)
    plt.tight_layout()
    save_Figure(plt, 'line_plot_cell_freq_fitness_drugs',
                                         r'..\visualisation\results_own_model')
    plt.show()

Figure_freq_fitness_dynamics()


""" AT therapy multidrug combinations """
# def Figure_3_senarios_MMd_GF_WMMd_IH(n_switches, t_steps_drug):
#     """ Function that makes a Figure that shows the effect of drug holidays.
#
#     Parameters:
#     -----------
#     n_switches: Int
#         The number of switches between giving drugs and not giving drugs.
#     t_steps_drug: List
#         List with the number of time steps drugs are administared and the breaks
#         are for the different Figures.
#     """
#     # Set start parameter values
#     N = 50
#     cMMr = 1.3
#     cMMd = 1.2
#     cOB = 0.8
#     cOC = 1
#     xOC = 0.3
#     xOB = 0.2
#     xMMd = 0.2
#     xMMr = 0.3
#
#     # Payoff matrices
#     matrix_no_drugs = np.array([
#         [0.0, 1.6, 2.2, 1.9],
#         [1.0, 0.0, -0.5, -0.5],
#         [2.2, 0, 0.2, 0.0],
#         [1.9, 0, -0.75, 0.2]])
#
#     matrix_drugs = np.array([
#         [0.0, 1.6, 2.2, 1.9],
#         [1.0, 0.0, -0.5, -0.5],
#         [0.64, 0, 0.2, 0.0],
#         [1.9, 0, -0.8, 0.2]])
#
#     matrix_no_drugs1 = np.array([
#         [0.0, 1.6, 2.2, 1.9],
#         [1.0, 0.0, -0.5, -0.5],
#         [2.2, 0, 0.2, 0.0],
#         [1.9, 0, -0.8, 0.2]])
#
#     matrix_drugs1 = np.array([
#         [0.0, 1.6, 2.2, 1.9],
#         [1.0, 0.0, -0.5, -0.5],
#         [1.42, 0, 0.2, 0.0],
#         [1.9, 0, -1.15, 0.2]])
#
#     WMMd_inhibitor = 1.08
#     WMMd_inhibitor1 = 0.54
#
#     # Make dataframe for the different drug hollyday duration values
#     df_total_switch_1 = switch_dataframe(n_switches, t_steps_drug[0], xOC, xOB,
#             xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix_no_drugs, matrix_drugs)
#     df_total_switch_2 = switch_dataframe(n_switches, t_steps_drug[1], xOC, xOB,
#             xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix_no_drugs, matrix_no_drugs, WMMd_inhibitor)
#     df_total_switch_3 = switch_dataframe(n_switches, t_steps_drug[2], xOC, xOB,
#             xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix_no_drugs1, matrix_drugs1, WMMd_inhibitor1)
#
#     t = np.linspace(0, 20, 20)
#     y0 = [xOC, xOB, xMMd, xMMr]
#     parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)
#
#     # Determine the ODE solutions
#     y = odeint(model_dynamics, y0, t, args=parameters)
#     df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
#                 'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})
#
#     # Determine the current frequencies
#     xOC = df_1['xOC'].iloc[-1]
#     xOB = df_1['xOB'].iloc[-1]
#     xMMd = df_1['xMMd'].iloc[-1]
#     xMMr = df_1['xMMr'].iloc[-1]
#
#     t = np.linspace(20, 100, 80)
#     y0 = [xOC, xOB, xMMd, xMMr]
#     parameters = (N, cOC, cOB, cMMd, cMMr, matrix_drugs)
#
#     # Determine the ODE solutions
#     y = odeint(model_dynamics, y0, t, args=parameters)
#     df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
#                 'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})
#
#     # Combine the dataframes
#     df_total = pd.concat([df_1, df_2])
#
#     # Save the data
#     save_dataframe(df_total_switch_1, 'df_cell_freq_G6_MMd_GF_inhibit.csv',
#                                                      r'..\data\data_own_model')
#     save_dataframe(df_total_switch_2, 'df_cell_freq_G8_MMd_GF_inhibit.csv',
#                                                      r'..\data\data_own_model')
#     save_dataframe(df_total_switch_3, 'df_cell_freq_G12_MMd_GF_inhibit.csv',
#                                                      r'..\data\data_own_model')
#     save_dataframe(df_total, 'df_cell_freq_MMd_GF_inhibit_continuously.csv',
#                                                      r'..\data\data_own_model')
#
#     # Create a Figure
#     fig, axs = plt.subplots(2, 2, figsize=(16, 10))
#
#     # Plot the data without drug holidays in the first plot
#     df_total.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[ \
#     'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[0, 0])
#     axs[0, 0].set_xlabel(' ')
#     axs[0, 0].set_ylabel('MM frequency')
#     axs[0, 0].set_title(f'Dynamics when MMd GF inhibitors are administered continuously')
#     axs[0, 0].legend(loc = 'upper right')
#     axs[0, 0].grid(True)
#
#     # Plot the data with drug holidays in the second plot
#     df_total_switch_1.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
#     'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[0, 1])
#     axs[0, 1].set_xlabel(' ')
#     axs[0, 1].set_ylabel('MM frequency')
#     axs[0, 1].set_title(f'Dynamics when MMd GF inhibitors are administered every 6 generations')
#     axs[0, 1].legend(loc = 'upper right')
#     axs[0, 1].grid(True)
#
#     # Plot the data with drug holidays in the third plot
#     df_total_switch_2.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
#     'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[1, 0])
#     axs[1, 0].set_xlabel('Generations')
#     axs[1, 0].set_ylabel('MM frequency')
#     axs[1, 0].set_title(f'Dynamics when MMd GF inhibitors are administered every 8 generations')
#     axs[1, 0].legend(loc = 'upper right')
#     axs[1, 0].grid(True)
#     plt.grid(True)
#
#     # Plot the data with drug holidays in the fourth plot
#     df_total_switch_3.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
#     'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[1, 1])
#     axs[1, 1].set_xlabel('Generations')
#     axs[1, 1].set_ylabel('MM frequency')
#     axs[1, 1].set_title(f'Dynamics when MMd GF inhibitors are administered every 10 generations')
#     axs[1, 1].legend(loc = 'upper right')
#     axs[1, 1].grid(True)
#     save_Figure(plt, 'line_plot_cell_freq_MMd_GF_inhibit',
#                                      r'..\visualisation\results_own_model')
#     plt.show()
#
# # Create a Figure that shows the effect of drug holidays
# list_t_steps_drug = [10, 10, 10]
# Figure_3_senarios_MMd_GF_WMMd_IH(10, list_t_steps_drug)

def Figure_3_senarios_MMd_GF_WMMd_IH(n_switches, t_steps_drug):
    """ Function that makes a Figure that shows the effect of drug holidays.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the number of time steps drugs are administared and the breaks
        are for the different Figures.
    """
    # Set start parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.3
    xOB = 0.2
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrices
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.8, 0.2]])

    matrix_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [0.64, 0, 0.2, 0.0],
        [1.9, 0, -0.8, 0.2]])

    matrix_drugs_half = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [1.42, 0, 0.2, 0.0],
        [1.9, 0, -1.0, 0.2]])

    WMMd_inhibitor = 1.08
    WMMd_inhibitor_half = 0.54

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_1 = pronto_switch_dataframe(n_switches, t_steps_drug[0],
                            t_steps_drug[0], xOC, xOB, xMMd, xMMr, N, cOC, cOB,
                            cMMd, cMMr, matrix_no_drugs, matrix_drugs)
    df_total_switch_2 = pronto_switch_dataframe(n_switches, t_steps_drug[1],
                        t_steps_drug[1], xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd,
                        cMMr, matrix_no_drugs, matrix_no_drugs, WMMd_inhibitor)
    df_total_switch_3 = pronto_switch_dataframe(n_switches, t_steps_drug[2],
                    t_steps_drug[2], xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                     matrix_no_drugs, matrix_drugs_half, WMMd_inhibitor_half)

    # Save the data
    save_dataframe(df_total_switch_1, 'df_cell_freq_G15_MMd_GF_IH.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_total_switch_2, 'df_cell_freq_G15_WMMd_IH.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_total_switch_3, 'df_cell_freq_G15_MMd_GF_WMMd_IH.csv',
                                                     r'..\data\data_own_model')

    # Create a Figure
    fig, axs = plt.subplots(3, 1, figsize=(10, 24))

    # Plot the data with MMd GF inhibitor drug holidays in the first plot
    df_total_switch_1.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[ \
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel(' ')
    axs[0].set_ylabel('MM frequency')
    axs[0].set_title(f"""Dynamics when MMd GF inhibitors are administered 15 generations""")
    axs[0].legend(loc = 'upper right')
    axs[0].set_xticklabels([])
    axs[0].grid(True)

    # Plot the data with WMMd inhibitor drug holidays in the second plot
    df_total_switch_2.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel(' ')
    axs[1].set_ylabel('MM frequency')
    axs[1].set_title(f"""Dynamics when WMMd inhibitors are administered every 15 generations""")
    axs[1].legend(loc = 'upper right')
    axs[1].set_xticklabels([])
    axs[1].grid(True)

    # Plot the data with MMd GF and WMMd inhibitor drug holidays in the third plot
    df_total_switch_3.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[2])
    axs[2].set_xlabel('Generations')
    axs[2].set_ylabel('MM frequency')
    axs[2].set_title(f"""Dynamics when MMd GF inhibitors and WMMd inhibitors are administered every 15 generations""")
    axs[2].legend(loc = 'upper right')
    axs[2].grid(True)
    plt.grid(True)
    save_Figure(plt, 'line_plot_cell_freq_G15_MMd_GF_WMMd_IH',
                                     r'..\visualisation\results_own_model')
    plt.show()

# # Create a Figure that shows the effect of drug holidays
# list_t_steps_drug = [15, 15, 15]
# Figure_3_senarios_MMd_GF_WMMd_IH(8, list_t_steps_drug)

def Figure_3_senarios_MMd_GF_WMMd_IH(n_switches, t_steps_drug):
    """ Function that makes a Figure that shows the effect of drug holidays.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the number of time steps drugs are administared and the breaks
        are for the different Figures.
    """
    # Set start parameter values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.3
    xOB = 0.2
    xMMd = 0.2
    xMMr = 0.3

    # Payoff matrices
    matrix_no_drugs = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.8, 0.2]])

    matrix_drugs_half = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [1.4, 0, 0.2, 0.0],
        [1.9, 0, -1.0, 0.2]])

    WMMd_inhibitor_half = 0.514

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_1 = pronto_switch_dataframe(n_switches, t_steps_drug[0],
                t_steps_drug[0], xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                matrix_no_drugs, matrix_drugs_half, WMMd_inhibitor_half)
    df_total_switch_2 = pronto_switch_dataframe(n_switches, t_steps_drug[1],
                t_steps_drug[1], xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                matrix_no_drugs, matrix_drugs_half, WMMd_inhibitor_half)
    df_total_switch_3 = pronto_switch_dataframe(n_switches, t_steps_drug[2],
                t_steps_drug[2], xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                matrix_no_drugs, matrix_drugs_half, WMMd_inhibitor_half)

    t = np.linspace(0, 20, 20)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

    # Determine the current frequencies
    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    t = np.linspace(20, 100, 80)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_drugs_half, WMMd_inhibitor_half)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total_MM': y[:, 3]+ y[:, 2]})

    # Combine the dataframes
    df_total = pd.concat([df_1, df_2])

    # Save the data
    save_dataframe(df_total_switch_1, 'df_cell_freq_G10_MMd_GF_WMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_total_switch_2, 'df_cell_freq_G12_MMd_GF_WMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_total_switch_3, 'df_cell_freq_G14_MMd_GF_WMMd_inhibit.csv',
                                                     r'..\data\data_own_model')
    save_dataframe(df_total, 'df_cell_freq_MMd_GF_WMMd_inhibit_continuously.csv',
                                                     r'..\data\data_own_model')

    # Create a Figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Plot the data without drug holidays in the first plot
    df_total.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[ \
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[0, 0])
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel('MM frequency')
    axs[0, 0].set_title(f"""Dynamics when MMd GF and WMMd inhibitors
    are administered continuously""")
    axs[0, 0].legend(loc = 'upper right')
    axs[0, 0].set_xticklabels([])
    axs[0, 0].grid(True)

    # Plot the data with drug holidays in the second plot
    df_total_switch_1.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[0, 1])
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel('MM frequency')
    axs[0, 1].set_title(f"""Dynamics when MMd GF and WMMd inhibitors are
    administered every 10 generations""")
    axs[0, 1].legend(loc = 'upper right')
    axs[0, 1].set_xticklabels([])
    axs[0, 1].grid(True)

    # Plot the data with drug holidays in the third plot
    df_total_switch_2.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations')
    axs[1, 0].set_ylabel('MM frequency')
    axs[1, 0].set_title(f"""Dynamics when MMd GF and WMMd inhibitors are
    administered every 12 generations""")
    axs[1, 0].legend(loc = 'upper right')
    axs[1, 0].grid(True)
    plt.grid(True)

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_3.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'], label=[\
    'Frequency OC', 'Frequency OB', 'Frequency MMd','Frequency MMr'], ax=axs[1, 1])
    axs[1, 1].set_xlabel('Generations')
    axs[1, 1].set_ylabel('MM frequency')
    axs[1, 1].set_title(f"""Dynamics when MMd GF and WMMd inhibitors are
    administered every 14 generations""")
    axs[1, 1].legend(loc = 'upper right')
    axs[1, 1].grid(True)
    save_Figure(plt, 'line_plot_cell_freq_MMd_GF_WMMd_inhibit',
                                     r'..\visualisation\results_own_model')
    plt.show()

# Create a Figure that shows the effect of drug holidays
# list_t_steps_drug = [10, 12, 14]
# Figure_3_senarios_MMd_GF_WMMd_IH(8, list_t_steps_drug)




def Figure_freq_fitness_dynamics_change_N():
    """Function that makes Figure of the OC, OB, MMd and MMr frequency and fitness
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
        [0.0, 1.6, 1.8, 2.1],
        [1.0, 0.0, -0.3, -0.3],
        [2, 0, 0.2, 0],
        [2.1, 0, -0.6, 0.2]])

    t = np.linspace(0, 30, 30)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
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

    # Determine the ODE solutions
    y = odeint(model_dynamics_change_N, y0, t, args=parameters)
    df_Figure_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_fitness_first_line= freq_to_fitness_values(df_Figure_first_line, N, cOC,
                                                        cOB, cMMd, cMMr, matrix)
    df_fitness_second_line =freq_to_fitness_values_change_N(df_Figure_second_line,
                                                N, cOC, cOB, cMMd, cMMr, matrix)

    # Create a Figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # Plot first subplot
    df_Figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
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
    df_Figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
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

Figure_freq_fitness_dynamics_change_N()


def Figure_freq_fitness_dynamics():
    """Function that makes Figure of the OC, OB, MMd and MMr frequency and fitness
     values over the time"""

    # Set start values
    N = 50
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.0, 1.6, 2.2, 1.9],
        [1.0, 0.0, -0.5, -0.5],
        [1.501, 0, 0.2, 0.0],
        [1.9, 0, -0.63, 0.2]])

    t = np.linspace(0, 180, 180)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_fitness_first_line= freq_to_fitness_values(df_Figure_first_line, N, cOC, cOB, cMMd, cMMr, matrix)
    df_fitness_second_line =freq_to_fitness_values(df_Figure_second_line, N, cOC, cOB, cMMd, cMMr, matrix)

    # Create a Figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # Plot first subplot
    df_Figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
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
    df_Figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
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

Figure_freq_fitness_dynamics()

def Figure_freq_dynamics():
    """Function that makes Figure of the xOC, xOB, xMMd and xMMr values over the time.
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})


    # Create a Figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_Figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where cOB<cOC<cMM<cMMd ')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_Figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where cOB<cOC<cMM<cMMd ')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

Figure_freq_dynamics()

def Figure_interaction_dynamics():
    """Function that makes Figure of the xOC, xOB, xMMd and xMMr values over the time.
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    # Set new start parameter value
    matrix = np.array([
        [0.0, 1.4, 2.0, 1.5],
        [0.6, 0.0, -0.3, -0.3],
        [2.0, 0, 0.3, 0],
        [1.5, 0, -0.2, 0.2]])


    t = np.linspace(0, 80, 80)

    # Initial conditions
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})


    # Create a Figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_Figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where cOB<cOC<cMMr<cMMd ')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_Figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where cOB<cOC<cMMr<cMMd')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

Figure_interaction_dynamics()

def Figure_freq_dynamics_2():
    """Function that makes Figure of the xOC, xOB, xMMd and xMMr values over the time.
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})


    # Create a Figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_Figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where cOB<cOC<cMM<cMMd ')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_Figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where cOB<cOC<cMM<cMMd ')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

Figure_freq_dynamics_2()

def Figure_freq_dynamics_drugs():
    """Function that makes Figure of the xOC, xOB, xMMd and xMMr values over the time.
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
        [0.0, 1.6, 2.2, 2.1],
        [1.0, 0.0, -0.4, -0.4],
        [2.2, 0, 0.2, 0],
        [2.1, 0, -0.7, 0.2]])

    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    WMMd_inhibitor_1 = 1.5

    # Initial conditions
    t = np.linspace(40, 140, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor_1)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_Figure_first_line = pd.concat([df_1, df_2])

    # Set new start parameter value
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.1

    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    # Payoff matrix
    WMMd_inhibitor_2 = 0.8

    # Initial conditions
    t = np.linspace(40, 140, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor_2)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_Figure_second_line = pd.concat([df_1, df_2])

    # Create a Figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_Figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Dynamics when drugs are added (strength {WMMd_inhibitor_1})')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_Figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Dynamics when drugs are added (strength {WMMd_inhibitor_2})')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

Figure_freq_dynamics_drugs()

def Figure_freq_dynamics_drug_strength():
    """Function that makes Figure of the xOC, xOB, xMMd and xMMr values over the time.
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
        [0.0, 1.6, 2.1, 2.0],
        [1.0, 0.0, -0.3, -0.2],
        [2.2, 0, 0.2, 0.0],
        [2.0, 0, -0.5, 0.2]])


    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    WMMd_inhibitor_1 = 1.5

    # Initial conditions
    t = np.linspace(40, 90, 60)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor_1)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})


    xOC = df_2['xOC'].iloc[-1]
    xOB = df_2['xOB'].iloc[-1]
    xMMd = df_2['xMMd'].iloc[-1]
    xMMr = df_2['xMMr'].iloc[-1]

    WMMd_inhibitor_1 = 0.0

    # Initial conditions
    t = np.linspace(90, 140, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor_1)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_3 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_Figure_first_line = pd.concat([df_1, df_2, df_3])


    # Set new start parameter value
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.0, 1.6, 2.1, 2.0],
        [1.0, 0.0, -0.3, -0.2],
        [2.2, 0, 0.2, 0.0],
        [2.0, 0, -0.5, 0.2]])


    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    WMMd_inhibitor_2 = 0.9

    # Initial conditions
    t = np.linspace(40, 90, 50)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor_2)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_2['xOC'].iloc[-1]
    xOB = df_2['xOB'].iloc[-1]
    xMMd = df_2['xMMd'].iloc[-1]
    xMMr = df_2['xMMr'].iloc[-1]

    WMMd_inhibitor_2 = 0.0

    # Initial conditions
    t = np.linspace(90, 140, 50)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor_2)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_3 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_Figure_second_line = pd.concat([df_1, df_2, df_3])

    # Set new start parameter value
    xOC = 0.4
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.1

    # Payoff matrix
    matrix = np.array([
        [0.0, 1.6, 2.1, 2.0],
        [1.0, 0.0, -0.3, -0.2],
        [2.2, 0, 0.2, 0.0],
        [2.0, 0, -0.5, 0.2]])


    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    WMMd_inhibitor_3 = 0.6

    # Initial conditions
    t = np.linspace(40, 140, 100)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor_3)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_Figure_third_line = pd.concat([df_1, df_2])

    # Create a Figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot first line data in the first subplot
    df_Figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Dynamics when drugs are added at G 40\n and stoped at G 90 (strength 1.5)')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_Figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Dynamics when drugs are added at G 40 \n and stoped at G 90 (strength 0.9)')
    axs[1].legend(loc='upper left')
    plt.tight_layout()

    # Plot second line data in the third subplot
    df_Figure_third_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[2])
    axs[2].set_xlabel('Generations')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title(f'Dynamics when drugs are added \n at G 40 (strength {WMMd_inhibitor_3})')
    axs[2].legend()
    plt.tight_layout()
    plt.show()

Figure_freq_dynamics_drug_strength()

def Figure_freq_dynamics_GF_inhibition():
    """Function that makes Figure of the xOC, xOB, xMMd and xMMr values over the time.
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

    # Determine the ODE solutions
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_Figure_first_line = pd.concat([df_1, df_2])

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

    # Determine the ODE solutions
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_Figure_second_line = pd.concat([df_1, df_2])

    # Create a Figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_Figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where GF from OC for MM get inhibited (works)')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_Figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where GF from OC for MM get inhibited (works not)')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

Figure_freq_dynamics_GF_inhibition()

def Figure_freq_dynamics_GF_inhibition_short():
    """Function that makes Figure of the xOC, xOB, xMMd and xMMr values over the time.
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

    # Determine the ODE solutions
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

    # Determine the ODE solutions
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_3 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_Figure_first_line = pd.concat([df_1, df_2, df_3])

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

    # Determine the ODE solutions
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

    # Determine the ODE solutions
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

    # Determine the ODE solutions
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_4 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})


    df_Figure_second_line = pd.concat([df_1, df_2, df_3, df_4])


    # Create a Figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_Figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where GF for MM from OC get inhibited')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_Figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where GF for MM from OC get inhibited')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

Figure_freq_dynamics_GF_inhibition_short()

def Figure_freq_dynamics_decrease_MMd():
    """Function that makes Figure of the xOC, xOB, xMMd and xMMr values over the time.
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

    # Determine the ODE solutions
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_Figure_first_line = pd.concat([df_1, df_2])

    # Set new start parameter value
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.3
    xMMr = 0.2

    # Initial conditions
    t = np.linspace(0, 55, 65)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the ODE solutions
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_Figure_second_line = pd.concat([df_1, df_2])


    # Create a Figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot first line data in the first subplot
    df_Figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                 label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                 'Frequency MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Dynamics for a scenario where MMd gets reduced (works)')
    axs[0].legend()

    # Plot second line data in the second subplot
    df_Figure_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                                  label=['Frequency OC', 'Frequency OB', 'Frequency MMd',
                                  'Frequency MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Dynamics for a scenario where MMd gets reduced (works not)')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

Figure_freq_dynamics_decrease_MMd()

def Figure_freq_dynamics_resistance():
    """Function that makes Figure of the xOC, xOB, xMMd and xMMr values over the time.
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

    # Determine the ODE solutions
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

    # Determine the ODE solutions
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

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_3 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMMd': y[:, 2], 'xMMr': y[:, 3]})

    df_Figure_first_line = pd.concat([df_1, df_2, df_3])

    # Plot first line data in the first subplot
    df_Figure_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
         label=['Frequency OC', 'Frequency OB', 'Frequency MMd', 'Frequency MMr'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where MMr develops')
    plt.legend()
    plt.show()

Figure_freq_dynamics_resistance()
