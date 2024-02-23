"""
Author:       Eva Nieuwenhuis
University:   UvA
Student id':  13717405
Description:  Code of the model that simulates the dynamics in the multiple myeloma
              (MM) microenvironment with four cell types: drug-sensitive MM cells
              (MMd), resistant MM cells (MMr), osteoblasts (OBs) and osteoclasts
              (OCs). The model is a public goods game in the framework of evolutionary
              game theory with collective interactions and linear benefits. In this
              model there is looked at the numbers of the four cell types.
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

def dOC_dt(nOC, nOB, nMMd, nMMr, N, cOC, cOB, cMMd, cMMr, gr_OC, dr_OC, matrix):
    """
    Function that calculates the change in number of osteoclasts.

    Parameters:
    -----------
    nOC: Float
        Frequency of OCs.
    nOB: Float
        Frequency of OBs.
    nMMd: Float
        Frequency of the MMd.
    nMMr: Float
        Frequency of the MMr.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter MMd.
    cMMr: Float
        Cost parameter MMr.
    gr_OC: Float
        Growth rate of the OCs.
    dr_OC: Float
        Dacay rate of the OCs.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    change_nOC: Float
        Change in the number of OCs.

    Example:
    -----------
    >>> dOC_dt(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
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

    # Calculate the Change on in the number of OCs
    change_nOC = (a*nOC*cOC + b*nOB*cOB + c*nMMd*cMMd + d* nMMr *cMMr)*(N - 1)/N - cOC
    return change_nOC

def dOB_dt(nOC, nOB, nMMd, nMMr, N, cOC, cOB, cMMd, cMMr, gr_OB, dr_OB, matrix):
    """
    Function that calculates the change in the number of osteoblast.

    Parameters:
    -----------
    nOC: Float
        Number of OCs.
    nOB: Float
        Number of OBs.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter MMd.
    cMMr: Float
        Cost parameter MMr.
    gr_OB: Float
        Growth rate of the OBs.
    dr_OB: Float
        Dacay rate of the OBs.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    change_nOB: Float
        Change in the number of OBs.

    Example:
    -----------
    >>> dOB_dt(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
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

    # Calculate the change in number of OBs
    change_nOB = (e*nOC*cOC + f*nOB*cOB + g*nMMd*cMMd + h* nMMr*cMMr)*(N - 1)/N - cOB
    return change_nOB

def dMMd_dt(nOC, nOB, nMMd, nMMr, N, cOC, cOB, cMMd, cMMr, gr_MMd, dr_MMd, matrix,
                                                            WMMd_inhibitor = 0):
    """
    Function that calculates the change in the number of a drug-senstive MM cells.

    Parameters:
    -----------
    nOC: Float
        Number of OCs.
    nOB: Float
         Number of OBs.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter MMd.
    cMMr: Float
        Cost parameter MMr.
    gr_MMd: Float
        Growth rate of the MMd.
    dr_MMd: Float
        Decay rate of the MMd.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd.

    Returns:
    --------
    change_nMMd: Float
        Change in the number of MMd.

    Example:
    -----------
    >>> dMMd_dt(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
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

    # Calculate the change in the number of MMd
    change_nMMd = (i*nOC*cOC + j*nOB*cOB + k*nMMd*cMMd + l* nMMr*cMMr - WMMd_inhibitor \
                                                        * cMMd)*(N - 1)/N - cMMd
    return change_nMMd

def dMMr_dt(nOC, nOB, nMMd, nMMr, N, cOC, cOB, cMMd, cMMr, gr_MMr, dr_MMr, matrix):
    """
    Function that calculates the change in the number of the MMr.

    Parameters:
    -----------
    nOC: Float
        Number of OCs.
    nOB: Float
        Number of OBs.
    nMMr: Float
        Number of the MMr.
    nMMd: Float
        Number of the MMd.
    N: Int
        Number of individuals within the interaction range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter MMd.
    cMMr: Float
        Cost parameter MMr.
    gr_MMr: Float
        Growth rate of the MMr.
    dr_MMr: Float
        Decay rate of the MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    change_MMr: Float
        Change in the number of MMr.

    Example:
    -----------
    >>> dMMr_dt(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
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

    # Calculate the change in the number of MMr
    change_MMr = (m*nOC*cOC + n*nOB*cOB + o*nMMd*cMMd + p* nMMr*cMMr)*(N - 1)/N - cMMr
    return change_MMr


def model_dynamics(y, t, N, cOC, cOB, cMMd, cMMr, growth_rates, decay_rates, matrix, WMMd_inhibitor = 0):
    """Function that determines the number dynamics in a population over time.

    Parameters:
    -----------
    y: List
        List with the values of nOC, nOB, nMMd and nMMr.
    t: Numpy.ndarray
        Array with all the time points.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMd: Float
        Cost parameter MMd.
    cMMr: Float
        Cost parameter MMr.
    growth_rates: List
        List with the growth rate values of the OCs, OBs, MMd and MMr.
    decay_rates: List
        List with the decay rate values of OCs, OBs, MMd and MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd.

    Returns:
    --------
    [nOC_change, nOB_change, nMMd_change, nMMr_change]: List
        List containing the changes in nOC, nOB, nMMd and nMMr.

    Example:
    -----------
    >>> model_dynamics([0.4, 0.2, 0.3, 0.1], 1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]))
    [0.030275999999999983, -0.010762000000000006, 0.0073170000000000145, -0.026830999999999994]
    """
    nOC, nOB, nMMd, nMMr = y

    # Determine the change values
    nOC_change = dOC_dt(nOC, nOB, nMMd, nMMr, N, cOC, cOB, cMMd, cMMr, growth_rates[0], decay_rates[0], matrix)
    nOB_change = dOB_dt(nOC, nOB, nMMd, nMMr,  N, cOC, cOB, cMMd, cMMr, growth_rates[1], decay_rates[1], matrix)
    nMMd_change = dMMd_dt(nOC, nOB, nMMd, nMMr, N, cOC, cOB, cMMd, cMMr, growth_rates[2], decay_rates[2], matrix, WMMd_inhibitor)
    nMMr_change = dMMr_dt(nOC, nOB, nMMd, nMMr, N, cOC, cOB, cMMd, cMMr, growth_rates[3], decay_rates[3], matrix)

    # Make floats of the arrays
    nOC_change = float(nOC_change)
    nOB_change = float(nOB_change)
    nMMd_change = float(nMMd_change)
    nMMr_change = float(nMMr_change)

    return [nOC_change, nOB_change, nMMd_change, nMMr_change]


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


def switch_dataframe(n_switches, t_steps_drug, t_steps_no_drug, nOC, nOB, nMMd, nMMr,
    N, cOC, cOB, cMMd, cMMr, growth_rates, decay_rates, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time for a given time of drug holiday and administration periods.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: Int
        The number of generations drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OCs.
    nOB: Float
        Number of OBs.
    nMMr: Float
        Number of the MMr.
    nMMd: Float
        Number of the MMd.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: Float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter MMr.
    cMMd: Float
        Cost parameter MMd.
    growth_rates: List
        List with the growth rate values of the OCs, OBs, MMd and MMr.
    decay_rates: List
        List with the decay rate values of OCs, OBs, MMd and MMr.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are administrated.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 15
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                    'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

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
            parameters = (N, cOC, cOB, cMMd, cMMr, growth_rates, decay_rates, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe tot total dataframe
            df_total_switch = pd.concat([df_total_switch, df])
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 1
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
            parameters = (N, cOC, cOB, cMMd, cMMr, growth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

            # Add dataframe tot total dataframe
            df_total_switch = pd.concat([df_total_switch, df])
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 0
            time += t_steps_no_drug

    return df_total_switch


def pronto_switch_dataframe(n_switches, t_steps_drug, t_steps_no_drug, nOC, nOB,
                            nMMd, nMMr, N, cOC, cOB, cMMd, cMMr, growth_rates, decay_rates, matrix_no_GF_IH,
                            matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
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
    nOC: Float
        Number of OCs.
    nOB: Float
        Number of OBs.
    nMMr: Float
        Number of the MMr.
    nMMd: Float
        Number of the MMd.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter MMr.
    cMMd: Float
        Cost parameter MMd.
    growth_rates: List
        List with the growth rate values of the OCs, OBs, MMd and MMr.
    decay_rates: List
        List with the decay rate values of OCs, OBs, MMd and MMr.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are administrated.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    t = np.linspace(0, t_steps_no_drug, t_steps_no_drug*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                    'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total xMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps_no_drug

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
            parameters = (N, cOC, cOB, cMMd, cMMr, rowth_rates, decay_rates, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total xMM': y[:, 3]+ y[:, 2]})

            # Add dataframe tot total dataframe
            df_total_switch = pd.concat([df_total_switch, df])
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 1
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
            parameters = (N, cOC, cOB, cMMd, cMMr, rowth_rates, decay_rates, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total xMM': y[:, 3]+ y[:, 2]})

            # Add dataframe tot total dataframe
            df_total_switch = pd.concat([df_total_switch, df])
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 0
            time += t_steps_no_drug

    return df_total_switch


def mimimal_tumour_num_t_steps(t_steps_drug, t_steps_no_drug, nOC, nOB, nMMd, nMMr,
        N, cOC, cOB, cMMd, cMMr, growth_rates, decay_rates, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time for a given time of a drug holiday.

    Parameters:
    -----------
    t_steps_drug: Int
        The number of generations drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OCs.
    nOB: Float
        Number of OBs.
    nMMr: Float
        Number of the MMr.
    nMMd: Float
        Number of the MMd.
    N: Int
        Number of cells in the difussion range.
    cOC: Float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMMr: Float
        Cost parameter MMr.
    cMMd: Float
        Cost parameter MMd.
    growth_rates: List
        List with the growth rate values of the OCs, OBs, MMd and MMr.
    decay_rates: List
        List with the decay rate values of OCs, OBs, MMd and MMr.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are administrated.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd.

    Returns:
    --------
    average_MM_number: float
        The average total MM number in the last period.

    Example:
    -----------
    average_MM_fractions: float
        The average total MM fraction in the last period.

    >>> matrix_no_GF_IH = np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]])
    >>> matrix_no_GF_IH - np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [0.8, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]])
    >>> mimimal_tumour_num_t_steps(5, 5, 0.2, 0.3, 0.2, 0.3, 10, 0.3, 0.2,
    ...                               0.3, 0.5, matrix_no_GF_IH, matrix_no_GF_IH)
    0.5624999973582969
    """
    # Deteremine the number of switches
    time_step = (t_steps_drug + t_steps_no_drug) / 2
    n_switches = int((110 // time_step) -1)

    # Create a dataframe of the numbers
    df = switch_dataframe(n_switches, t_steps_drug, t_steps_no_drug, nOC, nOB,
                                 nMMd, nMMr, N, cOC, cOB, cMMd, cMMr, growth_rates, decay_rates,
                                 matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine the average MM number in the last period with and without drugs
    last_MM_numbers = df['total nMM'].tail(int(time_step *2))
    average_MM_number = last_MM_numbers.sum() / (int(time_step*2))

    return float(average_MM_number)
