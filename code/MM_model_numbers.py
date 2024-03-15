"""
Author:       Eva Nieuwenhuis
University:   UvA
Student id':  13717405
Description:  Code of the model that simulates the dynamics in the multiple myeloma
              (MM) microenvironment with four cell types: drug-sensitive MM cells
              (MMd), resistant MM cells (MMr), osteoblasts (OB) and osteoclasts
              (OC). The model is a public goods game in the framework of
              evolutionary game theory with collective interactions. In this model
              there is looked at the numbers of the four cell types.
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
Example interaction matrix:
M = np.array([
       Goc Gob Gmmd Gmmr
    OC  [a,  b,  c,  d],
    OB  [e,  f,  g,  h],
    MMd [i,  j,  k,  l],
    MMr [m,  n,  o,  p]])
"""

def main():
    # Do doc tests
    doctest.testmod()

    # Make a figure showing the cell number dynamics by traditional therapy and
    # by adaptive therapy (original situation)
    list_t_steps_drug = [10, 10, 10]
    Figure_continuous_MTD_vs_AT(20, list_t_steps_drug)

    # Make a figure showing the cell fraction dynamics by traditional therapy and
    # by adaptive therapy for shorter holiday and administration periods compared
    # to the original situation
    list_t_steps_drug = [4, 4, 4]
    Figure_continuous_MTD_vs_AT_short_a_h(50, list_t_steps_drug)

    # Make a figure showing the cell fraction dynamics by traditional therapy and
    # by adaptive therapy for weaker IHs compared to the original situation
    list_t_steps_drug = [10, 10, 10]
    Figure_continuous_MTD_vs_AT_weak_a_h(20, list_t_steps_drug)

    # Make a 3D figure showthing the effect of different drug holiday and
    # administration periods
    Figure_3D_MM_numb_IH_add_and_holiday()

    # Make a figure that shows the MM number for different bOC,MMd values
    Figure_best_b_OC_MMd()

    # Make a figure that shows the MM number for different WMMd IH values
    Figure_best_WMMd_IH()

    # Make a 3D figure showing the effect of different WMMd and MMd GF IH strengths
    Figure_3D_MM_numb_MMd_IH_strength()

    # Make line plots showing the dynamics when the IH administration is longer
    # than the holiday and one it is the other way around.
    list_t_steps_drug = [3, 10]
    list_t_steps_no_drug = [10, 3]
    list_n_steps = [40, 40]
    Figure_duration_A_h_MMd_IH(list_n_steps, list_t_steps_drug,
                                                            list_t_steps_no_drug)

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
    nMMr: Float
        Number of the MMr.
    nMMd: Float
        Number of the MMd.
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


def switch_dataframe(n_switches, t_steps_drug, t_steps_no_drug, nOC, nOB, nMMd,
                                nMMr, growth_rates, decay_rates, matrix_no_GF_IH,
                                matrix_GF_IH, WMMd_inhibitor = 0):
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
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
                                                                    administrated.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administrated.
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
    t_steps = 60
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

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
            parameters = (growth_rates, decay_rates, matrix, WMMd_inhibitor)

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
            parameters = (growth_rates, decay_rates, matrix)

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

def continuous_add_IH_df(end_generation, nOC, nOB, nMMd, nMMr, growth_rates,
                decay_rates, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the cell type numbers when the IHs
    are administered continuously.

    Parameters:
    -----------
    end_generation: Int
        The last generation for which the fractions have to be calculated
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
        The dataframe with the cell numbers when IHs are continiously administerd.
    """
    t = np.linspace(0, 60, 60)
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

    t = np.linspace(60, end_generation, 200)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_GF_IH, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Combine the dataframes
    df_total = pd.concat([df_1, df_2])

    return df_total

def minimal_tumour_numb_t_steps(t_steps_drug, t_steps_no_drug, nOC, nOB, nMMd,
                                nMMr, growth_rates, decay_rates, matrix_no_GF_IH,
                                matrix_GF_IH, WMMd_inhibitor = 0):
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
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
                                                                    administrated.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    average_MM_number: Float
        The average total MM number in the last period.
    """
    # Deteremine the number of switches
    time_step = (t_steps_drug + t_steps_no_drug) / 2
    n_switches = int((400 // time_step) -1)

    # Create a dataframe of the numbers
    df = switch_dataframe(n_switches, t_steps_drug, t_steps_no_drug, nOC, nOB,
                                nMMd, nMMr, growth_rates, decay_rates,
                                matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)


    # Determine the average MM number in the last period with and without drugs
    last_MM_numbers = df['total nMM'].tail(int(time_step *2))
    average_MM_number = last_MM_numbers.sum() / (int(time_step*2))

    return float(average_MM_number)

def x_y_z_axis_values_3d_plot(dataframe, name):
    """ Function that determines the x, y and z axis values from the given
    dataframe. It also prints the administration and holiday duration leading
    to the lowest total MM number in the equilibrium

    Parameters:
    -----------
    Dataframe: dataFrame
        The dataframe with the generated data
    name: String
        The name of the administerd IH(s)

    Returns:
    --------
    X_values: Numpy.ndarray
        Array with the values for the x-axis
    Y_values: Numpy.ndarray
        Array with the values for the y-axis
    Z_values: Numpy.ndarray
        Array with the values for the z-axis
    """

    # Find the drug administration and holiday period causing the lowest MM fraction
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

    # Fill the 2D array with the MM fraction values by looping over each row
    for index, row in dataframe.iterrows():
        i = int(row.iloc[0]) - 2
        j = int(row.iloc[1]) - 2
        Z_values[j, i] = row.iloc[2]

    return (X_values, Y_values, Z_values)

def minimal_tumour_numb_b_OC_MMd(b_OC_MMd, nOC, nOB, nMMd, nMMr, growth_rates,
                                        decay_rates, matrix, b_OC_MMd_array):
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
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    b_OC_MMd_array: Float
        If True b_OC_MMd is an array and if False b_OC_MMd is a float.

    Returns:
    --------
    last_MM_number: float
        The average total MM number in the last period.

    Example:
    -----------
    >>> matrix = np.array([
    ...    [0.0, 0.4, 0.6, 0.5],
    ...    [0.3, 0.0, -0.3, -0.3],
    ...    [0.6, 0.0, 0.2, 0.0],
    ...    [0.55, 0.0, -0.6, 0.4]])
    >>> minimal_tumour_numb_b_OC_MMd(0.4, 20, 30, 20, 5,[0.8, 1.2, 0.3, 0.3],
    ...                                     [0.9, 0.08, 0.2, 0.1], matrix, False)
    26.318660931596057
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
    parameters = (growth_rates, decay_rates, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t_over, args=parameters)
    df_2 = pd.DataFrame({'Generation': t_over, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Determine the total MM number
    last_MM_number = df_2['total nMM'].iloc[-1]

    return float(last_MM_number)

def minimal_tumour_numb_WMMd_IH(WMMd_inhibitor, nOC, nOB, nMMd, nMMr,
                    growth_rates, decay_rates, matrix, WMMd_inhibitor_array):
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
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
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
        The average total MM number in the last period.

    >>> matrix = np.array([
    ...    [0.0, 0.4, 0.6, 0.5],
    ...    [0.3, 0.0, -0.3, -0.3],
    ...    [0.6, 0.0, 0.2, 0.0],
    ...    [0.55, 0.0, -0.6, 0.4]])
    >>> minimal_tumour_numb_WMMd_IH(0.3, 20, 30, 20, 5, [0.8, 1.2, 0.3, 0.3],
    ...                                 [0.9, 0.08, 0.2, 0.1], matrix, False)
    28.440744908435057
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
    parameters = (growth_rates, decay_rates, matrix, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t_over, args=parameters)
    df_2 = pd.DataFrame({'Generation': t_over, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Determine the total MM number
    last_MM_number = df_2['total nMM'].iloc[-1]

    return float(last_MM_number)


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

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.58, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.175, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.3, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.9, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.24

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.55

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_GF = switch_dataframe(n_switches, t_steps_drug[0],
                        t_steps_drug[0], nOC, nOB, nMMd, nMMr, growth_rates,
                        decay_rates, matrix_no_GF_IH, matrix_GF_IH)
    df_total_switch_WMMd = switch_dataframe(n_switches, t_steps_drug[1],
                t_steps_drug[1], nOC, nOB, nMMd, nMMr, growth_rates, decay_rates,
                matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_switch_comb = switch_dataframe(n_switches, t_steps_drug[2],
                t_steps_drug[2], nOC, nOB, nMMd, nMMr, growth_rates, decay_rates,
                matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)

    # Make dataframes for continiously administration
    df_total_GF = continuous_add_IH_df(260, nOC, nOB, nMMd, nMMr, growth_rates,
            decay_rates, matrix_no_GF_IH, matrix_GF_IH)
    df_total_WMMd = continuous_add_IH_df(260, nOC, nOB, nMMd, nMMr, growth_rates,
            decay_rates, matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_comb = continuous_add_IH_df(260, nOC, nOB, nMMd, nMMr, growth_rates,
            decay_rates, matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)


    # Save the data
    save_dataframe(df_total_switch_GF, 'df_cell_numb_switch_GF_IH.csv',
                                            r'..\data\data_model_numbers')
    save_dataframe(df_total_switch_WMMd, 'df_cell_numb_switch_WMMd_IH.csv',
                                            r'..\data\data_model_numbers')
    save_dataframe(df_total_switch_comb, 'df_cell_numb_switch_comb_IH.csv',
                                            r'..\data\data_model_numbers')
    save_dataframe(df_total_GF, 'df_cell_numb_continuous_GF_IH.csv',
                                             r'..\data\data_model_numbers')
    save_dataframe(df_total_WMMd, 'df_cell_numb_continuous_WMMd_IH.csv',
                                             r'..\data\data_model_numbers')
    save_dataframe(df_total_comb, 'df_cell_numb_continuous_comb_IH.csv',
                                             r'..\data\data_model_numbers')

    # Create a Figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))

    # Plot the data without drug holidays in the first plot
    df_total_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 0])
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel('Number', fontsize=11)
    axs[0, 0].set_title(f"Continuous MTD MMd GF IH ")
    axs[0, 0].grid(True)

    # Plot the data with drug holidays in the second plot
    df_total_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 1])
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel(' ')
    axs[0, 1].set_title(r"Continuous MTD $W_{MMd}$ IH")
    axs[0, 1].grid(True)

    # Plot the data with drug holidays in the second plot
    df_total_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 2])
    axs[0, 2].set_xlabel(' ')
    axs[0, 2].set_ylabel(' ')
    axs[0, 2].set_title(r"Continuous MTD MMd GF IH and $W_{MMd}$ IH")
    axs[0, 2].grid(True)

    # Plot the data with drug holidays in the third plot
    df_total_switch_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations', fontsize=11)
    axs[1, 0].set_ylabel('Number', fontsize=11)
    axs[1, 0].set_title(f"Adaptive therapy MMd GF IH")
    axs[1, 0].grid(True)
    plt.grid(True)

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 1])
    axs[1, 1].set_xlabel('Generations', fontsize=11)
    axs[1, 1].set_ylabel(' ')
    axs[1, 1].set_title(r"Adaptive therapy $W_{MMd}$ IH")
    axs[1, 1].grid(True)

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 2])
    axs[1, 2].set_xlabel('Generations', fontsize=11)
    axs[1, 2].set_ylabel(' ')
    axs[1, 2].set_title(r"Adaptive therapy MMd GF IH and $W_{MMd}$ IH")
    axs[1, 2].grid(True)
    save_Figure(plt, 'line_plot_cell_numb_AT_MTD',
                                 r'..\visualisation\results_model_numbers')

    # Create a single legend outside of all plots
    legend_labels = ['Number of OC', 'Number of OB', 'Number of MMd',
                                                                'Number of MMr']
    fig.legend(labels = legend_labels, loc='upper center', ncol=4, fontsize='large')
    plt.show()

""" Figure to determine the difference between traditional and adaptive therapy.
Shorter holiday and administration periods compared to the original situation"""
def Figure_continuous_MTD_vs_AT_short_a_h(n_switches, t_steps_drug):
    """ Function that makes a figure with 6 subplots showing the cell number
    dynamics by traditional therapy (continuous MTD) and adaptive therapy. The
    holiday and administration periods are short (4 generations).

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

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.58, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.175, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.3, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.9, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.24

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.55

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_GF = switch_dataframe(n_switches, t_steps_drug[0],
                        t_steps_drug[0], nOC, nOB, nMMd, nMMr, growth_rates,
                        decay_rates, matrix_no_GF_IH, matrix_GF_IH)
    df_total_switch_WMMd = switch_dataframe(n_switches, t_steps_drug[1],
                t_steps_drug[1], nOC, nOB, nMMd, nMMr, growth_rates, decay_rates,
                matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_switch_comb = switch_dataframe(n_switches, t_steps_drug[2],
                t_steps_drug[2], nOC, nOB, nMMd, nMMr, growth_rates, decay_rates,
                matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)

    # Make dataframes for continiously administration
    df_total_GF = continuous_add_IH_df(260, nOC, nOB, nMMd, nMMr, growth_rates,
            decay_rates, matrix_no_GF_IH, matrix_GF_IH)
    df_total_WMMd = continuous_add_IH_df(260, nOC, nOB, nMMd, nMMr, growth_rates,
            decay_rates, matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_comb = continuous_add_IH_df(260, nOC, nOB, nMMd, nMMr, growth_rates,
            decay_rates, matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)

    # Save the data
    save_dataframe(df_total_switch_GF, 'df_cell_numb_switch_GF_IH_short_a_h.csv',
                                            r'..\data\data_model_numbers')
    save_dataframe(df_total_switch_WMMd, 'df_cell_numb_switch_WMMd_IH_short_a_h.csv',
                                            r'..\data\data_model_numbers')
    save_dataframe(df_total_switch_comb, 'df_cell_numb_switch_comb_IH_short_a_h.csv',
                                            r'..\data\data_model_numbers')
    save_dataframe(df_total_GF, 'df_cell_numb_continuous_GF_IH_short_a_h.csv',
                                             r'..\data\data_model_numbers')
    save_dataframe(df_total_WMMd, 'df_cell_numb_continuous_WMMd_IH_short_a_h.csv',
                                             r'..\data\data_model_numbers')
    save_dataframe(df_total_comb, 'df_cell_numb_continuous_comb_IH_short_a_h.csv',
                                             r'..\data\data_model_numbers')

    # Create a Figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))

    # Plot the data without drug holidays in the first plot
    df_total_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 0])
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel('Number', fontsize=11)
    axs[0, 0].set_title(f"Continuous MTD MMd GF IH ")
    axs[0, 0].grid(True)

    # Plot the data with drug holidays in the second plot
    df_total_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 1])
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel(' ')
    axs[0, 1].set_title(r"Continuous MTD $W_{MMd}$ IH")
    axs[0, 1].grid(True)

    # Plot the data with drug holidays in the second plot
    df_total_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 2])
    axs[0, 2].set_xlabel(' ')
    axs[0, 2].set_ylabel(' ')
    axs[0, 2].set_title(r"Continuous MTD MMd GF IH and $W_{MMd}$ IH")
    axs[0, 2].grid(True)

    # Plot the data with drug holidays in the third plot
    df_total_switch_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations', fontsize=11)
    axs[1, 0].set_ylabel('Number', fontsize=11)
    axs[1, 0].set_title(f"Adaptive therapy MMd GF IH")
    axs[1, 0].grid(True)
    plt.grid(True)

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 1])
    axs[1, 1].set_xlabel('Generations', fontsize=11)
    axs[1, 1].set_ylabel(' ')
    axs[1, 1].set_title(r"Adaptive therapy $W_{MMd}$ IH")
    axs[1, 1].grid(True)

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 2])
    axs[1, 2].set_xlabel('Generations', fontsize=11)
    axs[1, 2].set_ylabel(' ')
    axs[1, 2].set_title(r"Adaptive therapy MMd GF IH and $W_{MMd}$ IH")
    axs[1, 2].grid(True)
    save_Figure(plt, 'line_plot_cell_numb_AT_MTD_short_a_h',
                                 r'..\visualisation\results_model_numbers')

    # Create a single legend outside of all plots
    legend_labels = ['Number of OC', 'Number of OB', 'Number of MMd',
                                                                'Number of MMr']
    fig.legend(labels = legend_labels, loc='upper center', ncol=4, fontsize='large')
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

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.58, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.25, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.32, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.9, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.22

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.42

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_GF = switch_dataframe(n_switches, t_steps_drug[0],
                        t_steps_drug[0], nOC, nOB, nMMd, nMMr, growth_rates,
                        decay_rates, matrix_no_GF_IH, matrix_GF_IH)
    df_total_switch_WMMd = switch_dataframe(n_switches, t_steps_drug[1],
                t_steps_drug[1], nOC, nOB, nMMd, nMMr, growth_rates, decay_rates,
                matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_switch_comb = switch_dataframe(n_switches, t_steps_drug[2],
                t_steps_drug[2], nOC, nOB, nMMd, nMMr, growth_rates, decay_rates,
                matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)

    # Make dataframes for continiously administration
    df_total_GF = continuous_add_IH_df(260, nOC, nOB, nMMd, nMMr, growth_rates,
            decay_rates, matrix_no_GF_IH, matrix_GF_IH)
    df_total_WMMd = continuous_add_IH_df(260, nOC, nOB, nMMd, nMMr, growth_rates,
            decay_rates, matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_comb = continuous_add_IH_df(260, nOC, nOB, nMMd, nMMr, growth_rates,
            decay_rates, matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)

    # Save the data
    save_dataframe(df_total_switch_GF, 'df_cell_numb_switch_GF_IH_weak_a_h.csv',
                                            r'..\data\data_model_numbers')
    save_dataframe(df_total_switch_WMMd, 'df_cell_numb_switch_WMMd_IH_weak_a_h.csv',
                                            r'..\data\data_model_numbers')
    save_dataframe(df_total_switch_comb, 'df_cell_numb_switch_comb_IH_weak_a_h.csv',
                                            r'..\data\data_model_numbers')
    save_dataframe(df_total_GF, 'df_cell_numb_continuous_GF_IH_weak_a_h.csv',
                                             r'..\data\data_model_numbers')
    save_dataframe(df_total_WMMd, 'df_cell_numb_continuous_WMMd_IH_weak_a_h.csv',
                                             r'..\data\data_model_numbers')
    save_dataframe(df_total_comb, 'df_cell_numb_continuous_comb_IH_weak_a_h.csv',
                                             r'..\data\data_model_numbers')

    # Create a Figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))

    # Plot the data without drug holidays in the first plot
    df_total_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 0])
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel('Number', fontsize=11)
    axs[0, 0].set_title(f"Continuous MTD MMd GF IH ")
    axs[0, 0].grid(True)

    # Plot the data with drug holidays in the second plot
    df_total_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 1])
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel(' ')
    axs[0, 1].set_title(r"Continuous MTD $W_{MMd}$ IH")
    axs[0, 1].grid(True)

    # Plot the data with drug holidays in the second plot
    df_total_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[0, 2])
    axs[0, 2].set_xlabel(' ')
    axs[0, 2].set_ylabel(' ')
    axs[0, 2].set_title(r"Continuous MTD MMd GF IH and $W_{MMd}$ IH")
    axs[0, 2].grid(True)

    # Plot the data with drug holidays in the third plot
    df_total_switch_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 0])
    axs[1, 0].set_xlabel('Generations', fontsize=11)
    axs[1, 0].set_ylabel('Number', fontsize=11)
    axs[1, 0].set_title(f"Adaptive therapy MMd GF IH")
    axs[1, 0].grid(True)
    plt.grid(True)

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 1])
    axs[1, 1].set_xlabel('Generations', fontsize=11)
    axs[1, 1].set_ylabel(' ')
    axs[1, 1].set_title(r"Adaptive therapy $W_{MMd}$ IH")
    axs[1, 1].grid(True)

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                                                    legend=False, ax=axs[1, 2])
    axs[1, 2].set_xlabel('Generations', fontsize=11)
    axs[1, 2].set_ylabel(' ')
    axs[1, 2].set_title(r"Adaptive therapy MMd GF IH and $W_{MMd}$ IH")
    axs[1, 2].grid(True)
    save_Figure(plt, 'line_plot_cell_numb_AT_MTD_weak_a_h',
                                 r'..\visualisation\results_model_numbers')

    # Create a single legend outside of all plots
    legend_labels = ['Number of OC', 'Number of OB', 'Number of MMd',
                                                                'Number of MMr']
    fig.legend(labels = legend_labels, loc='upper center', ncol=4, fontsize='large')
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

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.58, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.35, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_comb = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.3

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.38

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM number']
    df_holiday_GF_IH = pd.DataFrame(columns=column_names)

    # Loop over all the t_step values for drug administration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            numb_tumour = minimal_tumour_numb_t_steps(t_steps_drug,
                            t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                            decay_rates, matrix_no_GF_IH, matrix_GF_IH)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug':
                    int(t_steps_no_drug), 'Generations drug': int(t_steps_drug),
                                             'MM number': float(numb_tumour)}])
            df_holiday_GF_IH = pd.concat([df_holiday_GF_IH, new_row_df],
                                                            ignore_index=True)

    # Save the data
    save_dataframe(df_holiday_GF_IH, 'df_cell_numb_best_MMd_GH_IH_holiday.csv',
                                             r'..\data\data_model_numbers')

    # Determine the axis values
    X_GF_IH, Y_GF_IH, Z_GF_IH = x_y_z_axis_values_3d_plot(df_holiday_GF_IH,
                                                                        'GF IH')

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM number']
    df_holiday_W_IH = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            numb_tumour = minimal_tumour_numb_t_steps(t_steps_drug,
                    t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                    decay_rates,matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': int(t_steps_no_drug),
                                            'Generations drug': int(t_steps_drug),
                                             'MM number': float(numb_tumour)}])
            df_holiday_W_IH = pd.concat([df_holiday_W_IH, new_row_df],
                                                                ignore_index=True)

    # Save the data
    save_dataframe(df_holiday_W_IH, 'df_cell_numb_best_WMMd_IH_holiday.csv',
                                             r'..\data\data_model_numbers')

    # Determine the axis values
    X_W_IH, Y_W_IH, Z_W_IH = x_y_z_axis_values_3d_plot(df_holiday_W_IH, 'W IH')

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM number']
    df_holiday_comb = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            numb_tumour = minimal_tumour_numb_t_steps(t_steps_drug,
                t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, decay_rates,
                matrix_no_GF_IH, matrix_GF_IH_comb, WMMd_inhibitor_comb)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': int(t_steps_no_drug),
                                            'Generations drug': int(t_steps_drug),
                                            'MM number': float(numb_tumour)}])
            df_holiday_comb = pd.concat([df_holiday_comb, new_row_df],
                                                                ignore_index=True)

    # Save the data
    save_dataframe(df_holiday_comb, 'df_cell_numb_best_MMd_IH_holiday.csv',
                                             r'..\data\data_model_numbers')

    # Determine the axis values
    X_comb, Y_comb, Z_comb = x_y_z_axis_values_3d_plot(df_holiday_comb,
                                                            'IH comnbination')

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
            ax.set_zlabel('Number of MM')
            ax.set_title(r'A) $W_{MMd}$ IH', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 35, azim = -112)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('Number of MM')

        elif i == 2:
            surf = ax.plot_surface(X_GF_IH, Y_GF_IH, Z_GF_IH, cmap = 'coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IH')
            ax.set_ylabel('Generations IH')
            ax.set_zlabel('Number of MM')
            ax.set_title('B)  MMd GF IH', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 32, azim = -164)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')

            color_bar.set_label('Number of MM')

        elif i == 3:
            surf = ax.plot_surface(X_comb, Y_comb, Z_comb, cmap = 'coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IHs')
            ax.set_ylabel('Generations IHs')
            ax.set_zlabel('Number of MM')
            ax.set_title('C)  $W_{MMd}$ IH and MMd GF IH', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 39, azim = -121)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('Number of MM')

        else:
            # Hide the emply subplot
            ax.axis('off')

    # Add a color bar
    save_Figure(fig, '3d_plot_MM_numb_best_IH_h_a_periods',
                                r'..\visualisation\results_model_numbers')
    plt.show()


""" 3D plot showing the best IH strengths """
def Figure_3D_MM_numb_MMd_IH_strength():
    """ 3D plot that shows the average MM number for different MMd GF inhibitor
    and WMMd inhibitor strengths. It prints the IH streghts that caused the lowest
    total MM number."""

    # Set initial parameter values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.60, 0.50],
        [0.3, 0.0, -0.3, -0.3],
        [0.65, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.5],
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
            numb_tumour = minimal_tumour_numb_t_steps(t_steps_drug, t_steps_no_drug,
                                nOC, nOB, nMMd, nMMr, growth_rates, decay_rates,
                                matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Strength WMMd IH':\
                        round(strength_WMMd_IH/ 50, 3), 'Strength MMd GF IH': \
                round(strength_MMd_GF_IH/ 50, 3), 'MM number': numb_tumour}])

            df_holiday = pd.concat([df_holiday, new_row_df], ignore_index=True)

    # Save the data
    save_dataframe(df_holiday, 'df_cell_numb_best_MMd_IH_strength.csv',
                                             r'..\data\data_model_numbers')


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
    ax.set_title(r"""Average MM fraction with varying $W_{MMd}$ IH and
    MMd GF IH strengths""")

    # Turn to the right angle
    ax.view_init(elev = 40, azim = -134)

    # Add a color bar
    color_bar = fig.colorbar(surf, shrink = 0.6, location= 'left')
    color_bar.set_label('Number of MM')

    save_Figure(fig, '3d_plot_MM_numb_best_IH_strength',
                                r'..\visualisation\results_model_numbers')
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

    # Payoff matrix
    matrix = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.65, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    WMMd_IH_start = 0.2

    # Perform the optimization
    result = minimize(minimal_tumour_numb_WMMd_IH, WMMd_IH_start, args = (nOC,
                            nOB, nMMd, nMMr, growth_rates, decay_rates, matrix,
                            True), bounds=[(0, 0.8)], method='Nelder-Mead')

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
                                nMMr, growth_rates, decay_rates, matrix, False)
        dict_numb_tumour[WMMd_inhibitor] = numb_tumour

    # Save the data
    save_dictionary(dict_numb_tumour,
            r'..\data\data_model_numbers\dict_cell_numb_WMMd_IH.csv')

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
    save_Figure(plt, 'line_plot_cell_numb_change_WMMd_IH',
                                 r'..\visualisation\results_model_numbers')
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

    # Payoff matrix
    matrix = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.65, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    b_OC_MMd_start = 0.45

    # Perform the optimization
    result = minimize(minimal_tumour_numb_b_OC_MMd, b_OC_MMd_start, args = (nOC,
                            nOB, nMMd, nMMr, growth_rates, decay_rates, matrix,
                            True), bounds=[(0, 0.8)],  method='Nelder-Mead')

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
                                    growth_rates, decay_rates, matrix, False)
        dict_numb_tumour_GF[b_OC_MMd] = numb_tumour

    # Save the data
    save_dictionary(dict_numb_tumour_GF,
                 r'..\data\data_model_numbers\dict_cell_numb_b_OC_MMd.csv')

    # Make a list of the keys and one of the values
    b_OC_MMd_values = list(dict_numb_tumour_GF.keys())
    MM_numbers = list(dict_numb_tumour_GF.values())

    # Create the plot
    plt.plot(b_OC_MMd_values, MM_numbers, linestyle='-')
    plt.xlabel(r'$b_{OC, MMd}$ value ')
    plt.ylabel(r'Number of MM')
    plt.title(r'MM number for different $b_{OC, MMd}$ values')
    plt.grid(True)
    save_Figure(plt, 'line_plot_cell_numb_change_b_OC_MMd',
                                r'..\visualisation\results_model_numbers')
    plt.show()


""" Figure with a longer IH administration than holiday and the other way around"""
def Figure_duration_A_h_MMd_IH(n_switches, t_steps_drug, t_steps_no_drug):
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

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_GF_IH_half = np.array([
        [0.0, 0.4, 0.6, 0.5],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_half = 0.25

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_1 = switch_dataframe(n_switches[0], t_steps_drug[0],
            t_steps_no_drug[0], nOC, nOB, nMMd, nMMr, growth_rates, decay_rates,
            matrix_no_GF_IH, matrix_GF_IH_half, WMMd_inhibitor_half)
    df_total_switch_2 = switch_dataframe(n_switches[1], t_steps_drug[1],
            t_steps_no_drug[1], nOC, nOB, nMMd, nMMr, growth_rates, decay_rates,
            matrix_no_GF_IH, matrix_GF_IH_half, WMMd_inhibitor_half)

    # Save the data
    save_dataframe(df_total_switch_1, 'df_cell_numb_short_a_long_h_MMd_IH.csv',
                                             r'..\data\data_model_numbers')
    save_dataframe(df_total_switch_2, 'df_cell_numb_long_a_short_h_MMd_IH.csv.csv',
                                             r'..\data\data_model_numbers')

    # Create a Figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    g = 'generations'
    ta = t_steps_drug
    th = t_steps_no_drug

    # Plot the data with drug holidays in the second plot
    df_total_switch_1.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
        label=['Number OC', 'Number OB', 'Number MMd', 'Number MMr'], ax=axs[0])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Number of MM')
    axs[0].set_title(f"""Dynamics when the IH administrations lasted {ta[0]} {g}
    and the IH holidays lasted {th[0]} {g}""")
    axs[0].legend(loc = 'upper right')
    axs[0].grid(True)

    # Plot the data with drug holidays in the third plot
    df_total_switch_2.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
        label=['Number OC', 'Number OB', 'Number MMd','Number MMr'], ax=axs[1])
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Number of MM')
    axs[1].set_title(f"""Dynamics when the IH administrations lasted {ta[1]} {g}
    and the IH holidays lasted {th[1]} {g}""")
    axs[1].legend(loc = 'upper right')
    axs[1].grid(True)
    plt.grid(True)
    save_Figure(plt, 'line_plot_cell_numb_diff_h_and_a_MMd_IH',
                                 r'..\visualisation\results_model_numbers')
    plt.show()


if __name__ == "__main__":
    main()
