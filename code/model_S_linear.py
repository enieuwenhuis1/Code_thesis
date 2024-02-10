"""
Author:       Eva Nieuwenhuis
University:   UvA
Student id':  13717405
Description:  Code with the model that simulates linear dynamics in the multiple myeloma
              (MM) microenvironment with three cell types: MM cells, osteoblasts (OBs)
              and osteoclasts (OCs). The model is a public goods game in the framework
              of evolutionary game theory with collective interactions. It attempts to
              replicate certain Figures from the paper by Sartakhti et al. (2016).

              When running the code it also shows line plots of the frequencies, these
              are not in the paper and not saved but are shown for a better understanding
              of the dynamics.

Sartakhti, J. S., Manshaei, M. H., Bateni, S., & Archetti, M. (2016). Evolutionary
dynamics of Tumor-Stroma interactions in multiple myeloma. PLOS ONE, 11(12),
e0168856. https://doi.org/10.1371/journal.pone.0168856

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


"""For some plots it not fully clear which parameter values are used. This could be
an explenation for why the dound stable points are not alway the exact same points
as shown in the paper. However the give a formula to calculate the stable points
and using thhe formula, these plots show the coreect stable points for the used
parameter values.
"""

def main():
    # Do doc tests
    import doctest
    doctest.testmod()

    # Make Figure 2 in the paper of Sartakhti et al., 2016.
    Figure_2()

    # Make Figure 3 in the paper of Sartakhti et al., 2016.
    Figure_3()

    # Make Figure 5 in the paper of Sartakhti et al., 2016.
    Figure_5()

    # Make Figure 8A in the paper of Sartakhti et al., 2016.
    Figure_8A()

    # Make Figure 8B in the paper of Sartakhti et al., 2016.
    Figure_8B()

    # Make Figure 9A in the paper of Sartakhti et al., 2016.
    Figure_9A()

    # Make Figure 9B in the paper of Sartakhti et al., 2016.
    Figure_9B()

    # Make Figure 9C in the paper of Sartakhti et al., 2016.
    Figure_9C()

    # Make Figure 10A in the paper of Sartakhti et al., 2016.
    Figure_10A()

    # Make Figure 10B in the paper of Sartakhti et al., 2016.
    Figure_10B()

    # Make Figure 11 in the paper of Sartakhti et al., 2016.
    Figure_11()

    # Make Figure 12A middel in the paper of Sartakhti et al., 2016.
    Figure_12A_middel()

    # Make Figure 12A right in the paper of Sartakhti et al., 2016.
    Figure_12A_right()

    # Make Figure 12B middel in the paper of Sartakhti et al., 2016.
    Figure_12B_middel()

    # Make Figure 12B right in the paper of Sartakhti et al., 2016.
    Figure_12B_right()

    # Make Figure 12C middel in the paper of Sartakhti et al., 2016.
    Figure_12C_middel()

    # Make Figure 12C right in the paper of Sartakhti et al., 2016.
    Figure_12C_right()


def save_data(data_frame, file_name, folder_path):
    """ Function that saves a dataframe as csv file.

    Parameters:
    -----------
    data_frame: DataFrame
        The data frame contain the collected data.
    file_name: String
        The name of the csv file.
    folder_path: String
        Path to the folder where the data will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    data_frame.to_csv(file_path, index=False)

def save_Figure(Figure, file_name, folder_path):
    """Save the Figure to a specific folder.

    Parameters:
    -----------
    Figure: Matplotlib Figure
        Figure object that needs to be saved.
    file_name : String
        The name for the plot.
    folder_path: String:
        Path to the folder where the data will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    Figure.savefig(os.path.join(folder_path, file_name))

def save_ternary(Figure, file_name, folder_path):
    """Save the ternary plot in a specific folder.

    Parameters:
    -----------
    Figure: Matplotlib Figure
        Figure object that needs to be saved.
    file_name : String
        The name for the plot.
    folder_path: String:
        Path to the folder where the data will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    pio.write_image(Figure, os.path.join(folder_path, f'{file_name}.png'),
                                                                format='png')
def fitness_WOC(xOC, xOB, xMM, N, cOC, cOB, cMM, matrix):
    """
    Function that calculates the fitness of an osteoclast in a population.

    Parameters:
    -----------
    xOC: float
        Frequency of OCs.
    xOB: float
        Frequency of OBs.
    xMM: float
        Frequency of the MM cells.
    N: int
        Number of individuals within the interaction range.
    cOC: float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMM: float
        Cost parameter MM cells.
    matrix : numpy.ndarray
        3x3 matrix containing interaction factors.

    Returns:
    --------
    WOC : float
        Fitness of an OC.

    Example:
    -----------
    >>> fitness_WOC(0.5, 0.2, 0.3, 10, 0.3, 0.4, 0.3, np.array([
    ...    [0, 1, 2.5],
    ...    [1, 0, -0.3],
    ...    [2.5, 0, 0]]))
    -0.025499999999999967
    """
    # Extract the needed matrix values
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[0, 2]

    # Calculate the fitness value
    WOC = (c*cMM*xMM + b*cOB*xOB + a* cOC* xOC)*(N - 1)/N - cOC #(18)
    return WOC

def fitness_WOB(xOC, xOB, xMM, N, cOC, cOB, cMM, matrix):
    """
    Function that calculates the fitness of an osteoblast in a population.

    Parameters:
    -----------
    xOC: float
        Frequency of OCs.
    xOB: float
        Frequency of OBs.
    xMM: float
        Frequency of the MM cells.
    N: int
        Number of individuals within the interaction range.
    cOC: float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMM: float
        Cost parameter MM cells.
    matrix : numpy.ndarray
        3x3 matrix containing interaction factors.

    Returns:
    --------
    WOB : float
        Fitness of an OB.

    Example:
    -----------
    >>> fitness_WOB(0.5, 0.2, 0.3,10,  0.3, 0.4, 0.3, np.array([
    ...    [0, 1, 2.5],
    ...    [1, 0, -0.3],
    ...    [2.5, 0, 0]]))
    -0.2893
    """
    # Extract the necessary matrix values
    d = matrix[1, 0]
    e = matrix[1, 1]
    f = matrix[1, 2]

    # Calculate the fitness value
    WOB = (d*cOC*xOC + f*cMM*xMM + e*cOB*xOB)*(N - 1)/N - cOB #(19)
    return WOB

def fitness_WMM(xOC, xOB, xMM, N, cOC, cOB, cMM, matrix):
    """
    Function that calculates the fitness of an MM cell in a population.

    Parameters:
    -----------
    xOC: float
        Frequency of OCs.
    xOB: float
        Frequency of OBs.
    xMM: float
        Frequency of the MM cells.
    N: int
        Number of individuals within the interaction range.
    cOC: float
        Cost parameter OCs.
    cOB: float
        Cost parameter OBs.
    cMM: float
        Cost parameter MM cells.
    matrix : numpy.ndarray
        3x3 matrix containing interaction factors.

    Returns:
    --------
    WOB : float
        Fitness of an MM cell.

    Example:
    -----------
    >>> fitness_WMM(0.5, 0.2, 0.3, 10, 0.3, 0.4, 0.3, np.array([
    ...    [0, 1, 2.5],
    ...    [1, 0, -0.3],
    ...    [2.5, 0, 0]]))
    0.03750000000000003
    """
    # Extract the necessary matrix values
    g = matrix[2, 0]
    h = matrix[2, 1]
    i = matrix[2, 2]

    # Calculate the fitness value
    WMM = (g*cOC*xOC + i*cMM*xMM + h*cOB*xOB)*(N - 1)/N - cMM #(20)
    return WMM

def model_dynamics(y, t, N, c1, c2, c3, matrix):
    """Determines the frequenty dynamics in a population over time.

    Parameters:
    -----------
    y : List
        List with the values of xOC, xOB, and xMM.
    t : Float
        THE time point.
    N : Int
        Number of cells in the difussion range.
    c1 : Float
        Cost value of the OCs.
    c2 : Float
        Cost value of the OBs.
    c3 : Float
        Cost value of the MMs.
    matrix : Numpy array
        Matrix with the payoff values.

    Returns:
    --------
    [xOC_change, xOB_change, xMM_change]: List
        List containing the changes in frequencies of xOC, xOB, and xMM.
    """
    xOC, xOB, xMM = y

    # Determine the fitness values
    WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
    WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
    WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

    # Determine the average fitness
    W_average = xOC * WOC + xOB * WOB + xMM * WMM

    # Determine the new frequencies based on replicator dynamics
    xOC_change = xOC * (WOC - W_average)  # (15)
    xOB_change = xOB * (WOB - W_average)  # (16)
    xMM_change = xMM * (WMM - W_average)  # (17)

    return [xOC_change, xOB_change, xMM_change]

"""
Example payoff matrix:
M = np.array([
       Goc Gob Gmm
    OC [a, b, c],
    OB [d, e, f],
    MM [g, h, i]])
"""

"""Figure 2"""
def Figure_2():
    """Function that makes Figure 2 in the paper of Sartakhti et al., 2016."""
    # Set start parameter values
    xOC = 0.5
    xOB = 0.45
    xMM = 0.05
    N = 100
    c3 = 1.4
    c2 = 1.2
    c1 = 1

    # Payoff matrix
    matrix = np.array([
        [0, 1, 2.5],
        [1, 0, -0.3],
        [2.5, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 30, 60)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_2_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequencies
    xOC = 0.2
    xOB = 0.4
    xMM = 0.4

    # Initial conditions
    t = np.linspace(0, 30, 60)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_2_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_2_first_line, 'df_Figure_2_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_2_second_line, 'df_Figure_2_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make lists
    WOC_list = []
    WOB_list = []
    WMM_list = []
    W_average_list = []
    generation_list = []

    # Iterate over each row
    for index, row in df_Figure_2_first_line.iterrows():
        # Extract values of xOC, xOB, and xMM for the current row
        xOC = row['xOC']
        xOB = row['xOB']
        xMM = row['xMM']

        # Assuming N, c1, c2, c3, and matrix are defined elsewhere
        # Calculate fitness values
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Calculate the average fitness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Append the calculated values to the respective lists
        WOC_list.append(WOC)
        WOB_list.append(WOB)
        WMM_list.append(WMM)
        W_average_list.append(W_average)
        generation_list.append(index)

    # Create a new DataFrame with the calculated values
    df_fitness_first_line = pd.DataFrame({'Generation': generation_list,
    'WOC': WOC_list, 'WOB': WOB_list, 'WMM': WMM_list, 'W_average': W_average_list})

    # Make new lists
    WOC_list = []
    WOB_list = []
    WMM_list = []
    W_average_list = []
    generation_list = []

    # Iterate over each row
    for index, row in df_Figure_2_second_line.iterrows():
        # Extract values of xOC, xOB, and xMM for the current row
        xOC = row['xOC']
        xOB = row['xOB']
        xMM = row['xMM']

        # Assuming N, c1, c2, c3, and matrix are defined elsewhere
        # Calculate fitness values
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Calculate the average fitness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Append the calculated values to the respective lists
        WOC_list.append(WOC)
        WOB_list.append(WOB)
        WMM_list.append(WMM)
        W_average_list.append(W_average)
        generation_list.append(index)

    # Create a new DataFrame with the calculated values
    df_fitness_second_line = pd.DataFrame({'Generation': generation_list,
    'WOC': WOC_list, 'WOB': WOB_list, 'WMM': WMM_list, 'W_average': W_average_list})

    # Create a Figure and axes for subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

    # Plot the first subplot
    df_fitness_first_line.plot(x='Generation', y=['WOC', 'WOB', 'WMM', 'W_average'],
                                                                        ax=axes[0])
    axes[0].set_title('Fitness for a scenario where c2<c1<c3 (Figure 2)')
    axes[0].set_xlabel('Generations')
    axes[0].set_ylabel('Fitness')
    axes[0].legend(['Fitness OC', 'Fitness OB', 'Fitness MM', 'Average fitness'])

    # Plot the second subplot
    df_Figure_2_first_line.plot(x='Generation', y=['xOC', 'xOB', 'xMM'], ax=axes[1])
    axes[1].set_title('Dynamics for a scenario where c2<c1<c3 (Figure 2)')
    axes[1].set_xlabel('Generations')
    axes[1].set_ylabel('Fitness/Frequency')
    axes[1].legend(['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.tight_layout()
    plt.show()

    # Create a Figure and axes for subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

    # Plot the first subplot
    df_fitness_second_line.plot(x='Generation', y=['WOC', 'WOB', 'WMM', 'W_average'],
                                                                        ax=axes[0])
    axes[0].set_title('Fitness for a scenario where c2<c1<c3 (Figure 2)')
    axes[0].set_xlabel('Generations')
    axes[0].set_ylabel('Fitness')
    axes[0].legend(['Fitness OC', 'Fitness OB', 'Fitness MM', 'Average fitness'])

    # Plot the second subplot
    df_Figure_2_second_line.plot(x='Generation', y=['xOC', 'xOB', 'xMM'], ax=axes[1])
    axes[1].set_title('Dynamics for a scenario where c2<c1<c3 (Figure 2)')
    axes[1].set_xlabel('Generations')
    axes[1].set_ylabel('Fitness/Frequency')
    axes[1].legend(['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.tight_layout()
    plt.show()

    # Determine the direction of both lines in a ternary plot
    fig1 = px.line_ternary(df_Figure_2_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_2_second_line, a='xOC', b='xOB', c='xMM')

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics (Figure 2)')

    # Make the plot clear
    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig1.update_layout(title_text='Dynamics for a scenario where c2<c1<c3 (Figure 2)')
    save_ternary(fig1, 'Ternary_plot_Figure_2',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 3
The paramter values of this plot are not given so the values of Figure 2 are taken.
This is a possible reason that the plots differ from the ones in the paper. The
Fitness is also added. As expected the equilibriums are when the average fitness is
the highest"""

def Figure_3():
    """Function that makes Figure 3 in the paper of Sartakhti et al., 2016."""
    # Set start parameter values
    N = 10
    c3 = 1.4
    c2 = 1.2
    c1 = 1
    xOC = 0.499
    xOB = 0.499
    xMM = 0.002

    # Payoff matrix
    matrix = np.array([
        [0, 1, 2.5],
        [1, 0, -0.3],
        [2.5, 0, 0]])

    t = np.linspace(0, 40,100)

    # Initial conditions
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_3_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequencies
    xOC = 0.945
    xOB = 0.05
    xMM = 0.005

    # Initial conditions
    t = np.linspace(0, 40, 100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # defsetermine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_3_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequencies
    xOC = 0.01
    xOB = 0.19
    xMM = 0.8

    # Initial conditions
    t = np.linspace(0, 40, 100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_3_third_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_3_first_line, 'df_Figure_2_first_line.csv',
                                r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_3_second_line, 'df_Figure_2_second_line.csv',
                                r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_3_third_line, 'df_Figure_2_third_line.csv',
                                r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot with three subplos
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each dataframe in one subplot
    df_Figure_3_first_line.plot(ax=axes[0], x='Generation', y=['xOC', 'xOB', 'xMM'],
                            label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    axes[0].set_title('Dynamics for a scenario where c2<c1<c3 (Figure 3)')
    axes[0].set_xlabel('Generations')
    axes[0].set_ylabel('Fitness/Frequency')

    df_Figure_3_second_line.plot(ax=axes[1], x='Generation', y=['xOC', 'xOB', 'xMM'],
                            label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    axes[1].set_title('Dynamics for a scenario where c2<c1<c3 (Figure 3)')
    axes[1].set_xlabel('Generations')
    axes[1].set_ylabel('Fitness/Frequency')

    df_Figure_3_third_line.plot(ax=axes[2], x='Generation', y=['xOC', 'xOB', 'xMM'],
                            label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    axes[2].set_title('Dynamics for a scenario where c2<c1<c3 (Figure 3)')
    axes[2].set_xlabel('Generations')
    axes[2].set_ylabel('Fitness/Frequency')
    plt.tight_layout()
    save_Figure(fig1, 'Line_plot_Figure_3',
                     r'..\visualisation\reproduced_results_Sartakhti_linear')
    plt.show()

""" Figure 5"""
def Figure_5():
    """Function that makes Figure 5 in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 2
    c3 = 1.4
    c2 = 1.2
    c1 = 1
    xOC = 0.15
    xOB = 0.85
    xMM = 0.0

    # Payoff matrix
    matrix = np.array([
        [0, 1, 2.5],
        [1, 0, -0.3],
        [2.5, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 30, 30)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_5_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequencies
    xOC = 0.3
    xOB = 0.5
    xMM = 0.2

    # Initial conditions
    t = np.linspace(0, 30, 30)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_5_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_5_first_line, 'df_Figure_5_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_5_second_line, 'df_Figure_5_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make line plots
    df_Figure_5_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 5)')
    plt.legend()
    plt.show()

    df_Figure_5_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 5)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_5_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_5_second_line, a='xOC', b='xOB', c='xMM')
    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2<c1<c3 (Figure 5)')
    save_ternary(fig1, 'Ternary_plot_Figure_5',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 8A"""
def Figure_8A():
    """Function that makes Figure 8A in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 10
    c3 = 1.4
    c2 = 1.2
    c1 = 1
    xOC = 0.8
    xOB = 0.2
    xMM = 0.0

    # Payoff matrix
    matrix = np.array([
        [0, 1, 1.1],
        [1, 0, -0.3],
        [1.1, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 40,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_8A_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start parameter value
    xOC = 0.2
    xOB = 0.5
    xMM = 0.3

    # Initial conditions
    t = np.linspace(0, 40,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_8A_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_8A_first_line, 'df_Figure_8A_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_8A_second_line, 'df_Figure_8A_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_8A_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 8A)')
    plt.legend()
    plt.show()

    df_Figure_8A_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 8A)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_8A_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_8A_second_line, a='xOC', b='xOB', c='xMM')
    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2<c1<c3 (Figure 8A)')
    save_ternary(fig1, 'Ternary_plot_Figure_8A',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 8B"""
def Figure_8B():
    """Function that makes Figure 8B in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 10
    c3 = 1.4
    c2 = 1.2
    c1 = 1

    xOC = 0.8
    xOB = 0.2
    xMM = 0.0

    # Payoff matrix
    matrix = np.array([
        [0, 1, 1.8],
        [1, 0, -0.3],
        [1.8, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_8B_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequencies
    xOC = 0.1
    xOB = 0.7
    xMM = 0.2

    # Initial conditions
    t = np.linspace(0, 40, 40)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_8B_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_8B_first_line, 'df_Figure_8B_first_line.csv',
                                   r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_8B_second_line, 'df_Figure_8B_second_line.csv',
                                   r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_8B_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics (Figure 8B)')
    plt.legend()
    plt.show()

    df_Figure_8B_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', ])

    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 8B)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_8B_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_8B_second_line, a='xOC', b='xOB', c='xMM')
    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2<c1<c3 (Figure 8B)')
    save_ternary(fig1, 'Ternary_plot_Figure_8B',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""SENARIO 2"""
"""Figure 9A"""
def Figure_9A():
    """Function that makes Figure 9A in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 10
    c3 = 1
    c2 = 1
    c1 = 1
    xOC = 0.8
    xOB = 0.2
    xMM = 0.0

    # Payoff matrix
    matrix = np.array([
        [0, 1, 0.5],
        [1, 0, -0.3],
        [0.5, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 40,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_9A_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequencies
    xOC = 0.4
    xOB = 0.3
    xMM = 0.3

    # Initial conditions
    t = np.linspace(0, 40,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_9A_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_9A_first_line, 'df_Figure_9A_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_9A_second_line, 'df_Figure_9A_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_9A_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2=c1=c3 (Figure 9A)')
    plt.legend()
    plt.show()

    df_Figure_9A_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2=c1=c3 (Figure 9A)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_9A_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_9A_second_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2=c1=c3 (Figure 9A)')
    save_ternary(fig1, 'Ternary_plot_Figure_9A',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

""" Figure 9B """
def Figure_9B():
    """Function that makes Figure 9B in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 10
    c3 = 1
    c2 = 1
    c1 = 1
    xOC = 0.8
    xOB = 0.2
    xMM = 0.0

    # Payoff matrix
    matrix = np.array([
        [0, 1, 2],
        [1, 0, 0],
        [2, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 40,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_9B_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequencies
    xOC = 0.4
    xOB = 0.3
    xMM = 0.3

    # Initial conditions
    t = np.linspace(0, 40,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_9B_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_9B_first_line, 'df_Figure_9B_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_9B_second_line, 'df_Figure_9B_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_9B_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2=c1=c3 (Figure 9B)')
    plt.legend()
    plt.show()

    df_Figure_9B_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2=c1=c3 (Figure 9B)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_9B_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_9B_second_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2=c1=c3 (Figure 9B)')
    save_ternary(fig1, 'Ternary_plot_Figure_9B',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

""" Figure 9C """
def Figure_9C():
    """Function that makes Figure 9C in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 10
    c3 = 1
    c2 = 1
    c1 = 1
    xOC = 0.8
    xOB = 0.2
    xMM = 0.0

    # Payoff matrix
    matrix = np.array([
        [0, 1, 0.5],
        [1, 0, -1],
        [0.5, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 40,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_9C_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequencies
    xOC = 0.4
    xOB = 0.3
    xMM = 0.3

    # Initial conditions
    t = np.linspace(0, 40,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_9C_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_9C_first_line, 'df_Figure_9C_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_9C_second_line, 'df_Figure_9C_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_9C_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2=c1=c3 (Figure 9C)')
    plt.legend()
    plt.show()

    df_Figure_9C_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2=c1=c3 (Figure 9C)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_9C_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_9C_second_line, a='xOC', b='xOB', c='xMM')
    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2=c1=c3 (Figure 9C)')
    save_ternary(fig1, 'Ternary_plot_Figure_9C',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""SENARIO 3"""
"""Figure 10A"""
def Figure_10A():
    """Function that makes Figure 10A in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 10
    c3 = 0.8
    c2 = 1.2
    c1 = 1
    xOC = 0.7
    xOB = 0.2
    xMM = 0.1

    # Payoff matrix
    matrix = np.array([
        [0, 1, 0.5],
        [1, 0, -0.3],
        [0.5, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 40)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_10A_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start values
    xOC = 0.2
    xOB = 0.2
    xMM = 0.6

    # Initial conditions
    t = np.linspace(0, 40)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_10A_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_10A_first_line, 'df_Figure_10A_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_10A_second_line, 'df_Figure_10A_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_10A_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c3<c1<c2 (Figure 10A)')
    plt.legend()
    plt.show()

    df_Figure_10A_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c3<c1<c2 (Figure 10A)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_10A_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_10A_second_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c3<c1<c2 (Figure 10A)')
    save_ternary(fig1, 'Ternary_plot_Figure_10A',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 10B"""
def Figure_10B():
    """Function that makes Figure 10B in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 10
    c3 = 0.8
    c2 = 1.2
    c1 = 1
    xOC = 0.4
    xOB = 0.6
    xMM = 0.0

    # Payoff matrix
    matrix = np.array([
        [0, 1, 0.5],
        [1, 0, 0.3],
        [0.5, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 40,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_10B_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequenties
    xOC = 0.4
    xOB = 0.3
    xMM = 0.3

    # Initial conditions
    t = np.linspace(0, 40,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_10B_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    save_data(df_Figure_10B_first_line, 'df_Figure_10B_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_10B_second_line, 'df_Figure_10B_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_10B_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c3<c1<c2 (Figure 10B)')
    plt.legend()
    plt.show()

    df_Figure_10B_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c3<c1<c2 (Figure 10B)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_10B_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_10B_second_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c3<c1<c2 (Figure 10B)')
    save_ternary(fig1, 'Ternary_plot_Figure_10B',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 11"""
def Figure_11():
    """Function that makes Figure 11 in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 20
    c3 = 1.4
    c2 = 1
    c1 = 1.2
    xOC = 0.3
    xOB = 0.25
    xMM = 0.45

    # Payoff matrix
    matrix = np.array([
        [0, 1, 1.5],
        [1, 0, -0.3],
        [1.5, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 20, 20)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    xOC = 0.65
    xOB = 0.25
    xMM = 0.1

    # Initial conditions
    t = np.linspace(20, 50, 50)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})
    df_Figure_11_first_line = pd.concat([df_1, df_2])

    # Make lists
    WOC_list = []
    WOB_list = []
    WMM_list = []
    W_average_list = []
    generation_list = []


    save_data(df_Figure_11_first_line, 'df_Figure_11_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')


    # Make a plot
    df_Figure_11_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Effect reducing MM cells (Figure 11)')
    plt.legend()
    save_Figure(plt, 'Line_plot_Figure_11',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_11_first_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    fig1.update_layout(title_text= 'Effect reducing MM cells (Figure 11)')
    save_ternary(fig1, 'Ternary_plot_Figure_11',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 12 A middel"""
def Figure_12A_middel():
    """Function that makes Figure 12A middel in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 15
    c3 = 1.8
    c2 = 1
    c1 = 1.2
    xOC = 0.75
    xOB = 0.2
    xMM = 0.05

    # Payoff matrix
    matrix = np.array([
        [0, 1, 2.0],
        [1, 0, -0.3],
        [2.0, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 30,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12A_middel_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequenties
    xOC = 0.2
    xOB = 0.2
    xMM = 0.6

    # Initial conditions
    t = np.linspace(0, 30,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12A_middel_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_12A_middel_first_line, 'df_Figure_12A_middel_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_12A_middel_second_line, 'df_Figure_12A_middel_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_12A_middel_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    ])

    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12A middel)')
    plt.legend()
    plt.show()

    df_Figure_12A_middel_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    ])

    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12A middel)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_12A_middel_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_12A_middel_second_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2<c1<c3 (Figure 12A middel)')
    save_ternary(fig1, 'Ternary_plot_Figure_12A_middel',
                    r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 12 A right"""
def Figure_12A_right():
    """Function that makes Figure 12A middel in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 2
    c3 = 1.8
    c2 = 1
    c1 = 1.2
    xOC = 0.7
    xOB = 0.1
    xMM = 0.2

    # Payoff matrix
    matrix = np.array([
        [0, 1, 2.0],
        [1, 0, -0.3],
        [2.0, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 30,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12A_right_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequencies
    xOC = 0.3
    xOB = 0.3
    xMM = 0.4

    t = np.linspace(0, 30,100)

    # Initial conditions
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    #  the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12A_right_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_12A_right_first_line, 'df_Figure_12A_right_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_12A_right_second_line, 'df_Figure_12A_right_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_12A_right_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    ])

    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12A right)')
    plt.legend()
    plt.show()

    df_Figure_12A_right_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12A right)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_12A_right_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_12A_right_second_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2<c1<c3 (Figure 12A right)')
    save_ternary(fig1, 'Ternary_plot_Figure_12A_right',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

""" Figure 12B middel"""
def Figure_12B_middel():
    """Function that makes Figure 12B middel in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 15
    c3 = 1.4
    c2 = 1
    c1 = 1.2
    xOC = 0.4
    xOB = 0.1
    xMM = 0.5

    # Payoff matrix
    matrix = np.array([
        [0, 1, 0.5],
        [1, 0, -0.33],
        [0.5, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 30,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12B_middel_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequenties
    xOC = 0.1
    xOB = 0.2
    xMM = 0.7

    # Initial conditions
    t = np.linspace(0, 30,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12B_middel_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_12B_middel_first_line, 'df_Figure_12B_middel_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_12B_middel_second_line, 'df_Figure_12B_middel_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_12B_middel_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    ])

    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12B middel)')
    plt.legend()
    plt.show()

    df_Figure_12B_middel_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    ])

    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12B middel)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_12B_middel_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_12B_middel_second_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2<c1<c3 (Figure 12B middel)')
    save_ternary(fig1, 'Ternary_plot_Figure_12B_middel',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')

    fig1.show()

""" Figure 12B right"""
def Figure_12B_right():
    """Function that makes Figure 12B right in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 15
    c3 = 1.4
    c2 = 1
    c1 = 1.2
    xOC = 0.4
    xOB = 0.0
    xMM = 0.6

    # Payoff matrix
    matrix = np.array([
        [0, 1, 0.5],
        [1, 0, -0.33],
        [0.5, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 30,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12B_right_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequencies
    xOC = 0.1
    xOB = 0.2
    xMM = 0.7

    # Initial conditions
    t = np.linspace(0, 30,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12B_right_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_12B_right_first_line, 'df_Figure_12B_right_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_12B_right_second_line, 'df_Figure_12B_right_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_12B_right_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    ])

    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12B right)')
    plt.legend()
    plt.show()

    df_Figure_12B_right_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    ])

    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12B right)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_12B_right_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_12B_right_second_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2<c1<c3 (Figure 12B right)')
    save_ternary(fig1, 'Ternary_plot_Figure_12B_right',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')

    fig1.show()

""" Figure 12C middel"""
def Figure_12C_middel():
    """Function that makes Figure 12C middel in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 15
    c3 = 1.4
    c2 = 1
    c1 = 1.2
    xOC = 0.1
    xOB = 0.2
    xMM = 0.7

    # Payoff matrix
    matrix = np.array([
        [0, 1, 1],
        [1, 0, -0.5],
        [0.5, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 30,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12C_middel_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start frequencies
    xOC = 0.8
    xOB = 0.1
    xMM = 0.1

    # Initial conditions
    t = np.linspace(0, 30,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12C_middel_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_12C_middel_first_line, 'df_Figure_12C_middel_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_12C_middel_second_line, 'df_Figure_12C_middel_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_12C_middel_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12C middel)')
    plt.legend()
    plt.show()

    df_Figure_12C_middel_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                            label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12C middel)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_12C_middel_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_12C_middel_second_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2<c1<c3 (Figure 12C middel)')
    save_ternary(fig1, 'Ternary_plot_Figure_12C_middel',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

""" Figure 12C right"""
def Figure_12C_right():
    """Function that makes Figure 12C right in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 15
    c3 = 0.8
    c2 = 1
    c1 = 1.2
    xOC = 0.1
    xOB = 0.2
    xMM = 0.7

    # Payoff matrix
    matrix = np.array([
        [0, 1, 1],
        [1, 0, -0.5],
        [0.5, 0, 0]])

    # Initial conditions
    t = np.linspace(0, 30,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12C_right_first_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Set new start values
    xOC = 0.8
    xOB = 0.1
    xMM = 0.1

    # Initial conditions
    t = np.linspace(0, 30,100)
    y0 = [xOC, xOB, xMM]
    parameters = (N, c1, c2, c3, matrix)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_Figure_12C_right_second_line = pd.DataFrame({'Generation': t, 'xOC': y[:, 0],
    'xOB': y[:, 1], 'xMM': y[:, 2]})

    # Save the data as csv file
    save_data(df_Figure_12C_right_first_line, 'df_Figure_12C_right_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_Figure_12C_right_second_line, 'df_Figure_12C_right_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_Figure_12C_right_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12C right)')
    plt.legend()
    plt.show()

    df_Figure_12C_right_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                            label = ['Frequency OC', 'Frequency OB', 'Frequency MM'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Dynamics for a scenario where c2<c1<c3 (Figure 12C right)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_Figure_12C_right_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_Figure_12C_right_second_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics for a scenario where c2<c1<c3 (Figure 12C right)')
    save_ternary(fig1, 'Ternary_plot_Figure_12C_right',
                    r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

if __name__ == "__main__":
    main()
