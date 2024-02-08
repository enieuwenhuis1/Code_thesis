"""
Author:       Eva Nieuwenhuis
University:   UvA
Student id':  13717405
Description:  Code attempting to replicate certain figures from the paper by Sartakhti
              et al. (2016). The used model simulates the dynamics in the multiple
              myeloma (MM) microenvironment with three cell types: MM cells, osteoblasts
              (OBs) and osteoclasts (OCs). The model is a public goods game in the
              framework of evolutionary game theory with linear benefits.

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
from cancer_model import *
import plotly.io as pio
from scipy.integrate import odeint

def main():
    # Do doc tests
    import doctest
    doctest.testmod()

    # Make figure 2 in the paper of Sartakhti et al., 2016.
    figure_2()

    # Make figure 3 in the paper of Sartakhti et al., 2016.
    figure_3()

    # Make figure 5 in the paper of Sartakhti et al., 2016.
    figure_5()

    # Make figure 8A in the paper of Sartakhti et al., 2016.
    figure_8A()

    # Make figure 8B in the paper of Sartakhti et al., 2016.
    figure_8B()

    # Make figure 9A in the paper of Sartakhti et al., 2016.
    figure_9A()

    # Make figure 9B in the paper of Sartakhti et al., 2016.
    figure_9B()

    # Make figure 9C in the paper of Sartakhti et al., 2016.
    figure_9C()

    # Make figure 10A in the paper of Sartakhti et al., 2016.
    figure_10A()

    # Make figure 10B in the paper of Sartakhti et al., 2016.
    figure_10B()

    # Make figure 11 in the paper of Sartakhti et al., 2016.
    figure_11()

    # Make figure 12A middel in the paper of Sartakhti et al., 2016.
    figure_12A_middel()

    # Make figure 12A right in the paper of Sartakhti et al., 2016.
    figure_12A_right()

    # Make figure 12B middel in the paper of Sartakhti et al., 2016.
    figure_12B_middel()

    # Make figure 12B right in the paper of Sartakhti et al., 2016.
    figure_12B_right()

    # Make figure 12C middel in the paper of Sartakhti et al., 2016.
    figure_12C_middel()

    # Make figure 12C right in the paper of Sartakhti et al., 2016.
    figure_12C_right()

def save_figure(figure, file_name, folder_path):
    """Save the figure to a specific folder.

    Parameters:
    -----------
    figure: Matplotlib figure
        Figure object that needs to be saved.
    file_name : String
        The name for the plot.
    folder_path: String:
        Path to the folder where the data will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    figure.savefig(os.path.join(folder_path, file_name))

def save_ternary(figure, file_name, folder_path):
    """Save the ternary plot in a specific folder.

    Parameters:
    -----------
    figure: Matplotlib figure
        Figure object that needs to be saved.
    file_name : String
        The name for the plot.
    folder_path: String:
        Path to the folder where the data will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    pio.write_image(figure, os.path.join(folder_path, f'{file_name}.png'), format='png')


def fitness_WOC(x, y, z, N, c1, c2, c3, matrix):
    """
    Function that calculates the fitness of an osteoclast in a population.

    Parameters:
    -----------
    x: float
        Frequency of OCs.
    y: float
        Frequency of OBs.
    z: float
        Frequency of the MM cells.
    N: int
        Number of individuals within the interaction range.
    c1: float
        Cost parameter OCs.
    c2: float
        Cost parameter OBs.
    c3: float
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

    WOC = (c*c3*z + b*c2*y + a* c1* x)*(N - 1)/N - c1 #(18)
    return WOC

def fitness_WOB(x, y, z, N, c1, c2, c3, matrix):
    """
    Function that calculates the fitness of an osteoblast in a population.

    Parameters:
    -----------
    x: float
        Frequency of OCs.
    y: float
        Frequency of OBs.
    z: float
        Frequency of the MM cells.
    N: int
        Number of individuals within the interaction range.
    c1: float
        Cost parameter OCs.
    c2: float
        Cost parameter OBs.
    c3: float
        Cost parameter MM cells.
    matrix: numpy.ndarray
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

    WOB = (d*c1*x + f*c3*z + e*c2*y)*(N - 1)/N - c2 #(19)
    return WOB

def fitness_WMM(x, y, z, N, c1, c2, c3,matrix):
    """
    Function that calculates the fitness of an MM cell in a population.

    Parameters:
    -----------
    x: float
        Frequency of OCs.
    y: float
        Frequency of OBs.
    z: float
        Frequency of the MM cells.
    N: int
        Number of individuals within the interaction range.
    c1: float
        Cost parameter OCs.
    c2: float
        Cost parameter OBs.
    c3: float
        Cost parameter MM cells.
    matrix: numpy.ndarray
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

    WMM = (g*c1*x + i*c3*z + h*c2*y)*(N - 1)/N - c3 #(20)
    return WMM

def model_dynamics(y, t, N, c1, c2, c3, matrix):
    """Determines the frequenty dynamics in a population over time.

    Parameters:
    -----------
    y : List
        List with the values of xOC, xOB, and xMM.
    t : List
        List with all the time points.
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

"""Figure 2"""
def figure_2():
    """Function that makes figure 2 in the paper of Sartakhti et al., 2016."""
    # Set start parameter values
    xOC = 0.5
    xOB = 0.45
    xMM = 0.05
    N = 10
    c3 = 1.4
    c2 = 1.2
    c1 = 1

    # Payoff matrix
    matrix = np.array([
        [0, 1, 2.5],
        [1, 0, -0.3],
        [2.5, 0, 0]])

    generations = 50

    # Create a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_2_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_2_first_line = pd.concat([df_figure_2_first_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequencies
    xOC = 0.2
    xOB = 0.4
    xMM = 0.4

    # Create a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_2_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_2_second_line = pd.concat([df_figure_2_second_line, new_row],
                                                            ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_2_first_line, 'df_figure_2_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_2_second_line, 'df_figure_2_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Determine the direction of both lines in a ternary plot
    fig1 = px.line_ternary(df_figure_2_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_2_second_line, a='xOC', b='xOB', c='xMM')

    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.data[0].update(line=dict(color='red'))
    fig1.data[1].update(line=dict(color='blue'))
    fig1.update_layout(title_text= 'Dynamics (figure 2)')

    # Make the plot clear
    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig1.update_layout(title_text='Dynamics for cenario where c2<c1<c3 (figure 2)')
    save_ternary(fig1, 'Ternary_plot_figure_2',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 3
The paramter values of this plot are not given so the values of figure 2 are taken.
This is a possible reason that the plots differ from the ones in the paper. The
Fitness is also added. As expected the equilibriums are when the average fitness is
the highest"""

def figure_3():
    """Function that makes figure 3 in the paper of Sartakhti et al., 2016."""
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

    generations_plot = 20

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_3_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations_plot):
        generations = generation *10

        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generations, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_3_first_line = pd.concat([df_figure_3_first_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequencies
    xOC = 0.945
    xOB = 0.05
    xMM = 0.005

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_3_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations_plot):
        generations = generation *10

        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generations, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_3_second_line = pd.concat([df_figure_3_second_line, new_row],
                                                            ignore_index=True)

        # Update the xOC, xOB, xMM values and nOC, nOB and nMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequencies
    xOC = 0.01
    xOB = 0.19
    xMM = 0.8

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_3_third_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations_plot):
        generations = generation *10

        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generations, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_3_third_line = pd.concat([df_figure_3_third_line, new_row],
                                                            ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_3_first_line, 'df_figure_2_first_line.csv',
                                r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_3_second_line, 'df_figure_2_second_line.csv',
                                r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_3_third_line, 'df_figure_2_third_line.csv',
                                r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot with three subplos
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each dataframe in one subplot
    df_figure_3_first_line.plot(ax=axes[0], x='Generation', y=['xOC', 'xOB', 'xMM'])
    axes[0].set_title('Dynamics for scenario where c2<c1<c3 (figure 3)')
    axes[0].set_xlabel('Generations')
    axes[0].set_ylabel('Fitness/Frequency')

    df_figure_3_second_line.plot(ax=axes[1], x='Generation', y=['xOC', 'xOB', 'xMM'])
    axes[1].set_title('Dynamics for scenario where c2<c1<c3 (figure 3)')
    axes[1].set_xlabel('Generations')
    axes[1].set_ylabel('Fitness/Frequency')

    df_figure_3_third_line.plot(ax=axes[2], x='Generation', y=['xOC', 'xOB', 'xMM'])
    axes[2].set_title('Dynamics for scenario where c2<c1<c3 (figure 3)')
    axes[2].set_xlabel('Generations')
    axes[2].set_ylabel('Fitness/Frequency')
    plt.tight_layout()
    save_figure(fig1, 'Line_plot_figure_3',
                     r'..\visualisation\reproduced_results_Sartakhti_linear')
    plt.show()

""" Figure 5"""
def figure_5():
    """Function that makes figure 5 in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 2
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

    generations = 50

    # make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_5_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_5_first_line = pd.concat([df_figure_5_first_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequencies
    xOC = 0.4
    xOB = 0.3
    xMM = 0.3

    generations = 50

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_5_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_5_second_line = pd.concat([df_figure_5_second_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_5_first_line, 'df_figure_5_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_5_second_line, 'df_figure_5_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make line plots
    df_figure_5_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 5)')
    plt.legend()
    plt.show()

    df_figure_5_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 5)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_5_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_5_second_line, a='xOC', b='xOB', c='xMM')
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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2<c1<c3 (figure 5)')
    save_ternary(fig1, 'Ternary_plot_figure_5',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""figure 8A"""
def figure_8A():
    """Function that makes figure 8A in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_8A_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_8A_first_line = pd.concat([df_figure_8A_first_line, new_row],
                                                             ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start parameter value
    xOC = 0.2
    xOB = 0.5
    xMM = 0.3

    # Create a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_8A_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_8A_second_line = pd.concat([df_figure_8A_second_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_8A_first_line, 'df_figure_8A_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_8A_second_line, 'df_figure_8A_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_8A_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 8A)')
    plt.legend()
    plt.show()

    df_figure_8A_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 8A)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_8A_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_8A_second_line, a='xOC', b='xOB', c='xMM')
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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2<c1<c3 (figure 8A)')
    save_ternary(fig1, 'Ternary_plot_figure_8A',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""figure 8B"""
def figure_8B():
    """Function that makes figure 8B in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # Create a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_8B_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                        'xMM': xMM, 'W_average': W_average}])
        df_figure_8B_first_line = pd.concat([df_figure_8B_first_line, new_row],
                                                            ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)
        nOC = xOC * N
        nOB = xOB * N
        nMM = xMM * N

    # Set new start frequencies
    xOC = 0.2
    xOB = 0.5
    xMM = 0.3

    # Create a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_8B_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_8B_second_line = pd.concat([df_figure_8B_second_line, new_row],
                                                            ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_8B_first_line, 'df_figure_8B_first_line.csv',
                                   r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_8B_second_line, 'df_figure_8B_second_line.csv',
                                   r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_8B_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics (figure 8B)')
    plt.legend()
    plt.show()

    df_figure_8B_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 8B)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_8B_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_8B_second_line, a='xOC', b='xOB', c='xMM')
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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2<c1<c3 (figure 8B)')
    save_ternary(fig1, 'Ternary_plot_figure_8B',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""SENARIO 2"""
"""figure 9A"""
def figure_9A():
    """Function that makes figure 9A in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # Create a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_9A_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_9A_first_line = pd.concat([df_figure_9A_first_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequencies
    xOC = 0.4
    xOB = 0.3
    xMM = 0.3

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_9A_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_9A_second_line = pd.concat([df_figure_9A_second_line, new_row],
                                                            ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_9A_first_line, 'df_figure_9A_first_line.csv',
                                                r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_9A_second_line, 'df_figure_9A_second_line.csv',
                                                r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_9A_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2=c1=c3 (figure 9A)')
    plt.legend()
    plt.show()

    df_figure_9A_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2=c1=c3 (figure 9A)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_9A_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_9A_second_line, a='xOC', b='xOB', c='xMM')

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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2=c1=c3 (figure 9A)')
    save_ternary(fig1, 'Ternary_plot_figure_9A',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

""" Figure 9B """
def figure_9B():
    """Function that makes figure 9B in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_9B_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                        'xMM': xMM, 'W_average': W_average}])
        df_figure_9B_first_line = pd.concat([df_figure_9B_first_line, new_row],
                                                            ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequencies
    xOC = 0.4
    xOB = 0.3
    xMM = 0.3

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_9B_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_9B_second_line = pd.concat([df_figure_9B_second_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_9B_first_line, 'df_figure_9B_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_9B_second_line, 'df_figure_9B_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_9B_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2=c1=c3 (figure 9B)')
    plt.legend()
    plt.show()

    df_figure_9B_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2=c1=c3 (figure 9B)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_9B_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_9B_second_line, a='xOC', b='xOB', c='xMM')

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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2=c1=c3 (figure 9B)')
    save_ternary(fig1, 'Ternary_plot_figure_9B',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

""" Figure 9C """
def figure_9C():
    """Function that makes figure 9C in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_9C_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)


        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                          'xMM': xMM, 'W_average': W_average}])
        df_figure_9C_first_line = pd.concat([df_figure_9C_first_line, new_row],
                                                            ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)
        nOC = xOC * N
        nOB = xOB * N
        nMM = xMM * N

    # Set new start frequencies
    xOC = 0.4
    xOB = 0.3
    xMM = 0.3

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_9C_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_9C_second_line = pd.concat([df_figure_9C_second_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_9C_first_line, 'df_figure_9C_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_9C_second_line, 'df_figure_9C_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_9C_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2=c1=c3 (figure 9C)')
    plt.legend()
    plt.show()

    df_figure_9C_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2=c1=c3 (figure 9C)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_9C_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_9C_second_line, a='xOC', b='xOB', c='xMM')
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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2=c1=c3 (figure 9C)')
    save_ternary(fig1, 'Ternary_plot_figure_9C',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""SENARIO 3"""
"""Figure 10A"""
def figure_10A():
    """Function that makes figure 10A in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_10A_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                        'xMM': xMM, 'W_average': W_average}])
        df_figure_10A_first_line = pd.concat([df_figure_10A_first_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start values
    xOC = 0.2
    xOB = 0.2
    xMM = 0.6

    # Make a datadrame
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_10A_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                                'xMM': xMM, 'W_average': W_average}])
        df_figure_10A_second_line = pd.concat([df_figure_10A_second_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_10A_first_line, 'df_figure_10A_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_10A_second_line, 'df_figure_10A_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_10A_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c3<c1<c2 (figure 10A)')
    plt.legend()
    plt.show()

    df_figure_10A_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c3<c1<c2 (figure 10A)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_10A_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_10A_second_line, a='xOC', b='xOB', c='xMM')

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
    fig1.update_layout(title_text= 'Dynamics for scenario where c3<c1<c2 (figure 10A)')
    save_ternary(fig1, 'Ternary_plot_figure_10A',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 10B"""
def figure_10B():
    """Function that makes figure 10B in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_10B_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)


        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_10B_first_line = pd.concat([df_figure_10B_first_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequenties
    xOC = 0.4
    xOB = 0.3
    xMM = 0.3

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_10B_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_10B_second_line = pd.concat([df_figure_10B_second_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    save_data(df_figure_10B_first_line, 'df_figure_10B_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_10B_second_line, 'df_figure_10B_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_10B_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c3<c1<c2 (figure 10B)')
    plt.legend()
    plt.show()

    df_figure_10B_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c3<c1<c2 (figure 10B)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_10B_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_10B_second_line, a='xOC', b='xOB', c='xMM')

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
    fig1.update_layout(title_text= 'Dynamics for scenario where c3<c1<c2 (figure 10B)')
    save_ternary(fig1, 'Ternary_plot_figure_10B',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 11"""
def figure_11():
    """Function that makes figure 11 in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 10
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

    generations = 50

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_11_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        generations = generation *15

        # reduce number of MM cells
        if generation == 10:
            xOC = 0.65
            xOB = 0.25
            xMM = 0.1

        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)


        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generations, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_11_first_line = pd.concat([df_figure_11_first_line, new_row],
                                                                ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    save_data(df_figure_11_first_line, 'df_figure_11_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')


    # Make a plot
    df_figure_11_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Effect reducing MM cells (figure 11)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_11_first_line, a='xOC', b='xOB', c='xMM')

    fig1.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))

    # Add both lines to one ternary plot
    fig1.update_layout(title_text= 'Effect reducing MM cells (figure 11)')
    save_ternary(fig1, 'Ternary_plot_figure_11',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 12 A middel"""
def figure_12A_middel():
    """Function that makes figure 12A middel in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12A_middel_first_line = pd.DataFrame(columns=column_names)

    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_12A_middel_first_line = pd.concat([df_figure_12A_middel_first_line,
                                                        new_row], ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequenties
    xOC = 0.2
    xOB = 0.2
    xMM = 0.6

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12A_middel_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                         'xMM': xMM, 'W_average': W_average}])
        df_figure_12A_middel_second_line = pd.concat([df_figure_12A_middel_second_line,
                                                    new_row], ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_12A_middel_first_line, 'df_figure_12A_middel_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_12A_middel_second_line, 'df_figure_12A_middel_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_12A_middel_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12A middel)')
    plt.legend()
    plt.show()

    df_figure_12A_middel_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12A middel)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_12A_middel_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_12A_middel_second_line, a='xOC', b='xOB', c='xMM')

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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2<c1<c3 (figure 12A middel)')
    save_ternary(fig1, 'Ternary_plot_figure_12A_middel',
                    r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

"""Figure 12 A right"""
def figure_12A_right():
    """Function that makes figure 12A middel in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12A_right_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                        'xMM': xMM, 'W_average': W_average}])
        df_figure_12A_right_first_line = pd.concat([df_figure_12A_right_first_line,
                                                        new_row], ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequencies
    xOC = 0.3
    xOB = 0.3
    xMM = 0.4

    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12A_right_second_line = pd.DataFrame(columns=column_names)

    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                                'xMM': xMM, 'W_average': W_average}])
        df_figure_12A_right_second_line = pd.concat([df_figure_12A_right_second_line, new_row],
                                                                    ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_12A_right_first_line, 'df_figure_12A_right_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_12A_right_second_line, 'df_figure_12A_right_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_12A_right_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12A right)')
    plt.legend()
    plt.show()

    df_figure_12A_right_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12A right)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_12A_right_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_12A_right_second_line, a='xOC', b='xOB', c='xMM')

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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2<c1<c3 (figure 12A right)')
    save_ternary(fig1, 'Ternary_plot_figure_12A_right',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

""" Figure 12B middel"""
def figure_12B_middel():
    """Function that makes figure 12B middel in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12B_middel_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_12B_middel_first_line = pd.concat([df_figure_12B_middel_first_line,
                                                        new_row], ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequenties
    xOC = 0.1
    xOB = 0.2
    xMM = 0.7

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12B_middel_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                                'xMM': xMM, 'W_average': W_average}])
        df_figure_12B_middel_second_line = pd.concat([df_figure_12B_middel_second_line,
                                                    new_row], ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_12B_middel_first_line, 'df_figure_12B_middel_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_12B_middel_second_line, 'df_figure_12B_middel_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_12B_middel_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12B middel)')
    plt.legend()
    plt.show()

    df_figure_12B_middel_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12B middel)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_12B_middel_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_12B_middel_second_line, a='xOC', b='xOB', c='xMM')

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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2<c1<c3 (figure 12B middel)')
    save_ternary(fig1, 'Ternary_plot_figure_12B_middel',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')

    fig1.show()

""" Figure 12B right"""
def figure_12B_right():
    """Function that makes figure 12B right in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12B_right_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_12B_right_first_line = pd.concat([df_figure_12B_right_first_line,
                                                    new_row], ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequencies
    xOC = 0.1
    xOB = 0.2
    xMM = 0.7

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12B_right_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_12B_right_second_line = pd.concat([df_figure_12B_right_second_line,
                                                    new_row], ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_12B_right_first_line, 'df_figure_12B_right_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_12B_right_second_line, 'df_figure_12B_right_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_12B_right_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12B right)')
    plt.legend()
    plt.show()

    df_figure_12B_right_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12B right)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_12B_right_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_12B_right_second_line, a='xOC', b='xOB', c='xMM')

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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2<c1<c3 (figure 12B right)')
    save_ternary(fig1, 'Ternary_plot_figure_12B_right',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')

    fig1.show()

""" Figure 12C middel"""
def figure_12C_middel():
    """Function that makes figure 12C middel in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12C_middel_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                         'xMM': xMM, 'W_average': W_average}])
        df_figure_12C_middel_first_line = pd.concat([df_figure_12C_middel_first_line,
                                                    new_row], ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start frequencies
    xOC = 0.8
    xOB = 0.1
    xMM = 0.1

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12C_middel_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_12C_middel_second_line = pd.concat([df_figure_12C_middel_second_line,
                                                    new_row], ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_12C_middel_first_line, 'df_figure_12C_middel_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_12C_middel_second_line, 'df_figure_12C_middel_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_12C_middel_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12C middel)')
    plt.legend()
    plt.show()

    df_figure_12C_middel_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12C middel)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_12C_middel_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_12C_middel_second_line, a='xOC', b='xOB', c='xMM')

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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2<c1<c3 (figure 12C middel)')
    save_ternary(fig1, 'Ternary_plot_figure_12C_middel',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

""" Figure 12C right"""
def figure_12C_right():
    """Function that makes figure 12C right in the paper of Sartakhti et al., 2016."""
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

    generations = 50

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12C_right_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
        df_figure_12C_right_first_line = pd.concat([df_figure_12C_right_first_line,
                                                    new_row], ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Set new start values
    xOC = 0.8
    xOB = 0.1
    xMM = 0.1

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_12C_right_second_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, matrix)
        WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, matrix)

        # Determine the average fittness
        W_average = xOC * WOC + xOB * WOB + xMM * WMM

        # Determine the new frequencies based of replicator dynamics
        xOC_change = xOC * (WOC - W_average) # (15)
        xOB_change = xOB * (WOB - W_average) # (16)
        xMM_change = xMM * (WMM - W_average) # (17)

        # Add row to dataframe (first add row and the update because then also the
        # beginning values get added to the dataframe at generation =0)
        new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                                'xMM': xMM, 'W_average': W_average}])
        df_figure_12C_right_second_line = pd.concat([df_figure_12C_right_second_line, new_row], ignore_index=True)

        # Update the xOC, xOB, xMM values
        xOC = max(0, xOC + xOC_change)
        xOB = max(0, xOB + xOB_change)
        xMM = max(0, xMM + xMM_change)

    # Save the data as csv file
    save_data(df_figure_12C_right_first_line, 'df_figure_12C_right_first_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')
    save_data(df_figure_12C_right_second_line, 'df_figure_12C_right_second_line.csv',
                                    r'..\data\reproduced_data_Sartakhti_linear')

    # Make a plot
    df_figure_12C_right_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12C right)')
    plt.legend()
    plt.show()

    df_figure_12C_right_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM',
                                                                    'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for scenario where c2<c1<c3 (figure 12C right)')
    plt.legend()
    plt.show()

    # Make a ternary plot
    fig1 = px.line_ternary(df_figure_12C_right_first_line, a='xOC', b='xOB', c='xMM')
    fig2 = px.line_ternary(df_figure_12C_right_second_line, a='xOC', b='xOB', c='xMM')

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
    fig1.update_layout(title_text= 'Dynamics for scenario where c2<c1<c3 (figure 12C right)')
    save_ternary(fig1, 'Ternary_plot_figure_12C_right',
                    r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

if __name__ == "__main__":
    main()
