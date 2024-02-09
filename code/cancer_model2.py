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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ternary
import plotly.graph_objects as go
import plotly.express as px
from cancer_model import *
import plotly.io as pio
from scipy.special import comb


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

"""Figure 10A"""
def figure_10A():
    """Function that makes figure 10A in the paper of Sartakhti et al., 2016."""
    # Set start values
    N = 10
    c3 = 1
    c2 = 1.2
    c1 = 1.2
    xOC = 0.1
    xOB = 0.1
    xMM = 0.8

    # Payoff matrix
    matrix = np.array([
        [0.3, 1, 2],
        [1, 1.4, 1.5],
        [-0.5, -0.9, 1.2]])

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

    # Make a plot
    df_figure_10A_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for a scenario where c3<c1<c2 (figure 10A)')
    plt.legend()
    plt.show()

    df_figure_10A_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness/ Frequency')
    plt.title('Dynamics for a scenario where c3<c1<c2 (figure 10A)')
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
    fig1.update_layout(title_text= 'Dynamics for a scenario where c3<c1<c2 (figure 10A)')
    save_ternary(fig1, 'Ternary_plot_figure_10A',
                        r'..\visualisation\reproduced_results_Sartakhti_linear')
    fig1.show()

# figure_10A()

def figure_effect_growth_factor_inhibition():
    """Function that simulates the effect of growth inhibitor resistance"""
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
        [0.1, 1, 1.5],
        [1.2, 0.1, -0.3],
        [1.5, 0.9, 0.1]])

    generations = 100

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_11_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        generations = generation *1

        #Reduce effect of OC GF on MM cells
        if generation == 30:
            matrix = np.array([
                [0.3, 1, 1.5],
                [1.2, 0.1, -0.3],
                [1.2, 0.9, 0.2]])

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

    # Make a plot
    df_figure_11_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'])
    plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
    plt.xlabel('Generations')
    plt.ylabel('Frequency')
    plt.title('Effect GF inhibition')
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
    fig1.update_layout(title_text= 'Effect GF inhibition')

    fig1.show()

# figure_effect_growth_factor_inhibition()

def figure_effect_resistentie():
    """Function that simulates the effect of growth inhibitor resistance"""
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
        [0.3, 1, 1.5],
        [1.2, 0.1, -0.3],
        [1.5, 0.9, 0.2]])

    generations = 110

    # Make a dataframe
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_figure_11_first_line = pd.DataFrame(columns=column_names)

    # Determine the frequentie value a number of times
    for generation in range(generations):
        generations = generation *1

        # Reduce effect of OC GF on MM cells
        if generation == 15:
            matrix = np.array([
                [0.3, 1, 1.5],
                [1.2, 0.1, -0.3],
                [1.2, 0.9, 0.2]])

        # Development resistance
        if generation == 75:
            matrix = np.array([
                [0.3, 1, 1.5],
                [1.2, 0.1, -0.3],
                [1.5, 0.9, 0.2]])

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

    # Make a plot
    df_figure_11_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
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

    fig1.show()

figure_effect_resistentie()
