"""
Author:       Eva Nieuwenhuis
University:   UvA
Student id':  13717405
Description:  Code that attempts to replicate the formulas and Figures from the
              paper by Sartakhti et al. (2018) to get a better understanding of
              cancer modeling. The model simulates linear and nonlinear dynamics
              and collective interactions in the multiple myeloma (MM) micro-
              environment with three cell types: MM cells, osteoblasts (OBs) and
              osteoclasts (OCs).

              The found results do not align with those presented in the paper. This
              difference may be because of potential misinterpretations or errors in
              the formulas utilized. However, as the paper has not disclosed its code,
              direct comparisons cannot be made to determine the exact differences.
              Because the results are different I wrote my interpretation of some
              formulas in this code.


Sartakhti, J. S., Manshaei, M. H., & Archetti, M. (2018). Game Theory of tumour–Stroma
Interactions in Multiple Myeloma: Effect of nonlinear benefits. Games, 9(2), 32.
https://doi.org/10.3390/g9020032
"""

# Import the needed libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ternary
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy.integrate import odeint
import os
import doctest

def main():
    # Do doc tests
    doctest.testmod()

    # # Make Figure 1 in the paper of Sartakhti et al., 2018.
    # Figure_1()
    #
    # # Make the Figures in Figure 2 in the paper of Sartakhti et al., 2018.
    # Figure_2()
    #
    # # # Make the Figures in Figure 3 in the paper of Sartakhti et al., 2018.
    # Figure_3()
    #
    # # Make the Figures in Figure 4 in the paper of Sartakhti et al., 2018.
    Figure_4()
    #
    # # Make the Figures in Figure 5 in the paper of Sartakhti et al., 2018.
    # Figure_5()
    #
    # # Make the Figures in Figure 6 in the paper of Sartakhti et al., 2018.
    # Figure_6()
    #
    # # Make the Figures in Figure 7 in the paper of Sartakhti et al., 2018.
    # Figure_7()

    # Make the Figures in Figure 8 in the paper of Sartakhti et al., 2018.
    Figure_8()

    # Make the Figures in Figure 9 in the paper of Sartakhti et al., 2018.
    Figure_9()

    # Make Figure 10 in the paper of Sartakhti et al., 2018.
    Figure_10()

"""
x_OC = fraction osteoclasten
x_OB = fraction osteoblasen
x_MM = fraction multiplemyeloma cells
x_OC + x_OB + x_MM = 1

N = Number of cells within the difusie range of the difusible factors
"""

def probability_number_cells(nOC, nOB, N, xOC, xOB, xMM):
    """ Function that calulates the probability that a group of cells contains
    specific numbers of OC (nOC), OB (nOB), and MM (N - nOC - nOB) cells (1).

    Parameters:
    -----------
    nOC: Int
        The number of osteoclasts in the population.
    nOB: Int
        The Number of osteoblasts in the population.
    N: Int
        The total number of cells in the group excluding the focal cell itself.
    xOC: Float
        The fraction of osteoclasts in the population.
    xOB: Float
        The fraction of osteoblasts in the population.
    xMM: Float
        The fraction of multiple myeloma cells in the population.

    Returns:
    -----------
    probability: Float
        Probability that population contains nOC, nOB and N - nOC - nOB MM cells.

    Example:
    -----------
    >>> probability_number_cells(2, 3, 10, 0.3, 0.4, 0.3)
    0.05878656000000001
    """
    # Number of ways to choose nOC OC cells and nOB OB cells from a total of N−1 cells
    combination_part = math.factorial(N - 1)/ (math.factorial(nOC) * math.factorial(nOB) \
                                                    * math.factorial(N - 1 - nOC - nOB))

    # Probability of having nOC osteoclasts, nOB osteoblast and N - nOB - nOC - 1
    # multiple myeloma cells
    probability_part = (xOC**nOC) * (xOB**nOB) * (xMM**(N - nOB - nOC - 1))

    # Calculate the final probability
    probability = combination_part * probability_part # (1)

    return probability

"""
Payoff is the benefit that a cell receives through their actions. For osteoclasts
(OC), osteoblasts (OB), and multiple myeloma cells (MM), the payoffs are calculated
based on the number of cells of each type, the effects of the diffusible factors
and the factor production costs.

VOC= bOC,OC(nOC+1)+ bOB,OC(nOB)+ bMM,OC(N−1−nOC−nOB)−cOC
- Positive terms: the effects of diffusible factors
- Negative term: the cost of producing diffusible factors
​"""

def payoff_OC(nOC, nOB, N, bOC_OC, bOB_OC, bMM_OC, cOC):
    """Function that calculates the payoff for osteoclasts (2).

     Parameters:
    -----------
    nOC: Int
        The number of osteoclasts in population.
    nOB: Int
        The Number of osteoblasts in the population.
    N: Int
        The total number of cells in the group excluding the focal cell itself.
    bOC_OC: Float
        The benefit on a OC of the diffusible factors produced by an OC.
    bOB_OC: Float
        The benefit on a OC of the diffusible factors produced by an OB.
    bMM_OC: Float
        The benefit on a OC of the diffusible factors produced by an MM cell.
    cOC: Float
        The cost of producing diffusible factors by OC.

    Returns:
    -----------
    VOC: Float
        Payoff for osteoclasts.

    Example:
    -----------
    >>> payoff_OC(2, 3, 15, 0.1, 0.2, 0.3, 0.4)
    3.1999999999999997
    """
    VOC = (bOC_OC * (nOC + 1)) + (bOB_OC * nOB) + (bMM_OC * (N - 1 - nOC - nOB)) \
                                                                        - cOC #(2)

    return VOC

def payoff_OB(nOC, nOB, N, bOC_OB, bOB_OB, bMM_OB, cOB):
    """Function that calculates the payoff for osteoblasts (3).

     Parameters:
    -----------
    nOC: Int
        The number of osteoclasts in population.
    nOB: Int
        The Number of osteoblasts in the population.
    N: Int
        The total number of cells in the group excluding the focal cell itself.
    bOC_OB: Float
        The benefit on a OB of the diffusible factors produced by an OC.
    bOB_OB: Float
        The benefit on a OB of the diffusible factors produced by an OB.
    bMM_OB: Float
        The benefit on a OB of the diffusible factors produced by an MM cell.
    cOB: Float
        The cost of producing diffusible factors by OB.

    Returns:
    -----------
    VOB: Float
        Payoff for osteoblasts.

    Example:
    -----------
    >>> payoff_OB(2, 3, 15, 0.1, 0.2, 0.3, 0.4)
    3.3
    """
    VOB = (bOC_OB * nOC) + (bOB_OB * (nOB + 1)) + (bMM_OB * (N - 1 - nOC - nOB)) \
                                                                        - cOB #(3)

    return VOB

def payoff_MM(nOC, nOB, N, bOC_MM, bOB_MM, bMM_MM, cMM):
    """Function that calculates the payoff for multiple myeloma cells (4).

    Parameters:
    -----------
    nOC: Int
        The number of osteoclasts in population.
    nOB: Int
        The Number of osteoblasts in the population.
    N: Int
        The total number of cells in the group excluding the focal cell itself.
    bOC_MM: Float
        The benefit on a MM cell of the diffusible factors produced by an OC.
    bOB_MM: Float
        The benefit on a MM cell of the diffusible factors produced by an OB.
    bMM_MM: Float
        The benefit on a MM cell of the diffusible factors produced by an MM cell.
    cMM: Float
        The cost of producing diffusible factors by multiple myeloma cells

    Returns:
    -----------
    VMM: Float
        Payoff for multiple myeloma cells.

    Example:
    -----------
    >>> payoff_MM(2, 3, 15, 0.1, 0.2, 0.3, 0.4)
    3.4
    """
    VMM = (bOC_MM * nOC) + (bOB_MM * nOB) + (bMM_MM * (N - nOC - nOB)) - cMM #(4)
    return VMM

"""
Fitness (Wi) is determined by multiplying the payoffs obtained by each cell type
in the formed population with the probability that population with those cell type
fractions occurs. The outer sum iterates over values of nOC and the inner sum
iterates over values of nOB. The fitness values are normalized using N/(N-1).
"""

def calculate_fitness(N, xOC, xOB, xMM, bOC_OC, bOB_OC, bMM_OC, cOC, bOC_OB, bOB_OB,
                                        bMM_OB, cOB, bOC_MM, bOB_MM, bMM_MM, cMM):
    """ Function that calculates the fitness of the osteoblasts, osteoclasts and
    multiple myeloma cells (5).

    Parameters:
    -----------
    N: Int
       The total number of cells in the group excluding the focal cell itself.
    xOC: Float
       The fraction of osteoclasts in the population.
    xOB: Float
       The fraction of osteoblasts in the population.
    xMM: Float
       The fraction of multiple myeloma cells in the population.
    bOC_OC: Float
       The benefit on a OC of the diffusible factors produced by an OC.
    bOB_OC: Float
       The benefit on a OC of the diffusible factors produced by an OB.
    bMM_OC: Float
       The benefit on a OC of the diffusible factors produced by an MM cell.
    cOC: Float
       The cost of producing diffusible factors by OC.
    bOC_OB: Float
       The benefit on a OB of the diffusible factors produced by an OC.
    bOB_OB: Float
       The benefit on a OB of the diffusible factors produced by an OB.
    bMM_OB: Float
       The benefit on a OB of the diffusible factors produced by an MM cell.
    cOB: Float
       The cost of producing diffusible factors by OB.
    bOC_MM: Float
       The benefit on an MM cell of the diffusible factors produced by an OC.
    bOB_MM: Float
       The benefit on an MM cell of the diffusible factors produced by an OB.
    bMM_MM: Float
       The benefit on an MM cell of the diffusible factors produced by an MM cell.
    cMM: Float
       The cost of producing diffusible factors by MM cells.

    Returns:
    -----------
    normalized_fitness_OC: Float
        The normalized fitness of the osteoclasts.
    normalized_fitness_OB: Float
        The normalized fitness of the osteoblasts.
    normalized_fitness_MM: Float
        The normalized fitness of the multiple myeloma.

    Example:
    -----------
    >>> calculate_fitness(10, 0.3, 0.4, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
    ... 0.8, 0.9, 1.0, 1.1, 1.2)
    (0.15821162519999998, 0.5527329200999999, 0.9472542150000003)
    """
    fitness_OC = 0
    fitness_OB = 0
    fitness_MM = 0

    # Loop over the range of nOC values. (-1 is left out of the range because then the
    # range goes to N-1 if you have range(1, N-1) Then the range goes to N-2)
    for nOC in range(1, N):

        # Loop over the range of nOB values
        for nOB in range(0, N- nOC):

            # Calculate the probability of nOC, nOB and (N-nOC-nOB) MM cells
            probability_value = probability_number_cells(nOC, nOB, N, xOC, xOB, xMM)

            # Determine the fitness of the OC, OB and MM cells
            payoff_OC_value = payoff_OC(nOC, nOB, N, bOC_OC, bOB_OC, bMM_OC, cOC)
            fitness_OC += probability_value * (payoff_OC_value)
            payoff_OB_value = payoff_OB(nOC, nOB, N, bOC_OB, bOB_OB, bMM_OB, cOB)
            fitness_OB += probability_value * (payoff_OB_value)
            payoff_MM_value = payoff_MM(nOC, nOB, N, bOC_MM, bOB_MM, bMM_MM, cMM)
            fitness_MM += probability_value * (payoff_MM_value)

    # Normalize the fitness values
    normalization_factor = 1/ (N-1)
    normalized_fitness_OC = normalization_factor * fitness_OC
    normalized_fitness_OB = normalization_factor * fitness_OB
    normalized_fitness_MM = normalization_factor * fitness_MM

    return normalized_fitness_OC, normalized_fitness_OB, normalized_fitness_MM

"""
Replicator dynamics says that cells with a higher fitness will increase in fraction
over time, while those with lower fitness will decrease. W* represents the average
fitness in the population.
"""

def calculate_replicator_dynamics(xOC, xOB, xMM, WOC, WOB, WMM):
    """ Function that calculates the fraction of osteoblasts, osteoclasts and
    multiple myeloma cells in the next generation based on replicator dynamics.

    Parameters:
    -----------
    xOC: Float
        The fraction of osteoclasts in the population.
    xOB: Float
        The fraction of osteoblasts in the population.
    xMM: Float
        The fraction of multiple myeloma cells in the population.
    WOC: Float
        The fitness of the osteoclasts.
    WOB: Float
        The fitness of the osteoblasts.
    WMM: Float
        The fitness of the multiple myeloma cells.

    Returns:
    -----------
    xOC_change: Float
         The change in fraction of osteoclasts in the population.
    xOB_change: Float
         The change infraction of osteoblasts in the population.
    xMM_change: Float
         The change in fraction of multiple myeloma cells in the population.
    W_average: Float
        The average fitness.

    Example:
    -----------
    >>> calculate_replicator_dynamics(0.3, 0.4, 0.3, 0.1, 0.2, 0.3)
    (-0.03, 0.0, 0.029999999999999992, 0.2)
    """
    # Determine the average fittness
    W_average = xOC * WOC + xOB * WOB + xMM * WMM


    # Determine the new fractions based of replicator dynamics
    xOC_change = xOC * (WOC - W_average) # (6)
    xOB_change = xOB * (WOB - W_average) # (7)
    xMM_change = xMM * (WMM - W_average) # (8)

    return xOC_change, xOB_change, xMM_change, W_average

"""
The benefit function gives the benefit of the diffusible factors of cell type i on cell
type j. The more cells of type i (higher n) the higher the benefit becaues more
diffusible factors (10).
"""

def sigmoid(n_i, h, B_max, s, N):
    """ Functionthat calculates the sigmoid value.

    Parameters:
    -----------
    n_i: Int
        The number of cells of type i.
    h: Float
        The position of the inflection point.
    B_max: Float
        The maximum benefit.
    s: Float
        The  steepness of the function.
    N: Int
        The total number of cell in the group.

    Returns:
    -----------
    sigmoid_value: Float
        The output of the sigmoid function.

    Example:
    -----------
    >>> sigmoid(2, 0.5, 10, 2, 20)
    3.1002551887238754
    """
    sigmoid_value = B_max / (1 + np.exp(s * (h - n_i/ N)))
    return sigmoid_value

def benefit_function(n_i, h, B_max, s, N):
    """ Function that calculates the benefit value of the diffusible factors
    produced by cell type i on cell type j (9).

    Parameters:
    -----------
    n_i: Int
        The number of cells of type i.
    h: Float
        The position of the inflection point.
    B_max: Float
        The maximum benefit.
    s: Float
        The steepness of the function.
    N: Int
        The total number of cell in the group.

    Returns:
    -----------
    benefit_value: Float
        Value that indicates the effect of the diffusible factors produced by cell
        type i on cell type j

    Example:
    -----------
    >>> benefit_function(4, 0.5, 10, 2, 20)
    0.18480653891012727
    """
    # Avoid deviding by zero
    if B_max == 0:
        benefit_value = 1
    else:
        benefit_value = (sigmoid(n_i, h, B_max, s, N) - sigmoid(0, h, B_max, s, N)) / \
                            (sigmoid(N, h, B_max, s, N) - sigmoid(0, h, B_max, s, N))


    # If the benefit value is nan set it to zero
    if math.isnan(benefit_value):
        benefit_value = 1

    return benefit_value

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

def collect_data(file_name, folder_path):
    """ Function that reads the data from a csv file to a dataframe.

    Parameters:
    -----------
    file_name: String
        The name of the csv file.
    folder_path: String:
        Path to the folder where the data will be saved.

    Returns:
    --------
    data_frame: DataFrame
        The data frame contain the collected data.
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    data_frame = pd.read_csv(file_path)

    return data_frame

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

def save_ternary(Figure, file_name, folder_path):
    """Save the ternary plot in a specific folder.

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
    pio.write_image(Figure, os.path.join(folder_path, f'{file_name}.png'),
                                                                    format='png')

def dynamics_same_h_and_s(y, t, parameters):
    """Determines the fracuenty dynamics in a population over time. The h value and
    s value are for all interactions the same.

    Parameters:
    -----------
    y: List
        List containing the current fractions of the OCs, OBs and MM cells
    t: Numpy.ndarray
        Array with the time points.
    parameters: tuple
        Tuple containing parameters required for computation -> (N, h, s, BOC_OC,
        BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, cOC_value,
        cOB_value, cMM_value)

    Returns:
    -----------
    [xOC_change, xOB_change, xMM_change]: List
        List with the calculated change in fractions of xOC, xOB and MM cells
    """
    xOC, xOB, xMM = y
    N, h, s, BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM,\
                                        cOC_value, cOB_value, cMM_value = parameters

    # Determine the absolute cell type numbers
    nOC = xOC * N
    nOB = xOB * N
    nMM = xMM * N

    # Calculate the benefit values
    bOC_OC = benefit_function(nOC, h, BOC_OC, s, N)
    bOB_OC = benefit_function(nOB, h, BOB_OC, s, N)
    bMM_OC = benefit_function(nMM, h, BMM_OC, s, N)

    bOC_OB = benefit_function(nOC, h, BOC_OB, s, N)
    bOB_OB = benefit_function(nOB, h, BOB_OB, s, N)
    bMM_OB = benefit_function(nMM, h, BMM_OB, s, N)

    bOC_MM = benefit_function(nOC, h, BOC_MM, s, N)
    bOB_MM = benefit_function(nOB, h, BOB_MM, s, N)
    bMM_MM = benefit_function(nMM, h, BMM_MM, s, N)

    # Determine the fitness values
    fitness_OC, fitness_OB, fitness_MM = calculate_fitness(N, xOC, xOB, xMM, bOC_OC,
                                bOB_OC, bMM_OC, cOC_value, bOC_OB, bOB_OB, bMM_OB,
                                    cOB_value, bOC_MM, bOB_MM, bMM_MM, cMM_value)

    # Determine the change of the xOC, xOB, xMM values and W average value
    xOC_change, xOB_change, xMM_change, W_average = calculate_replicator_dynamics(
                                xOC, xOB, xMM, fitness_OC, fitness_OB, fitness_MM)

    return [xOC_change, xOB_change, xMM_change]

def dynamics_different_h_and_s(y, t, parameters):
    """
    Simulate the dynamics of a population with three strategies over time. The s and
    h value is deppendent on the interaction kind.

    Parameters:
    -----------
    y: List
        List containing the current fractions of the OCs, OBs and MM cells
    t: Numpy.ndarray
        Array with the time points.
    parameters: tuple
        Tuple containing parameters required for computation -> (NN, hOC_OC, hOC_OB,
        hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB, hMM_MM, sOC_OC, sOC_OB,
        sOC_MM, sOB_OC, sOB_OB, sOB_MM, sMM_OC, sMM_OB, sMM_MM, BOC_OC, BOB_OC,
        BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, cOC_value, cOB_value,
        cMM_value)

    Returns:
    -----------
    [xOC_change, xOB_change, xMM_change]: List
        List with the calculated change in fractions of xOC, xOB and MM cells
    """
    # Unpack state variables and parameters
    xOC, xOB, xMM = y
    N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB, hMM_MM, \
    sOC_OC, sOC_OB, sOC_MM, sOB_OC, sOB_OB, sOB_MM, sMM_OC, sMM_OB, sMM_MM, \
    BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, \
    cOC_value, cOB_value, cMM_value = parameters

    # Determine the absolute cell type numbers
    nOC = xOC * N
    nOB = xOB * N
    nMM = xMM * N

    # Calculate benefit values for each interaction
    bOC_OC = benefit_function(nOC, hOC_OC, BOC_OC, sOC_OC, N)
    bOB_OC = benefit_function(nOB, hOB_OC, BOB_OC, sOB_OC, N)
    bMM_OC = benefit_function(nMM, hMM_OC, BMM_OC, sMM_OC, N)

    bOC_OB = benefit_function(nOC, hOC_OB, BOC_OB, sOC_OB, N)
    bOB_OB = benefit_function(nOB, hOB_OB, BOB_OB, sOB_OB, N)
    bMM_OB = benefit_function(nMM, hMM_OB, BMM_OB, sMM_OB, N)

    bOC_MM = benefit_function(nOC, hOC_MM, BOC_MM, sOC_MM, N)
    bOB_MM = benefit_function(nOB, hOB_MM, BOB_MM, sOB_MM, N)
    bMM_MM = benefit_function(nMM, hMM_MM, BMM_MM, sMM_MM, N)

    # Determine fitness values for each strategy
    fitness_OC, fitness_OB, fitness_MM = calculate_fitness(N, xOC, xOB, xMM,
                                bOC_OC, bOB_OC, bMM_OC, cOC_value, bOC_OB, bOB_OB,
                                 bMM_OB, cOB_value, bOC_MM, bOB_MM, bMM_MM, cMM_value)

    # Determine changes in strategy fractions
    xOC_change, xOB_change, xMM_change, _ = calculate_replicator_dynamics(
                                xOC, xOB, xMM, fitness_OC, fitness_OB, fitness_MM)

    return [xOC_change, xOB_change, xMM_change]

"""Figure 1"""
def Figure_1():
    """Function that recreates Figure 1 in the paper of Sartakhti et al., 2018."""
    # Number of cells
    N = 10

    # Cost of producing diffusible factors
    cOC_value = 0.1
    cOB_value = 0.2
    cMM_value = 0.3

    # Maximal benefit values
    BOC_OC = 0.0
    BOC_OB = 1.0
    BOC_MM = 1.1
    BOB_OC = 1.0
    BOB_OB = 0.0
    BOB_MM = 0.0
    BMM_OC = 1.1
    BMM_OB = -0.3
    BMM_MM = 0.0

    # Steepness and inflection point
    s = 1e-10
    h = 0.7

    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.4
    xOB = 0.3
    xMM = 0.3

    nOC = 1
    nOB = 3
    nMM = 6

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, h, s, BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM,\
                                BOB_MM, BMM_MM, cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 300, 300)

    # Solve ODE
    y = odeint(dynamics_same_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_1 = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                        'xOB': xOB_values, 'xMM': xMM_values})

    # Save the data as csv file
    save_data(df_Figure_1, 'data_Figure_1.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Make lists
    WOC_list = []
    WOB_list = []
    WMM_list = []
    W_average_list = []
    generation_list = []

    # Iterate over each row
    for index, row in df_Figure_1.iterrows():
        # Extract values of xOC, xOB, and xMM for the current row
        xOC = row['xOC']
        xOB = row['xOB']
        xMM = row['xMM']

        nOC = xOC * N
        nOB = xOB * N
        nMM = xMM * N

        # Calculate the benefit values
        bOC_OC = benefit_function(nOC, h, BOC_OC, s, N)
        bOB_OC = benefit_function(nOB, h, BOB_OC, s, N)
        bMM_OC = benefit_function(nMM, h, BMM_OC, s, N)

        bOC_OB = benefit_function(nOC, h, BOC_OB, s, N)
        bOB_OB = benefit_function(nOB, h, BOB_OB, s, N)
        bMM_OB = benefit_function(nMM, h, BMM_OB, s, N)

        bOC_MM = benefit_function(nOC, h, BOC_MM, s, N)
        bOB_MM = benefit_function(nOB, h, BOB_MM, s, N)
        bMM_MM = benefit_function(nMM, h, BMM_MM, s, N)

        # Determine the fitness values
        fitness_OC, fitness_OB, fitness_MM = calculate_fitness(N, xOC, xOB, xMM, bOC_OC,
                                    bOB_OC, bMM_OC, cOC_value, bOC_OB, bOB_OB, bMM_OB,
                                        cOB_value, bOC_MM, bOB_MM, bMM_MM, cMM_value)

        # Calculate the average fitness
        W_average = xOC * fitness_OC + xOB * fitness_OB + xMM * fitness_MM

        # Append the calculated values to the respective lists
        WOC_list.append(fitness_OC)
        WOB_list.append(fitness_OB)
        WMM_list.append(fitness_MM)
        W_average_list.append(W_average)
        generation_list.append(index)

    # Create a new DataFrame with the calculated values
    df_fitness = pd.DataFrame({'Generation': generation_list, 'WOC': WOC_list,
                'WOB': WOB_list, 'WMM': WMM_list, 'W_average': W_average_list})

    # Create a Figure and axes for subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

    # Plot the first subplot
    df_fitness.plot(x='Generation', y=['WOC', 'WOB', 'WMM', 'W_average'], ax=axes[0])
    axes[0].set_title('Fitness linear benfits (Figure 1)')
    axes[0].set_xlabel('Generations')
    axes[0].set_ylabel('Fitness')
    axes[0].legend(['Fitness OC', 'Fitness OB', 'Fitness MM', 'Average fitness'],
                                                            loc = 'upper left')

    # Plot the second subplot
    df_Figure_1.plot(x='Generation', y=['xOC', 'xOB', 'xMM'], ax=axes[1])
    axes[1].set_title('Dynamics linear benefits (Figure 1)')
    axes[1].set_xlabel('Generations')
    axes[1].set_ylabel('Fraction')
    axes[1].legend(['fraction OC', 'fraction OB', 'fraction MM'], loc = 'upper left')
    plt.tight_layout()
    save_Figure(plt, 'Line_plot_Figure_1',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot
    fig = px.line_ternary(df_Figure_1, a='xOB', b='xMM', c='xOC')

    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text='Linear benefits (Figure 1)')
    save_ternary(fig, 'Ternary_plot_Figure_1',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

"""Figure 2"""
def Figure_2():
    """Function that recreates the Figures of Figure 2 in the paper of Sartakhti
    et al., 2018."""
    # Number of cells
    N = 10

    # Cost of producing diffusible factors
    cOC_value = 0.1
    cOB_value = 0.2
    cMM_value = 0.3

    # Maximal benefit values
    BOC_OC = 0
    BOC_OB = 1.0
    BOC_MM = 1.1
    BOB_OC = 1
    BOB_OB = 0
    BOB_MM = 0
    BMM_OC = 1.1
    BMM_OB = -0.3
    BMM_MM = 0

    # The inflection points
    h_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Steepness of the function and a random maximal benefit for the demostration
    s_value = 20
    B_value = 1

    # Create a DataFrame to store the data
    df_sigmoides_Figure_2 = pd.DataFrame(columns=['n_values', 'benefit_values',
                                                                        'h_value'])
    # Loop over h values
    for h_value in h_values:
        n_values = np.linspace(0, N, 100)
        benefit_values = [benefit_function(n, h_value, B_value, s_value,
                                                            N) for n in n_values]

        # Add the data to the dataframe
        df_sigmoides_Figure_2 = pd.concat([df_sigmoides_Figure_2, pd.DataFrame({
        'n_values': n_values, 'benefit_values': benefit_values, 'h_value': h_value})])

    # Save the data as csv file
    save_data(df_sigmoides_Figure_2, 'data_sigmoides_Figure_2.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Make a plot
    fig, axes = plt.subplots(1, len(h_values), figsize=(14, 5))
    for i, (h_value, group) in enumerate(df_sigmoides_Figure_2.groupby('h_value')):
        axes[i].plot(group['n_values'], group['benefit_values'], label=f'h={h_value}')

        # Give titles
        axes[i].set_title(f'Sigmoide benefit h={h_value}')
        axes[i].set_xlabel('Number of producers')
        axes[i].set_ylabel('Benefit')
        axes[i].set_xticks([0, 10])
        axes[i].set_xticklabels(['0', 'N'], fontsize=11)
        axes[i].set_yticks([0, 1])
        axes[i].set_yticklabels(['0', r'$B_{ij}$'], fontsize=11)

    # Show the plot
    plt.tight_layout()
    save_Figure(plt, 'Benefit_curves_Figure_2',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    df_ternary_Figure_2 = pd.DataFrame(columns=['Generation', 'xOC', 'xOB', 'xMM',
                                                                        'h_value'])

    # Cost of producing diffusible factors
    cOC_value = 0.1
    cOB_value = 0.2
    cMM_value = 0.3

    # Maximal benefit values
    BOC_OC = 0
    BOC_OB = 1.0
    BOC_MM = 1.1
    BOB_OC = 1
    BOB_OB = 0
    BOB_MM = 0
    BMM_OC = 1.1
    BMM_OB = -0.3
    BMM_MM = 0

    # Loop over the inflection point values
    for h_value in h_values:

        # Reset initial values for each h iteration
        xOC = 0.3
        xOB = 0.3
        xMM = 0.4
        N = 10
        nOC = 3
        nOB = 3
        nMM = 4


        # Steepness of the function at the inflection point
        s_value = 20
        generations = 100

        # Set initial condition and parameters
        y0 = [xOC, xOB, xMM]
        parameters = (N, h_value, s_value, BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB,
                BMM_OB, BOC_MM, BOB_MM, BMM_MM, cOC_value, cOB_value, cMM_value)
        t = np.linspace(0, 200)

        # Solve ODE
        y = odeint(dynamics_same_h_and_s, y0, t, args=(parameters,))

        # Extract the solution and create dataframe
        xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
        df = pd.DataFrame({'Generation': t, 'xOC': xOC_values, 'xOB': xOB_values,
                                                                'xMM': xMM_values})
        df['h_value'] = h_value
        df_ternary_Figure_2 = pd.concat([df, df_ternary_Figure_2], ignore_index=True)

    # Save the data as csv file
    save_data(df_ternary_Figure_2, 'data_ternary_Figure_2.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Loop over all the h values
    for h_value in h_values:
        # Create a subset DataFrame for the current h_value
        subset_df = df_ternary_Figure_2[df_ternary_Figure_2['h_value'] == h_value]
        fig = px.line_ternary(subset_df, a='xOB', b='xMM', c='xOC')
        fig.update_layout(
            ternary=dict(
                aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
                baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
                caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
        fig.update_layout(title_text=f"""Sigmoide benefits with an inflection point
        h at {h_value} (Figure 2)""")
        name = f'subset_plot_h_{h_value}_Figure_2'
        save_ternary(fig, name,
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
        fig.show()


""" Figure 3"""
def Figure_3():
    """Function that recreates the Figures of Figure 3 in the paper of Sartakhti
    et al., 2018."""
    # Number of cells
    N = 25

    # Cost of producing diffusible factors
    cOC_value = 0.1
    cOB_value = 0.1
    cMM_value = 0.1

    # Maximal benefit values
    BOC_OC = 0.6
    BOB_OC = 1.0
    BMM_OC = 3.0
    BOC_OB = 1.1
    BOB_OB = 0.6
    BMM_OB = -0.5
    BOC_MM = 2.0
    BOB_MM = 0.0
    BMM_MM = 1.0

    # Positions of the inflection points
    hOC_OC = 0.0
    hOC_OB = 0.01
    hOC_MM = 0.2
    hOB_OC = 0.05
    hOB_OB = 0.05
    hOB_MM = 0.2
    hMM_OC = 0.5
    hMM_OB = 0.5
    hMM_MM = 0.5

    # Steepness of the function at the inflection points
    sOC_OC = 50
    sOC_OB = 30
    sOC_MM = 50
    sOB_OC = 30
    sOB_OB = 30
    sOB_MM = 30
    sMM_OC = 5
    sMM_OB = 20
    sMM_MM = 50

    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.3
    xOB = 0.2
    xMM = 0.5
    nOC = 5
    nOB = 5
    nMM = 15

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,\
    hMM_MM, sOC_OC, sOC_OB, sOC_MM, sOB_OC, sOB_OB, sOB_MM, sMM_OC, sMM_OB, sMM_MM, \
    BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, \
    cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 100, 200)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_3_nonlinear = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                            'xOB': xOB_values, 'xMM': xMM_values})

    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.3
    xOB = 0.2
    xMM = 0.5
    nOC = 5
    nOB = 5
    nMM = 15

    s_linear = 1e-10

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,
    hMM_MM, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear,
    s_linear, s_linear, BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM,
    BOB_MM, BMM_MM, cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 100, 100)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_3_linear = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                        'xOB': xOB_values, 'xMM': xMM_values})

    # Save the data as csv file
    save_data(df_Figure_3_nonlinear, 'data_Figure_3_nonlinear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')
    save_data(df_Figure_3_linear, 'data_Figure_3_linear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Make a line plot of non-linear data
    df_Figure_3_nonlinear.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('Nonlinear benefits (Figure 3)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_3_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot of non-linear data
    fig = px.line_ternary(df_Figure_3_nonlinear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text=f'Nonlinear benefits (Figure 3)')
    save_ternary(fig, 'Ternary_plot_Figure_3_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

    # Make a line plot of linear data
    df_Figure_3_linear.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                    label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.legend(loc ='upper right')
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('Linear benfits (Figure 3)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_3_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot  of linear data
    fig = px.line_ternary(df_Figure_3_linear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text= 'Linear benefits (Figure 3)')
    save_ternary(fig, 'Ternary_plot_Figure_3_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

"""Figure 4"""
def Figure_4():
    """Function that recreates the Figures of Figure 4 in the paper of Sartakhti
    et al., 2018."""
    # Number of cells
    N = 20

    # Cost of producing diffusible factors
    cOC_value = 1.5
    cOB_value = 0.5
    cMM_value = 2.0

    # Maximal benefit values
    BOC_OC = 0.6
    BOB_OC = 1.0
    BMM_OC = 3.0
    BOC_OB = 1.1
    BOB_OB = 0.5
    BMM_OB = -0.5
    BOC_MM = 2.3
    BOB_MM = 0.0
    BMM_MM = 1.5

    # Positions of the inflection points
    hOC_OC = 0.0
    hOC_OB = 0.0
    hOC_MM = 0.0
    hOB_OC = 0.0
    hOB_OB = 0.0
    hOB_MM = 0.0
    hMM_OC = 0.3
    hMM_OB = 0.5
    hMM_MM = 0.1

    # Steepness of the function at the inflection points
    sOC_OC = 10
    sOC_OB = 10
    sOC_MM = 100
    sOB_OC = 10
    sOB_OB = 10
    sOB_MM = 20
    sMM_OC = 10
    sMM_OB = 10
    sMM_MM = 100

    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.2
    xOB = 0.5
    xMM = 0.3
    nOC = 4
    nOB = 10
    nMM = 6

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,\
    hMM_MM, sOC_OC, sOC_OB, sOC_MM, sOB_OC, sOB_OB, sOB_MM, sMM_OC, sMM_OB, sMM_MM, \
    BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, cOC_value,\
    cOB_value, cMM_value)
    t = np.linspace(0, 500, 500)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_4_nonlinear = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                            'xOB': xOB_values, 'xMM': xMM_values})

    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.2
    xOB = 0.5
    xMM = 0.3
    nOC = 4
    nOB = 10
    nMM = 6

    s_linear = 10e-10

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,\
    hMM_MM, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear,\
    s_linear, s_linear, BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, \
    BOB_MM, BMM_MM, cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 2000, 2000)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_4_linear = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                            'xOB': xOB_values, 'xMM': xMM_values})


    # Save the data as csv file
    save_data(df_Figure_4_nonlinear, 'data_Figure_4_nonlinear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')
    save_data(df_Figure_4_linear, 'data_Figure_4_linear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Make a line plot of non-linear data
    df_Figure_4_nonlinear.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                    label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('Nonlinear benefits (Figure 4)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_4_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot of non-linear data
    fig = px.line_ternary(df_Figure_4_nonlinear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text= 'Nonlinear benefits (Figure 4)')
    save_ternary(fig, 'Ternary_plot_Figure_4_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

    # Make a line plot of linear data
    df_Figure_4_linear.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('Linear benefits (Figure 4)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_4_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot  of linear data
    fig = px.line_ternary(df_Figure_4_linear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text= 'Linear benefits (Figure 4)')
    save_ternary(fig, 'Ternary_plot_Figure_4_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

"""Figure 5"""
def Figure_5():
    """Function that recreates the Figures of Figure 5 in the paper of Sartakhti
    et al., 2018."""
    # Number of cells
    N = 20

    # Cost of producing diffusible factors
    cOC_value = 0.1
    cOB_value = 0.2
    cMM_value = 0.3

    # Maximal benefit values
    BOC_OC = 0.55
    BOB_OC = 1.0
    BMM_OC = 0.8
    BOC_OB = 1.1
    BOB_OB = 0.5
    BMM_OB = -0.5
    BOC_MM = 0.6
    BOB_MM = 0.0
    BMM_MM = 1.5

    # Positions of the inflection points
    hOC_OC = 0.0
    hOC_OB = 0.0
    hOC_MM = 0.0
    hOB_OC = 0.0
    hOB_OB = 0.0
    hOB_MM = 0.0
    hMM_OC = 0.3
    hMM_OB = 0.2
    hMM_MM = 0.1

    # Steepness of the function at the inflection points
    sOC_OC = 10
    sOC_OB = 10
    sOC_MM = 100
    sOB_OC = 10
    sOB_OB = 10
    sOB_MM = 20
    sMM_OC = 10
    sMM_OB = 10
    sMM_MM = 100

    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.2
    xOB = 0.5
    xMM = 0.3

    nOC = 4
    nOB = 10
    nMM = 6

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,
    hMM_MM, sOC_OC, sOC_OB, sOC_MM, sOB_OC, sOB_OB, sOB_MM, sMM_OC, sMM_OB, sMM_MM, \
    BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, \
    cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 100)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_5_nonlinear = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                            'xOB': xOB_values, 'xMM': xMM_values})

    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.2
    xOB = 0.5
    xMM = 0.3

    nOC = 4
    nOB = 10
    nMM = 6

    s_linear = 10e-10
    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,
    hMM_MM, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear,
    s_linear, s_linear, BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM,
    BOB_MM, BMM_MM, cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 100)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_5_linear = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                            'xOB': xOB_values, 'xMM': xMM_values})

    # Save the data as csv file
    save_data(df_Figure_5_nonlinear, 'data_Figure_5_nonlinear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')
    save_data(df_Figure_5_linear, 'data_Figure_5_linear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Make a line plot of non-linear df
    df_Figure_5_nonlinear.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('Nonlinear benefits (Figure 5)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_5_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot of non-linear data
    fig = px.line_ternary(df_Figure_5_nonlinear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text= 'Nonlinear benefits (Figure 5)')
    save_ternary(fig, 'Ternary_plot_Figure_5_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

    # Make a line plot of linear data
    df_Figure_5_linear.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                    label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.legend(loc ='upper right')
    plt.title('linear benefits (Figure 5)')
    save_Figure(plt, 'Line_plot_Figure_5_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot  of linear data
    fig = px.line_ternary(df_Figure_5_linear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text= 'Linear benefits (Figure 5)')
    save_ternary(fig, 'Ternary_plot_Figure_5_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

"""Figure 6"""
def Figure_6():
    """Function that recreates the Figures of Figure 6 in the paper of Sartakhti
    et al., 2018."""
    # Number of cells
    N = 20

    # Cost of producing diffusible factors
    cOC_value = 0.1
    cOB_value = 0.1
    cMM_value = 0.1

    # Maximal benefit values
    BOC_OC = 0.45
    BOB_OC = 0.9
    BMM_OC = 2.0
    BOC_OB = 0.9
    BOB_OB = 0.3
    BMM_OB = -0.3
    BOC_MM = 2.0
    BOB_MM = 0.0
    BMM_MM = 0.9

    # Positions of the inflection points
    hOC_OC = 0.05
    hOC_OB = 0.05
    hOC_MM = 0.5
    hOB_OC = 0.05
    hOB_OB = 0.05
    hOB_MM = 0.5
    hMM_OC = 0.5
    hMM_OB = 0.5
    hMM_MM = 0.5

    # Steepness of the function at the inflection points
    sOC_OC = 10
    sOC_OB = 10
    sOC_MM = 50
    sOB_OC = 10
    sOB_OB = 10
    sOB_MM = 10
    sMM_OC = 50
    sMM_OB = 50
    sMM_MM = 50

    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.2
    xOB = 0.5
    xMM = 0.3

    nOC = 4
    nOB = 10
    nMM = 6

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,
    hMM_MM, sOC_OC, sOC_OB, sOC_MM, sOB_OC, sOB_OB, sOB_MM, sMM_OC, sMM_OB, sMM_MM, \
    BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, \
    cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 200)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_6_nonlinear = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                            'xOB': xOB_values, 'xMM': xMM_values})

    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.2
    xOB = 0.5
    xMM = 0.3

    nOC = 4
    nOB = 10
    nMM = 6

    s_linear = 10e-10

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,
    hMM_MM, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear,
    s_linear, BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, \
    cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 200)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_6_linear = pd.DataFrame({'Generation': t, 'xOC': xOC_values, 'xOB':
                                                xOB_values, 'xMM': xMM_values})

    # Save the data as csv file
    save_data(df_Figure_6_nonlinear, 'data_Figure_6_nonlinear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')
    save_data(df_Figure_6_linear, 'data_Figure_6_linear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Make a line plot of non-linear data
    df_Figure_6_nonlinear.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('Nonlinear benefits (Figure 6)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_6_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot of non-linear data
    fig = px.line_ternary(df_Figure_6_nonlinear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text= 'Nonlinear benefits (Figure 6)')
    save_ternary(fig, 'Ternary_plot_Figure_6_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

    # Make a line plot of linear data
    df_Figure_6_linear.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('Linear benefits (Figure 6)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_6_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot  of linear data
    fig = px.line_ternary(df_Figure_6_linear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text= 'Linear benefits (Figure 6)')
    save_ternary(fig, 'Ternary_plot_Figure_6_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

"""Figure 7"""
def Figure_7():
    """Function that recreates the Figures of Figure 7 in the paper of Sartakhti
    et al., 2018."""
    # Number of cells
    N = 10

    # Cost of producing diffusible factors
    cOC_value = 0.1
    cOB_value = 0.12
    cMM_value = 0.14

    # Maximal benefit values
    BOC_OC = 1.0
    BOB_OC = 0.7
    BMM_OC = 0.9
    BOC_OB = 1.0
    BOB_OB = 0.7
    BMM_OB = 0.9
    BOC_MM = 1.0
    BOB_MM = 0.7
    BMM_MM = 0.9

    # Positions of the inflection points
    hOC_OC = 0.4
    hOC_OB = 0.7
    hOC_MM = 0.1
    hOB_OC = 0.7
    hOB_OB = 0.4
    hOB_MM = 0.2
    hMM_OC = 0.4
    hMM_OB = 0.3
    hMM_MM = 0.7

    # Steepness of the function at the inflection points
    sOC_OC = 20
    sOC_OB = 20
    sOC_MM = 5
    sOB_OC = 10
    sOB_OB = 10
    sOB_MM = 50
    sMM_OC = 10
    sMM_OB = 5
    sMM_MM = 5

    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.2
    xOB = 0.5
    xMM = 0.3

    nOC = 2
    nOB = 5
    nMM = 3

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,
    hMM_MM, sOC_OC, sOC_OB, sOC_MM, sOB_OC, sOB_OB, sOB_MM, sMM_OC, sMM_OB, sMM_MM, \
    BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, \
    cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 1000, 1000)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_7_nonlinear = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                            'xOB': xOB_values, 'xMM': xMM_values})

    # Make lists
    WOC_list = []
    WOB_list = []
    WMM_list = []
    W_average_list = []
    generation_list = []

    # Iterate over each row
    for index, row in df_Figure_7_nonlinear.iterrows():
        # Extract values of xOC, xOB, and xMM for the current row
        xOC = row['xOC']
        xOB = row['xOB']
        xMM = row['xMM']

        # Calculate the number of individuals following each strategy
        nOC = xOC * N
        nOB = xOB * N
        nMM = xMM * N

        # Calculate benefit values for each interaction
        bOC_OC = benefit_function(nOC, hOC_OC, BOC_OC, sOC_OC, N)
        bOB_OC = benefit_function(nOB, hOB_OC, BOB_OC, sOB_OC, N)
        bMM_OC = benefit_function(nMM, hMM_OC, BMM_OC, sMM_OC, N)

        bOC_OB = benefit_function(nOC, hOC_OB, BOC_OB, sOC_OB, N)
        bOB_OB = benefit_function(nOB, hOB_OB, BOB_OB, sOB_OB, N)
        bMM_OB = benefit_function(nMM, hMM_OB, BMM_OB, sMM_OB, N)

        bOC_MM = benefit_function(nOC, hOC_MM, BOC_MM, sOC_MM, N)
        bOB_MM = benefit_function(nOB, hOB_MM, BOB_MM, sOB_MM, N)
        bMM_MM = benefit_function(nMM, hMM_MM, BMM_MM, sMM_MM, N)

        # Determine fitness values for each strategy
        fitness_OC, fitness_OB, fitness_MM = calculate_fitness(N, xOC, xOB, xMM,
                                    bOC_OC, bOB_OC, bMM_OC, cOC_value, bOC_OB, bOB_OB,
                                     bMM_OB, cOB_value, bOC_MM, bOB_MM, bMM_MM, cMM_value)


        # Calculate the average fitness
        W_average = xOC * fitness_OC + xOB * fitness_OB + xMM * fitness_MM

        # Append the calculated values to the respective lists
        WOC_list.append(fitness_OC)
        WOB_list.append(fitness_OB)
        WMM_list.append(fitness_MM)
        W_average_list.append(W_average)
        generation_list.append(index)

    # Create a new DataFrame with the calculated values
    df_fitness_nonlinear = pd.DataFrame({'Generation': generation_list, 'WOC': \
        WOC_list, 'WOB': WOB_list, 'WMM': WMM_list, 'W_average': W_average_list})


    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.2
    xOB = 0.5
    xMM = 0.3

    nOC = 2
    nOB = 5
    nMM = 3

    s_linear = 10e-10

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,
    hMM_MM, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear,
    s_linear, BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, \
    cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 400, 400)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_7_linear = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                        'xOB': xOB_values, 'xMM': xMM_values})

    # Save the data as csv file
    save_data(df_Figure_7_nonlinear, 'data_Figure_7_nonlinear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')
    save_data(df_Figure_7_linear, 'data_Figure_7_linear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Create a Figure and axes for subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

    # Plot the first subplot
    df_fitness_nonlinear.plot(x='Generation', y=['WOC', 'WOB', 'WMM', 'W_average'],
                                                                        ax=axes[0])
    axes[0].set_title('Fitness nonlinear benfits (Figure 7)')
    axes[0].set_xlabel('Generations')
    axes[0].set_ylabel('Fitness')
    axes[0].legend(['Fitness OC', 'Fitness OB', 'Fitness MM', 'Average fitness'])

    # Plot the second subplot
    df_Figure_7_nonlinear.plot(x='Generation', y=['xOC', 'xOB', 'xMM'], ax=axes[1])
    axes[1].set_title('Dynamics nonlinear benfits (Figure 7)')
    axes[1].set_xlabel('Generations')
    axes[1].set_ylabel('Fraction')
    axes[1].legend(['fraction OC', 'fraction OB', 'fraction MM'])
    plt.tight_layout()
    save_Figure(plt, 'Line_plot_Figure_7_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()


    # Make a ternary plot of non-linear data
    fig = px.line_ternary(df_Figure_7_nonlinear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text= 'Nonlinear benefits (Figure 7)')
    save_ternary(fig, 'Ternary_plot_Figure_7_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

    # Make a line plot of linear data
    df_Figure_7_linear.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('Linear benefits (Figure 7)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_7_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot  of linear data
    fig = px.line_ternary(df_Figure_7_linear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text= 'Linear benefits (Figure 7)')
    save_ternary(fig, 'Ternary_plot_Figure_7_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

"""Figure 8"""
def Figure_8():
    """Function that recreates the Figures of Figure 8 in the paper of Sartakhti
    et al., 2018."""
    # Number of cells
    N = 20

    # Cost of producing diffusible factors
    cOC_value = 1.2
    cOB_value = 1.0
    cMM_value = 1.8

    # Maximal benefit values
    BOC_OC = 1.1
    BOB_OC = 0.95
    BMM_OC = 1.8
    BOC_OB = 1.1
    BOB_OB = 1.1
    BMM_OB = -0.35
    BOC_MM = 1.1
    BOB_MM = 1.5
    BMM_MM = 0.35

    # Positions of the inflection points
    hOC_OC = 0.0
    hOC_OB = 0.0
    hOC_MM = 0.0
    hOB_OC = 0.0
    hOB_OB = 0.0
    hOB_MM = 0.0
    hMM_OC = 0.2
    hMM_OB = 0.2
    hMM_MM = 0.2

    # Steepness of the function at the inflection points
    sOC_OC = 4
    sOC_OB = 4
    sOC_MM = 40
    sOB_OC = 4
    sOB_OB = 4
    sOB_MM = 4
    sMM_OC = 6
    sMM_OB = 6
    sMM_MM = 1000


    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.35
    xOB = 0.4
    xMM = 0.25

    nOC = 7
    nOB = 8
    nMM = 5

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,
    hMM_MM, sOC_OC, sOC_OB, sOC_MM, sOB_OC, sOB_OB, sOB_MM, sMM_OC, sMM_OB, sMM_MM, \
    BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, \
    cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 200, 200)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_8_nonlinear = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                            'xOB': xOB_values, 'xMM': xMM_values})

    # Make lists
    WOC_list = []
    WOB_list = []
    WMM_list = []
    W_average_list = []
    generation_list = []

    # Iterate over each row
    for index, row in df_Figure_8_nonlinear.iterrows():
        # Extract values of xOC, xOB, and xMM for the current row
        xOC = row['xOC']
        xOB = row['xOB']
        xMM = row['xMM']

        # Calculate the number of individuals following each strategy
        nOC = xOC * N
        nOB = xOB * N
        nMM = xMM * N

        # Calculate benefit values for each interaction
        bOC_OC = benefit_function(nOC, hOC_OC, BOC_OC, sOC_OC, N)
        bOB_OC = benefit_function(nOB, hOB_OC, BOB_OC, sOB_OC, N)
        bMM_OC = benefit_function(nMM, hMM_OC, BMM_OC, sMM_OC, N)

        bOC_OB = benefit_function(nOC, hOC_OB, BOC_OB, sOC_OB, N)
        bOB_OB = benefit_function(nOB, hOB_OB, BOB_OB, sOB_OB, N)
        bMM_OB = benefit_function(nMM, hMM_OB, BMM_OB, sMM_OB, N)

        bOC_MM = benefit_function(nOC, hOC_MM, BOC_MM, sOC_MM, N)
        bOB_MM = benefit_function(nOB, hOB_MM, BOB_MM, sOB_MM, N)
        bMM_MM = benefit_function(nMM, hMM_MM, BMM_MM, sMM_MM, N)

        # Determine fitness values for each strategy
        fitness_OC, fitness_OB, fitness_MM = calculate_fitness(N, xOC, xOB, xMM,
                                bOC_OC, bOB_OC, bMM_OC, cOC_value, bOC_OB, bOB_OB,
                                bMM_OB, cOB_value, bOC_MM, bOB_MM, bMM_MM, cMM_value)


        # Calculate the average fitness
        W_average = xOC * fitness_OC + xOB * fitness_OB + xMM * fitness_MM

        # Append the calculated values to the respective lists
        WOC_list.append(fitness_OC)
        WOB_list.append(fitness_OB)
        WMM_list.append(fitness_MM)
        W_average_list.append(W_average)
        generation_list.append(index)

    # Create a new DataFrame with the calculated values
    df_fitness_nonlinear = pd.DataFrame({'Generation': generation_list, 'WOC': \
        WOC_list, 'WOB': WOB_list, 'WMM': WMM_list, 'W_average': W_average_list})

    # Initial fractions and values --> are needed to make a plot but are not mentioned
    xOC = 0.35
    xOB = 0.4
    xMM = 0.25

    nOC = 7
    nOB = 8
    nMM = 5

    s_linear = 10e-10

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, hOC_OC, hOC_OB, hOC_MM, hOB_OC, hOB_OB, hOB_MM, hMM_OC, hMM_OB,
    hMM_MM, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear, s_linear,
    s_linear, BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, \
    cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 200, 200)

    # Solve ODE
    y = odeint(dynamics_different_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_8_linear = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                            'xOB': xOB_values, 'xMM': xMM_values})

    # Save the data as csv file
    save_data(df_Figure_8_nonlinear, 'data_Figure_8_nonlinear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')
    save_data(df_Figure_8_linear, 'data_Figure_8_linear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')
    save_data(df_fitness_nonlinear, 'data_fitness_Figure_8_linear.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Create a Figure and axes for subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

    # Plot the first subplot
    df_fitness_nonlinear.plot(x='Generation', y=['WOC', 'WOB', 'WMM', 'W_average'],
                                                                        ax=axes[0])
    axes[0].set_title('Fitness nonlinear benefits (Figure 8)')
    axes[0].set_xlabel('Generations')
    axes[0].set_ylabel('Fitness')
    axes[0].legend(['Fitness OC', 'Fitness OB', 'Fitness MM', 'Average fitness'])

    # Plot the second subplot
    df_Figure_8_nonlinear.plot(x='Generation', y=['xOC', 'xOB', 'xMM'], ax=axes[1])
    axes[1].set_title('Dynamics nonlinear benefits (Figure 8)')
    axes[1].set_xlabel('Generations')
    axes[1].set_ylabel('Fraction')
    axes[1].legend(['fraction OC', 'fraction OB', 'fraction MM'])
    plt.tight_layout()
    save_Figure(plt, 'Line_plot_Figure_8_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot of nonlinear data
    fig = px.line_ternary(df_Figure_8_nonlinear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text= 'Nonlinear benefits (Figure 8)')
    save_ternary(fig, 'Ternary_plot_Figure_8_nonlinear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

    # Make a line plot of linear data
    df_Figure_8_linear.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('Linear benefits (Figure 8)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_8_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot of linear data
    fig = px.line_ternary(df_Figure_8_linear, a='xOB', b='xMM', c='xOC')
    fig.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
    fig.update_layout(title_text= 'Linear benefits (Figure 8)')
    save_ternary(fig, 'Ternary_plot_Figure_8_linear',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig.show()

"""Figure 9"""
def Figure_9():
    """Function that recreates the Figures of Figure 9 in the paper of Sartakhti
    et al., 2018."""

    # Make the needed dataframes
    column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
    df_Figure_9_no_treatment = pd.DataFrame(columns=column_names)
    df_Figure_9_reducing_MM = pd.DataFrame(columns=column_names)
    df_Figure_9_increasing_h = pd.DataFrame(columns=column_names)

    # Reset initial values for each h iteration
    xOC = 0.4
    xOB = 0.4
    xMM = 0.2
    N = 10
    nOC = 4
    nOB = 4
    nMM = 2

    # Cost of producing diffusible factors
    cOC_value = 0.1
    cOB_value = 0.2
    cMM_value = 0.3

    # Maximal benefit values
    BOC_OC = 0
    BOC_OB = 1.0
    BOC_MM = 1.1
    BOB_OC = 1
    BOB_OB = 0
    BOB_MM = 0
    BMM_OC = 1.1
    BMM_OB = -0.3
    BMM_MM = 0

    # Steepness of the function and a random maximal benefit for the demostration
    s_value = 20
    h_value = 0.3

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, h_value, s_value, BOC_OC, BOB_OC, BMM_OC, BOC_OB,
    BOB_OB, BMM_OB, BOC_MM, BOB_MM, BMM_MM, cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 100)

    # Solve ODE
    y = odeint(dynamics_same_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_9_no_treatment = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                        'xOB': xOB_values, 'xMM': xMM_values})

    # Save the data as csv file
    save_data(df_Figure_9_no_treatment, 'data_Figure_9_no_treatment.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Reset initial values for each h iteration
    xOC = 0.5
    xOB = 0.49
    xMM = 0.01
    N = 10
    nOC = 5
    nOB = 4
    nMM = 1

    # Steepness of the function and a random maximal benefit for the demostration
    s_value = 20
    h_value = 0.3

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, h_value, s_value, BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB,
                BMM_OB, BOC_MM, BOB_MM, BMM_MM, cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 100)

    # Solve ODE
    y = odeint(dynamics_same_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_9_reducing_MM = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                        'xOB': xOB_values, 'xMM': xMM_values})

    # Save the data as csv file
    save_data(df_Figure_9_reducing_MM, 'data_Figure_9_reducing_MM.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Reset initial values for each h iteration
    xOC = 0.4
    xOB = 0.4
    xMM = 0.2
    N = 10
    nOC = 4
    nOB = 4
    nMM = 2

    # Steepness of the function and a random maximal benefit for the demostration
    s_value = 20
    h_value = 0.7

    # Set initial condition and parameters
    y0 = [xOC, xOB, xMM]
    parameters = (N, h_value, s_value, BOC_OC, BOB_OC, BMM_OC, BOC_OB, BOB_OB,
    BMM_OB, BOC_MM, BOB_MM, BMM_MM, cOC_value, cOB_value, cMM_value)
    t = np.linspace(0, 100)

    # Solve ODE
    y = odeint(dynamics_same_h_and_s, y0, t, args=(parameters,))

    # Extract the solution and create dataframe
    xOC_values, xOB_values, xMM_values = y[:, 0], y[:, 1], y[:, 2]
    df_Figure_9_increasing_h = pd.DataFrame({'Generation': t, 'xOC': xOC_values,
                                        'xOB': xOB_values, 'xMM': xMM_values})

    # Save the data as csv file
    save_data(df_Figure_9_increasing_h, 'data_Figure_9_increasing_h.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Make a line plot of no treatment
    df_Figure_9_no_treatment.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('No treatment (Figure 9)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_9_reducing_MM',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot to show the effect of a decrease in MM cells
    fig1 = px.line_ternary(df_Figure_9_reducing_MM, a='xOB', b='xMM', c='xOC')
    fig2 = px.line_ternary(df_Figure_9_no_treatment, a='xOB', b='xMM', c='xOC')
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
    fig1.update_layout(title_text= 'Reducing MM cells (Figure 9)')
    save_ternary(fig1, 'Ternary_plot_Figure_9_reducing_MM',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig1.show()

    # Make a line plot of the effect of a decrease in MM cells
    df_Figure_9_reducing_MM.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('Reducing MM cells (Figure 9)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_9_reducing_MM',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

    # Make a ternary plot to show the effect of an increase in h
    fig3 = px.line_ternary(df_Figure_9_increasing_h, a='xOB', b='xMM', c='xOC')
    fig3.update_layout(
        ternary=dict(
            aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
            caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))


    # Add both lines to one ternary plot
    for trace in fig2.data:
        fig3.add_trace(trace)
    fig3.data[0].update(line=dict(color='red'))
    fig3.data[1].update(line=dict(color='blue'))
    fig3.update_layout(title_text= 'Increasing h (Figure 9)')
    save_ternary(fig3, 'Ternary_plot_Figure_9_increasing_h',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    fig3.show()

    # Make a line plot of the effect of a increase in the h value
    df_Figure_9_increasing_h.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'],
                        label = ['fraction OC', 'fraction OB', 'fraction MM'])
    plt.xlabel('Generations')
    plt.ylabel('Fraction')
    plt.title('Increase of the inflection point h (Figure 9)')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Line_plot_Figure_9_increasing_h',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

"""Figure 10"""
def Figure_10():
    """Function that recreates Figure 10 in the paper of Sartakhti et al., 2018."""
    # Parameters
    N = 10
    h_value = 0.4
    B_value = 1.0

    # Steepness values
    steepness_values = [0.1, 1.0, 10.0, 20.0, 100.0]

    # Create a data frame
    df_Figure_10 = pd.DataFrame(columns=['n_values', 'benefit_values', 's_value'])

    # Loop over the steepness values
    for s_value in steepness_values:
        n_values = np.linspace(0, N, 100)
        benefit_data= [benefit_function(n, h_value, B_value, s_value, N) for n \
                                                                    in n_values]
        df_Figure_10 = pd.concat([df_Figure_10, pd.DataFrame({'n_values': n_values,
                            'benefit_values': benefit_data, 's_value': s_value})])

    # Save the data as csv file
    save_data(df_Figure_10, 'data_Figure_10.csv',
                                r'..\data\reproduced_data_Sartakhti_nonlinear')

    # Make a plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for s_value, group in df_Figure_10.groupby('s_value'):
        plt.plot(group['n_values'], group['benefit_values'], label=f's={s_value}')

    # Make the plot clear
    ax.set_xticks([0, 10])
    ax.set_xticklabels(['0', 'N'], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', r'$B_{ij}$'], fontsize=11)
    plt.title('Sigmoid benefits for different steepness values')
    plt.xlabel('Number of cells (ni)')
    plt.ylabel('Benefit')
    plt.legend(loc ='upper right')
    save_Figure(plt, 'Benefit_function_Figure_10',
                    r'..\visualisation\reproduced_results_Sartakhti_nonlinear')
    plt.show()

if __name__ == "__main__":
    main()
