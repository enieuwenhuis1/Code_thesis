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

"""
The formulas described in the materials and methods (formulas 1 to 10).
x_OC = frequency osteoclasten
x_OB = frequency osteoblasen
x_MM = frequency multiplemyeloma cells
x_OC + x_OB + x_MM = 1

N = Number of cells within the difuusie range of the growth factors
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
        The frequency of osteoclasts in the population.
    xOB: Float
        The frequency of osteoblasts in the population.
    xMM: Float
        The frequency of multiple myeloma cells in the population.

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
    combination_part_1 = math.factorial(N - 1)/ (math.factorial(nOC) * math.factorial(N - 1 - nOC))
    combination_part_2 = math.factorial(N - 1-nOC)/ (math.factorial(nOB) * math.factorial(N - 1 - nOC - nOB))
    combination_part = combination_part_1 * combination_part_2

    # Probability of having nOC osteoclasts, nOB osteoblast and N - nOB - nOC - 1
    # multiple myeloma cells
    probability_part = (xOC**nOC) * (xOB**nOB) * (xMM**(N - nOB - nOC - 1))

    # Calculate the final probability
    probability = combination_part * probability_part # (1)

    return probability

"""
In game theory, a "payoff" represents the benefit that a player receives through their
actions, decisions, or strategies in a given game. For osteoclasts (OC), osteoblasts
(OB), and multiple myeloma cells (MM), the payoffs are calculated based on the number
of cells of each type in a group, the effects of beneficial growth factors produced by
each cell type, and the associated costs.

VOC= bOC,OC(nOC+1)+ bOB,OC(nOB)+ bMM,OC(N−1−nOC−nOB)−cOC
- Positive terms: positive contributions to the payoff, the effects of growth factors
- Negative term: the cost of producing growth factors
​"""

def payoff_OC(nOC, nOB, N, bOC_OC, bOB_OC, bMM_OC, cOC, cOB, cMM):
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
        The benefit on a OC of the growth factors produced by an OC.
    bOB_OC: Float
        The benefit on a OC of the growth factors produced by an OB.
    bMM_OC: Float
        The benefit on a OC of the growth factors produced by an MM cell.
    cOC: Float
        The cost of producing growth factors by OC.

    Returns:
    -----------
    VOC: Float
        Payoff for osteoclasts.

    Example:
    -----------
    >>> payoff_OC(2, 3, 15, 0.1, 0.2, 0.3, 0.4)
    3.1999999999999997
    """
    print(nOC, nOB, N, bOC_OC, bOB_OC, bMM_OC, cOC, 'values')
    VOC = (bOC_OC * (nOC + 1)) + (bOB_OC * nOB) + (bMM_OC * (N - 1 - nOC - nOB)) \
                                                                        - cOC #(2)
    # VOC = (((bOC_OC * (nOC + 1) *cOC) + (bOB_OC * nOB*cOB) + (bMM_OC * cMM* (N - 1 - nOC - nOB)))/ N )\
    #                                                                     - cOC #(2)
    # VOC = (((bOC_OC * (nOC + 1) ) + (bOB_OC * nOB) + (bMM_OC * (N - 1 - nOC - nOB)))/ N )\
    #                                                                     - cOC #(2)
    return VOC

def payoff_OB(nOC, nOB, N, bOC_OB, bOB_OB, bMM_OB, cOC, cOB, cMM):
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
        The benefit on a OB of the growth factors produced by an OC.
    bOB_OB: Float
        The benefit on a OB of the growth factors produced by an OB.
    bMM_OB: Float
        The benefit on a OB of the growth factors produced by an MM cell.
    cOB: Float
        The cost of producing growth factors by OB.

    Returns:
    -----------
    VOB: Float
        Payoff for osteoblasts.

    Example:
    -----------
    >>> payoff_OB(2, 3, 15, 0.1, 0.2, 0.3, 0.4)
    3.3
    """
    print(nOC, nOB, N, bOC_OB, bOB_OB, bMM_OB, cOB, 'values')
    VOB = (bOC_OB * nOC) + (bOB_OB * (nOB + 1)) + (bMM_OB * (N - 1 - nOC - nOB)) \
                                                                        - cOB #(3)
    # VOB = (((bOC_OB * nOC *cOC) + (bOB_OB * (nOB + 1)* cOB) + (bMM_OB * cMM *(N - 1 - nOC - nOB)))/N) \
    #                                                                     - cOB #(3)
    # VOB = (((bOC_OB * nOC) + (bOB_OB * (nOB + 1)) + (bMM_OB * (N - 1 - nOC - nOB)))/N) \
    #                                                                     - cOB #(3)
    return VOB

def payoff_MM(nOC, nOB, N, bOC_MM, bOB_MM, bMM_MM, cOC, cOB, cMM):
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
        The benefit on a MM cell of the growth factors produced by an OC.
    bOB_MM: Float
        The benefit on a MM cell of the growth factors produced by an OB.
    bMM_MM: Float
        The benefit on a MM cell of the growth factors produced by an MM cell.
    cMM: Float
        The cost of producing growth factors by multiple myeloma cells

    Returns:
    -----------
    VMM: Float
        Payoff for multiple myeloma cells.

    Example:
    -----------
    >>> payoff_MM(2, 3, 15, 0.1, 0.2, 0.3, 0.4)
    3.4
    """
    VMM = (((bOC_MM * nOC* cOC) + (bOB_MM * nOB* cOB) + (bMM_MM * cMM *(N - nOC - nOB)))/N) - cMM
    VMM = (bOC_MM * nOC) + (bOB_MM * nOB) + (bMM_MM * (N - nOC - nOB)) - cMM #(4)
    # VMM = (((bOC_MM * nOC) + (bOB_MM * nOB) + (bMM_MM  *(N - nOC - nOB)))/N) - cMM
    return VMM

"""
Fitness (Wi) is calculated by considering the payoffs obtained in the randomly formed
groups weighted by the probability that such groups occur.
N/(N-1) = Normalization factor to ensure that the fitness values are on a comparable
          scale across different population sizes
nested summation = the outer sum iterates over values of nOC and the inner sum iterates
                over values of nOB. The constraints 1 ≤ nOC ≤ N-1 and 0≤ nOB ≤ N-1-nOC.
                In outer ring a value of OC is choosen and in inner ring a value for OB is
                choosen
P(nOC, nOB)= the probability of a group having a particular combination of osteoclasts
            and osteoblasts and multiply myeloma cells
Vi = the payoff for type i.
"""

def calculate_fitness(nOC, nOB, N, xOC, xOB, xMM, bOC_OC, bOB_OC, bMM_OC, cOC, bOC_OB, bOB_OB,
                                        bMM_OB, cOB, bOC_MM, bOB_MM, bMM_MM, cMM):
    """ Function that calculates the fitness of the osteoblasts, osteoclasts and
    multiple myeloma cells (5).

    Parameters:
    -----------
    N: Int
       The total number of cells in the group excluding the focal cell itself.
    xOC: Float
       The frequency of osteoclasts in the population.
    xOB: Float
       The frequency of osteoblasts in the population.
    xMM: Float
       The frequency of multiple myeloma cells in the population.
    bOC_OC: Float
       The benefit on a OC of the growth factors produced by an OC.
    bOB_OC: Float
       The benefit on a OC of the growth factors produced by an OB.
    bMM_OC: Float
       The benefit on a OC of the growth factors produced by an MM cell.
    cOC: Float
       The cost of producing growth factors by OC.
    bOC_OB: Float
       The benefit on a OB of the growth factors produced by an OC.
    bOB_OB: Float
       The benefit on a OB of the growth factors produced by an OB.
    bMM_OB: Float
       The benefit on a OB of the growth factors produced by an MM cell.
    cOB: Float
       The cost of producing growth factors by OB.
    bOC_MM: Float
       The benefit on an MM cell of the growth factors produced by an OC.
    bOB_MM: Float
       The benefit on an MM cell of the growth factors produced by an OB.
    bMM_MM: Float
       The benefit on an MM cell of the growth factors produced by an MM cell.
    cMM: Float
       The cost of producing growth factors by MM cells.

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
            payoff_OC_value = payoff_OC(nOC, nOB, N, bOC_OC, bOB_OC, bMM_OC, cOC, cOB, cMM)
            print('P1',payoff_OC_value)
            fitness_OC += probability_value * payoff_OC_value
            payoff_OB_value = payoff_OB(nOC, nOB, N, bOC_OB, bOB_OB, bMM_OB, cOC, cOB, cMM)
            print('P2',payoff_OB_value)
            fitness_OB += probability_value * payoff_OB_value
            payoff_MM_value = payoff_MM(nOC, nOB, N, bOC_MM, bOB_MM, bMM_MM, cOC, cOB, cMM)
            print('P3',payoff_MM_value)
            fitness_MM += probability_value * payoff_MM_value

    # Normalize the fitness values
    normalization_factor = 1
    normalized_fitness_OC = fitness_OC/ normalization_factor
    normalized_fitness_OB = fitness_OB/ normalization_factor
    normalized_fitness_MM = fitness_MM/ normalization_factor

    return normalized_fitness_OC, normalized_fitness_OB, normalized_fitness_MM

"""
Replicator dynamics says that cells with a higher fitness will increase in frequency
over time, while those with lower fitness will decrease. W* represents the average
fitness in the population: W* = xOC(WOC-W*)+xOB(WOB-W*) + xMM(WMM-W*).
The frequencies of each cell type change based on the difference between the fitness
of each cell type and the average fitness in the population.
"""

def calculate_replicator_dynamics(xOC, xOB, xMM, WOC, WOB, WMM):
    """ Function that calculates the frequency of osteoblasts, osteoclasts and
    multiple myeloma cells in the next generation based on replicator dynamics.

    Parameters:
    -----------
    xOC: Float
        The frequency of osteoclasts in the population.
    xOB: Float
        The frequency of osteoblasts in the population.
    xMM: Float
        The frequency of multiple myeloma cells in the population.
    WOC: Float
        The fitness of the osteoclasts.
    WOB: Float
        The fitness of the osteoblasts.
    WMM: Float
        The fitness of the multiple myeloma cells.

    Returns:
    -----------
    xOC_change: Float
         The change in frequency of osteoclasts in the population.
    xOB_change: Float
         The change infrequency of osteoblasts in the population.
    xMM_change: Float
         The change in frequency of multiple myeloma cells in the population.
    W_average: Float
        The average fitness.

    Example:
    -----------
    >>> calculate_replicator_dynamics(0.3, 0.4, 0.3, 0.1, 0.2, 0.3)
    (-0.03, 0.0, 0.029999999999999992, 0.2)
    """
    # Determine the average fittness
    W_average = xOC * WOC + xOB * WOB + xMM * WMM

    # Determine the new frequencies based of replicator dynamics
    xOC_change = xOC * (WOC - W_average) # (6)
    xOB_change = xOB * (WOB - W_average) # (7)
    xMM_change = xMM * (WMM - W_average) # (8)

    return xOC_change, xOB_change, xMM_change, W_average

"""
The benefit function gives the benefit of the growth factors of cell type i on cell
type j. The more cells of type i (higher n) the higher the benefit becaues more
growth factors (10).
"""

def sigmoid(n_i, h, B_max, s, N):
    """ Functionthat calculates the sigmoid value.

    Parameters:
    -----------
    n_i : Int
        The number of cells of type i.
    h : Float
        The position of the inflection point.
    B_max : Float
        The maximum benefit.
    s : Float
        The  steepness of the function.
    N : Int
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
    """ Function that calculates the benefit value of the growth factors produced
    by cell type i on cell type j (9).

    Parameters:
    -----------
    n_i : Int
        The number of cells of type i.
    h : Float
        The position of the inflection point.
    B_max : Float
        The maximum benefit.
    s : Float
        The steepness of the function.
    N : Int
        The total number of cell in the group.

    Returns:
    -----------
    benefit_value: Float
        Value that indicates the effect of the growth factors produced by cell
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
    file_name : String
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

# if __name__ == "__main__":
#     # Do doc tests
#     # import doctest
#     # doctest.testmod()
