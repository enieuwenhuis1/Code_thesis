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
import pandas as pd
import matplotlib.pyplot as plt
import ternary
import plotly.graph_objects as go
import plotly.express as px
import os

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
        The number of osteoclasts in cell group.
    nOB: Int
        The Number of osteoblasts in the cell group.
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
        Probability that cell group contains nOC, nOB and N - nOC - nOB MM cells.
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
In game theory, a "payoff" represents the benefit that a player receives through their
actions, decisions, or strategies in a given game. For osteoclasts (OC), osteoblasts
(OB), and multiple myeloma cells (MM), the payoffs are calculated based on the number
of cells of each type in a group, the effects of beneficial growth factors produced by
each cell type, and the associated costs.

VOC=bOC,OC (nOC+1)+bOB,OC(nOB)+bMM,OC(N−1−nOC−nOB)−cOC
- Positive terms: positive contributions to the payoff, the effects of growth factors
- Negative term: the cost of producing growth factors
​"""

def payoff_OC(nOC, nOB, N, bOC_OC, bOB_OC, bMM_OC, cOC):
    """Function that calculates the payoff for osteoclasts (2).

     Parameters:
    -----------
    nOC: Int
        The number of osteoclasts in cell group.
    nOB: Int
        The Number of osteoblasts in the cell group.
    N: Int
        The total number of cells in the group excluding the focal cell itself.
    bOC_OC: Float
        The benefit on a OC of the growth factors produced by an OC.
    bOB_OC: Float
        The benefit on a OC of the growth factors produced by an OB.
    bMM_OC: Float
        The benefit on a OC of the growth factors produced by an MM.
    cOC: Float
        The cost of producing growth factors by OC.

    Returns:
    -----------
    VOC: Float
        Payoff for osteoclasts.
    """
    VOC = (bOC_OC * (nOC + 1)) + (bOB_OC * nOB) + (bMM_OC * (N - 1 - nOC - nOB)) - cOC #(2)
    return VOC

def payoff_OB(nOC, nOB, N, bOC_OB, bOB_OB, bMM_OB, cOB):
    """Function that calculates the payoff for osteoblasts (3).

     Parameters:
    -----------
    nOC: Int
        The number of osteoclasts in cell group.
    nOB: Int
        The Number of osteoblasts in the cell group.
    N: Int
        The total number of cells in the group excluding the focal cell itself.
    bOC_OB: Float
        The benefit on a OB of the growth factors produced by an OC.
    bOB_OB: Float
        The benefit on a OB of the growth factors produced by an OB.
    bMM_OB: Float
        The benefit on a OB of the growth factors produced by an MM.
    cOB: Float
        The cost of producing growth factors by OB.

    Returns:
    -----------
    VOC: Float
        Payoff for osteoblasts.
    """
    VOB = (bOC_OB * nOC) + (bOB_OB * (nOB + 1)) + (bMM_OB * (N - 1 - nOC - nOB)) - cOB #(3)
    return VOB

def payoff_MM(nOC, nOB, N, bOC_MM, bOB_MM, bMM_MM, cMM):
    """Function that calculates the payoff for multiple myeloma cells (4).

    Parameters:
    -----------
    nOC: Int
        The number of osteoclasts in cell group.
    nOB: Int
        The Number of osteoblasts in the cell group.
    N: Int
        The total number of cells in the group excluding the focal cell itself.
    bOC_MM: Float
        The benefit on a OC of the growth factors produced by an OC.
    bOB_MM: Float
        The benefit on a OC of the growth factors produced by an OB.
    bMM_MM: Float
        The benefit on a OC of the growth factors produced by an MM.
    cMM: Float
        The cost of producing growth factors by MM

    Returns:
    -----------
    VOC: Float
        Payoff for multiple myeloma.
    """
    VMM = (bOC_MM * nOC) + (bOB_MM * nOB) + (bMM_MM * (N - nOC - nOB)) - cMM #(4)
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

def calculate_fitness(N, xOC, xOB, xMM, bOC_OC, bOB_OC, bMM_OC, cOC, bOC_OB, bOB_OB,
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
       The benefit on a OC of the growth factors produced by an MM.
    cOC: Float
       The cost of producing growth factors by OC.
    bOC_OB: Float
       The benefit on a OB of the growth factors produced by an OC.
    bOB_OB: Float
       The benefit on a OB of the growth factors produced by an OB.
    bMM_OB: Float
       The benefit on a OB of the growth factors produced by an MM.
    cOB: Float
       The cost of producing growth factors by OB.
    bOC_MM: Float
       The benefit on a OC of the growth factors produced by an OC.
    bOB_MM: Float
       The benefit on a OC of the growth factors produced by an OB.
    bMM_MM: Float
       The benefit on a OC of the growth factors produced by an MM.
    cMM: Float
       The cost of producing growth factors by MM.

    Returns:
    -----------
    normalized_fitness_OC: Float
        The normalized fitness of the osteoclasts.
    normalized_fitness_OB: Float
        The normalized fitness of the osteoblasts.
    normalized_fitness_MM: Float
        The normalized fitness of the multiple myeloma.
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
            fitness_OC += probability_value * payoff_OC_value
            payoff_OB_value = payoff_OB(nOC, nOB, N, bOC_OB, bOB_OB, bMM_OB, cOB)
            fitness_OB += probability_value * payoff_OB_value
            payoff_MM_value = payoff_MM(nOC, nOB, N, bOC_MM, bOB_MM, bMM_MM, cMM)
            fitness_MM += probability_value * payoff_MM_value

    # Normalize the fitness values
    normalization_factor = 1 / (N - 1)
    normalized_fitness_OC = normalization_factor * fitness_OC
    normalized_fitness_OB = normalization_factor * fitness_OB
    normalized_fitness_MM = normalization_factor * fitness_MM

    return normalized_fitness_OC, normalized_fitness_OB, normalized_fitness_MM

"""
Replicator dynamics says that cells with a higher fitness will increase in frequency
over time, while those with lower fitness will decrease. W* represents the average
fitness in the population: W* = xOC(WOC-W*)+xOB(WOB-W*) + xMM(WMM-W*).
The frequencies of each cell type change based on the difference between the fitness
of each cell type and the average fitness in the population.
"""

def calculate_replicator_dynamics(N, xOC, xOB, xMM, WOC, WOB, WMM):
    """ Function that calculates the frequency of osteoblasts, osteoclasts and
    multiple myeloma cells in the next generation based on replicator dynamics.

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

def sigmoid(n, h, B_max, s, N):
    """ Functionthat calculates the sigmoid value.

    Parameters:
    -----------
    n : Int
        The number of cells of type i.
    h : Float
        The position of the inflection point.
    B : Float
        The maximum benefit.
    s : Float
        The  steepness of the function.
    N : Int
        The total number of cell in the group.

    Returns:
    --------
    sigmoid_value: Float
        The output of the sigmoid function.
    """
    sigmoid_value = B_max / (1 + np.exp(s * (h - n/ N)))
    return sigmoid_value

def benefit_function(n, h, B_max, s, N):
    """ Function that calculates the benefit value of the growth factors produced
    by cell type i on cell type j (9).

    Parameters:
    -----------
    n : Int
        The number of cells of type i.
    h : Float
        The position of the inflection point.
    B : Float
        The maximum benefit.
    s : Float
        The  steepness of the function.
    N : Int
        The total number of cell in the group.

    Returns:
    --------
    benefit_value: Float
        Value that indicates the effect of the growth factors produced by cell
        type i on cell type j
    """
    benefit_value = (sigmoid(n, h, B_max, s, N) - sigmoid(0, h, B_max, s, N)) / \
                        (sigmoid(N, h, B_max, s, N) - sigmoid(0, h, B_max, s, N))

    # If the benefit value is nan set it to zero
    if math.isnan(benefit_value):
        benefit_value = 0

    return benefit_value

def save_data(data_frame, file_name, folder_path):
    """ Function that saves a dataframe as csv file

    Parameters:
    -----------
    file_name: String
        The name of the csv file.
    data_frame:
        The data frame contain the collected data.
    folder_path: String:
        Path to the folder where the data will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    data_frame.to_csv(file_path, index=False)

def collect_data(file_name, folder_path):
    """ Function that reads the data from a csv file to a dataframe

    Parameters:
    -----------
    file_name : String
        The name of the csv file.
    folder_path: String:
        Path to the folder where the data will be saved.

    Returns:
    --------
    data_frame:
        The data frame contain the collected data.
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    data_frame = pd.read_csv(file_path)

    return data_frame
# 
# def save_figure(figure, file_name, folder_path):
#     """
#     Save the figure to a specific folder.
#
#     Parameters:
#     -----------
#     figure: Matplotlib figure
#         Figure object that needs to be saved.
#     file_name : String
#         The name for the plot.
#     folder_path: String:
#         Path to the folder where the data will be saved.
#     """
#     os.makedirs(folder_path, exist_ok=True)
#     figure.savefig(os.path.join(folder_path, file_name))
#
# def save_ternary(figure, file_name, folder_path):
#     """
#     Save the ternary plot in a specific folder.
#
#     Parameters:
#     -----------
#     figure: Matplotlib figure
#         Figure object that needs to be saved.
#     file_name : String
#         The name for the plot.
#     folder_path: String:
#         Path to the folder where the data will be saved.
#     """
#     os.makedirs(folder_path, exist_ok=True)
#     pio.write_image(figure, os.path.join(folder_path, f'{file_name}.png'), format='png')
#
#
# """Figure in materials and methods (figure 10)"""
# # Parameters
# N = 10
# h_value = 0.4
# B_value = 1.0
#
# # Steepness values
# steepness_values = [0.1, 1.0, 10.0, 20.0, 100.0]
#
# # Create a data frame
# df_figure_10 = pd.DataFrame(columns=['n_values', 'benefit_values', 's_value'])
#
# # Loop over the steepness values
# for s_value in steepness_values:
#     n_values = np.linspace(0, N, 100)
#     benefit_data= [benefit_function(n, h_value, B_value, s_value, N) for n in n_values]
#
#     # Add the data to the dataframe
#     df_figure_10 = pd.concat([df_figure_10, pd.DataFrame({'n_values': n_values, 'benefit_values': benefit_data, 's_value': s_value})])
#
# # Save the data as csv file
# save_data(df_figure_10, 'data_figure_10.csv', r'..\data\reproduced_data_Sartakhti')
#
# # collect the dataframe from the csv file
# data_figure_10 = collect_data('data_figure_10.csv', r'..\data\reproduced_data_Sartakhti')
#
# # Make a plot
# fig, ax = plt.subplots(figsize=(10, 6))
# for s_value, group in data_figure_10.groupby('s_value'):
#     plt.plot(group['n_values'], group['benefit_values'], label=f's={s_value}')
#
#
# # Make the plot clear
# ax.set_xticks([0, 10])
# ax.set_xticklabels(['0', 'N'], fontsize=11)
# ax.set_yticks([0, 1])
# ax.set_yticklabels(['0', r'$B_{ij}$'], fontsize=11)
# plt.title('Sigmoid Benefits')
# plt.xlabel('Number of cells (ni)')
# plt.ylabel('Benefit')
# plt.legend()
# save_figure(plt, 'Benefit_curve_figure_10', r'..\visualisation\reproduced_results_Sartakhti')
# plt.show()
#
#
# """ Start figure 1 """
# """
# I want to recreate figure 1 here but I don't know how to do that.
#
# 1. To create a triangle I now use: px.line_ternary. But I don't know how to add the
#     color differences of the fitness and the arrows of the direction. I also tried
#     it in mathplotlib but that doesn't work very well either. So maybe you know a
#     better way (possibly in mathplotlib)?
# 2. The line that is now being plotted does not correspond to what the figure in
#     the article shows, so something is going wrong there as well. So maybe there
#     is an error in the formulas that I can't find?
# 3. Furthermore, I don't know what to take as starting values for xOC, xOB and xMM
#     and nOC, nOB and nMM because that is not stated anywhere?
# 4. I also don't know if I should update nOC, nOB and nMM in the loop? And if so does
#     N stay te same number over different generations of does it become bigger.
# 5. And is fitness the average fitness or just one of the cell types?
# """
#
# # Number of cells
# N = 10
#
# # Cost of producing growth factors
# cOC_value = 0.1
# cOB_value = 0.2
# cMM_value = 0.3
#
# # Maximal benefit values
# BOC_OC = 0.0
# BOC_OB = 1.0
# BOC_MM = 1.1
# BOB_OC = 1.0
# BOB_OB = 0.0
# BOB_MM = 0.0
# BMM_OC = 1.1
# BMM_OB = -0.3
# BMM_MM = 0.0
#
# # Steepness and inflection point
# s = 0.00001
# h = 0.7
#
# # Initial frequencies and values --> are needed to make a plot but are not mentioned
# xOC = 0.2
# xOB = 0.1
# xMM = 0.7
#
# nOC = 2
# nOB = 1
# nMM = 7
#
# # Simulation parameters
# generations = 200
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_1 = pd.DataFrame(columns=column_names)
#
# # Run simulation
# for generation in range(generations):
#
#     # Calculate the benefit values
#     bOC_OC = benefit_function(nOC, h, BOC_OC, s, N)
#     bOB_OC = benefit_function(nOB, h, BOB_OC, s, N)
#     bMM_OC = benefit_function(nMM, h, BMM_OC, s, N)
#
#     bOC_OB = benefit_function(nOC, h, BOC_OB, s, N)
#     bOB_OB = benefit_function(nOB, h, BOB_OB, s, N)
#     bMM_OB = benefit_function(nMM, h, BMM_OB, s, N)
#
#     bOC_MM = benefit_function(nOC, h, BOC_MM, s, N)
#     bOB_MM = benefit_function(nOB, h, BOB_MM, s, N)
#     bMM_MM = benefit_function(nMM, h, BMM_MM, s, N)
#
#     # Determine the fitness values
#     fitness_OC, fitness_OB, fitness_MM = calculate_fitness(N, xOC, xOB, xMM, bOC_OC,
#                                 bOB_OC, bMM_OC, cOC_value, bOC_OB, bOB_OB, bMM_OB,
#                                 cOB_value, bOC_MM, bOB_MM, bMM_MM, cMM_value)
#
#     # Determine the change of the xOC, xOB, xMM values and W average value
#     xOC_change, xOB_change, xMM_change, W_average = calculate_replicator_dynamics(
#                                 N, xOC, xOB, xMM, fitness_OC, fitness_OB, fitness_MM)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_1 = pd.concat([df_figure_1, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     # nOC = int(xOC * N)
#     # nOB = int(xOB * N)
#     # nMM = int(xMM * N)
#
# # Save the data as csv file
# save_data(df_figure_1, 'data_figure_1.csv', r'..\data\reproduced_data_Sartakhti')
#
# # collect the dataframe from the csv file
# data_figure_1 = collect_data('data_figure_1.csv', r'..\data\reproduced_data_Sartakhti')
#
# # Plotting
# data_figure_1.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('fitness/ frequency')
# plt.legend()
# save_figure(plt, 'Line_plot_figure_1', r'..\visualisation\reproduced_results_Sartakhti')
# plt.show()
#
# # Make a ternary plot
# """ So when i plot it in a ternary plot it does not go to the right point"""
# fig = px.line_ternary(data_figure_1, a='xOB', b='xMM', c='xOC')
#
# fig.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
# save_ternary(fig, 'Ternary_plot_figure_1', r'..\visualisation\reproduced_results_Sartakhti')
# fig.show()
#
#
# """ Sigmoid benefits figure 2"""
# # Number of cells
# N = 10
#
# # Cost of producing growth factors
# cOC_value = 0.1
# cOB_value = 0.2
# cMM_value = 0.3
#
# # Maximal benefit values
# BOC_OC = 0
# BOC_OB = 1.0
# BOC_MM = 1.1
# BOB_OC = 1
# BOB_OB = 0
# BOB_MM = 0
# BMM_OC = 1.1
# BMM_OB = -0.3
# BMM_MM = 0
#
# # The inflection points
# h_values = [0.1, 0.3, 0.5, 0.7, 0.9]
#
# # Steepness of the function and a random maximal benefit for the demostration
# s_value = 20
# B_value = 1
#
# # Create a DataFrame to store the data
# df_sigmoides_figure_2 = pd.DataFrame(columns=['n_values', 'benefit_values', 'h_value'])
#
# # Loop over h values
# for h_value in h_values:
#     n_values = np.linspace(0, N, 100)
#     benefit_values = [benefit_function(n, h_value, B_value, s_value, N) for n in n_values]
#
#     # Add the data to the dataframe
#     df_sigmoides_figure_2 = pd.concat([df_sigmoides_figure_2, pd.DataFrame({'n_values': n_values, 'benefit_values': benefit_values, 'h_value': h_value})])
#
# # Save the data as csv file
# save_data(df_sigmoides_figure_2, 'data_sigmoides_figure_2.csv', r'..\data\reproduced_data_Sartakhti')
#
# # collect the dataframe from the csv file
# data_sigmoides_figure_2 = collect_data('data_sigmoides_figure_2.csv', r'..\data\reproduced_data_Sartakhti')
#
# # Make a plot
# fig, axes = plt.subplots(1, len(h_values), figsize=(12, 5))
# for i, (h_value, group) in enumerate(data_sigmoides_figure_2.groupby('h_value')):
#     axes[i].plot(group['n_values'], group['benefit_values'], label=f'h={h_value}')
#
#     # Give titles
#     axes[i].set_title(f'h={h_value}')
#     axes[i].set_xlabel('Number of producers')
#     axes[i].set_ylabel('Benefit')
#     axes[i].set_xticks([0, 10])
#     axes[i].set_xticklabels(['0', 'N'], fontsize=11)
#     axes[i].set_yticks([0, 1])
#     axes[i].set_yticklabels(['0', r'$B_{ij}$'], fontsize=11)
#
# # Show the plot
# plt.tight_layout()
# save_figure(plt, 'Benefit_curves_figure_2', r'..\visualisation\reproduced_results_Sartakhti')
# plt.show()
#
#
# """ Start figure 7 line plot. But it is not correct"""
# # Number of cells
# N = 10
#
# # Cost of producing growth factors
# cOC_value = 0.1
# cOB_value = 0.12
# cMM_value = 0.14
#
# # Maximal benefit values
# BOC_OC = 1.0
# BOB_OC = 0.7
# BMM_OC = 0.9
# BOC_OB = 1.0
# BOB_OB = 0.7
# BMM_OB = 0.9
# BOC_MM = 1.0
# BOB_MM = 0.7
# BMM_MM = 0.9
#
# # Positions of the inflection points
# hOC_OC = 0.4
# hOC_OB = 0.7
# hOC_MM = 0.1
# hOB_OC = 0.7
# hOB_OB = 0.4
# hOB_MM = 0.2
# hMM_OC = 0.4
# hMM_OB = 0.3
# hMM_MM = 0.7
#
# # steepness of the function at the inflection points
# sOC_OC = 20
# sOC_OB = 20
# sOC_MM = 5
# sOB_OC = 10
# sOB_OB = 10
# sOB_MM = 50
# sMM_OC = 10
# sMM_OB = 5
# sMM_MM = 5
#
# # Initial frequencies and values --> are needed to make a plot but are not mentioned
# xOC = 0.2
# xOB = 0.5
# xMM = 0.3
#
# nOC = 2
# nOB = 5
# nMM = 3
#
# # Simulation parameters
# generations = 100
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_7 = pd.DataFrame(columns=column_names)
#
# # Run simulation
# for generation in range(generations):
#
#     # Calcuate the benefit values
#     bOC_OC = benefit_function(nOC, hOC_OC, BOC_OC, sOC_OC, N)
#     bOB_OC = benefit_function(nOB, hOB_OC, BOB_OC, sOB_OC, N)
#     bMM_OC = benefit_function(nMM, hMM_OC, BMM_OC, sMM_OC, N)
#
#     bOC_OB = benefit_function(nOC, hOC_OB, BOC_OB, sOC_OB, N)
#     bOB_OB = benefit_function(nOB, hOB_OB, BOB_OB, sOB_OB, N)
#     bMM_OB = benefit_function(nMM, hMM_OB, BMM_OB, sMM_OB, N)
#
#     bOC_MM = benefit_function(nOC, hOC_MM, BOC_MM, sOC_MM, N)
#     bOB_MM = benefit_function(nOB, hOB_MM, BOB_MM, sOB_MM, N)
#     bMM_MM = benefit_function(nMM, hMM_MM, BMM_MM, sMM_MM, N)
#
#     # Determine the fitness values
#     fitness_OC, fitness_OB, fitness_MM = calculate_fitness(N, xOC, xOB, xMM,
#                                 bOC_OC, bOB_OC, bMM_OC, cOC_value, bOC_OB, bOB_OB,
#                                  bMM_OB, cOB_value, bOC_MM, bOB_MM, bMM_MM, cMM_value)
#
#     # Determine the change of the xOC, xOB, xMM values and W average value
#     xOC_change, xOB_change, xMM_change, W_average = calculate_replicator_dynamics(
#                                 N, xOC, xOB, xMM, fitness_OC, fitness_OB, fitness_MM)
#
#     # Add row to dataframe
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_7 = pd.concat([df_figure_7, new_row], ignore_index=True)
#
#     # Update xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
# # Save the data as csv file
# save_data(df_figure_7, 'data_figure_7.csv', r'..\data\reproduced_data_Sartakhti')
#
# # collect the dataframe from the csv file
# data_figure_7 = collect_data('data_figure_7.csv', r'..\data\reproduced_data_Sartakhti')
#
# # Make a line plot
# data_figure_7.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('fitness/ frequency')
# plt.legend()
# save_figure(plt, 'Line_plot_figure_7', r'..\visualisation\reproduced_results_Sartakhti')
# plt.show()
#
# # Make a ternary plot
# fig = px.line_ternary(data_figure_7, a='xOB', b='xMM', c='xOC')
# save_ternary(fig, 'Ternary_plot_figure_7', r'..\visualisation\reproduced_results_Sartakhti')
# fig.show()
#
# """ Start figure 8 line plot. But it is not correct"""
# # Number of cells
# N = 20
#
# # Cost of producing growth factors
# cOC_value = 1.2
# cOB_value = 1.0
# cMM_value = 1.8
#
# # Maximal benefit values
# BOC_OC = 1.1
# BOB_OC = 0.95
# BMM_OC = 1.8
# BOC_OB = 1.1
# BOB_OB = 1.1
# BMM_OB = -0.35
# BOC_MM = 1.1
# BOB_MM = 1.5
# BMM_MM = 0.35
#
# # Positions of the inflection points
# hOC_OC = 0.0
# hOC_OB = 0.0
# hOC_MM = 0.0
# hOB_OC = 0.0
# hOB_OB = 0.0
# hOB_MM = 0.0
# hMM_OC = 0.2
# hMM_OB = 0.2
# hMM_MM = 0.2
#
# # steepness of the function at the inflection points
# sOC_OC = 4
# sOC_OB = 4
# sOC_MM = 40
# sOB_OC = 4
# sOB_OB = 4
# sOB_MM = 4
# sMM_OC = 6
# sMM_OB = 6
# sMM_MM = 1000
#
# # Initial frequencies and values --> are needed to make a plot but are not mentioned
# xOC = 0.1
# xOB = 0.2
# xMM = 0.7
#
# nOC = 2
# nOB = 4
# nMM = 16
#
# # Simulation parameters
# generations = 100
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_8 = pd.DataFrame(columns=column_names)
#
# # Run simulation
# for generation in range(generations):
#
#     # Calcuate the benefit values
#     bOC_OC = benefit_function(nOC, hOC_OC, BOC_OC, sOC_OC, N)
#     bOB_OC = benefit_function(nOB, hOB_OC, BOB_OC, sOB_OC, N)
#     bMM_OC = benefit_function(nMM, hMM_OC, BMM_OC, sMM_OC, N)
#
#     bOC_OB = benefit_function(nOC, hOC_OB, BOC_OB, sOC_OB, N)
#     bOB_OB = benefit_function(nOB, hOB_OB, BOB_OB, sOB_OB, N)
#     bMM_OB = benefit_function(nMM, hMM_OB, BMM_OB, sMM_OB, N)
#
#     bOC_MM = benefit_function(nOC, hOC_MM, BOC_MM, sOC_MM, N)
#     bOB_MM = benefit_function(nOB, hOB_MM, BOB_MM, sOB_MM, N)
#     bMM_MM = benefit_function(nMM, hMM_MM, BMM_MM, sMM_MM, N)
#
#     # Determine the fitness values
#     fitness_OC, fitness_OB, fitness_MM = calculate_fitness(N, xOC, xOB, xMM,
#                             bOC_OC, bOB_OC, bMM_OC, cOC_value, bOC_OB, bOB_OB,
#                             bMM_OB, cOB_value, bOC_MM, bOB_MM, bMM_MM, cMM_value)
#
#     # Determine the change of the xOC, xOB, xMM values and W average value
#     xOC_change, xOB_change, xMM_change, W_average = calculate_replicator_dynamics(
#                                 N, xOC, xOB, xMM, fitness_OC, fitness_OB, fitness_MM)
#
#     # Add row to dataframe
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_8 = pd.concat([df_figure_8, new_row], ignore_index=True)
#
#     # Update xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     # nOC = int(xOC * N)
#     # nOB = int(xOB * N)
#     # nMM = int(xMM * N)
#
# # Save the data as csv file
# save_data(df_figure_8, 'data_figure_8.csv', r'..\data\reproduced_data_Sartakhti')
#
# # collect the dataframe from the csv file
# data_figure_8 = collect_data('data_figure_8.csv', r'..\data\reproduced_data_Sartakhti')
#
# # Make a line plot
# data_figure_8.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('fitness/ frequency')
# plt.title('figure 8')
# plt.legend()
#
# save_figure(plt, 'Line_plot_figure_8', r'..\visualisation\reproduced_results_Sartakhti')
# plt.show()
#
# # Make a ternary plot
# fig = px.line_ternary(data_figure_8, a='xOB', b='xMM', c='xOC')
# save_ternary(fig, 'Ternary_plot_figure_8', r'..\visualisation\reproduced_results_Sartakhti')
# fig.show()
