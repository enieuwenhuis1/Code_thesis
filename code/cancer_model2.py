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
# def probability_number_cells(nOC, nOB, N, xOC, xOB, xMM):
#     """ Function that calulates the probability that a group of cells contains
#     specific numbers of OC (nOC), OB (nOB), and MM (N - nOC - nOB) cells (1).
#
#     Parameters:
#     -----------
#     nOC: Int
#         The number of osteoclasts in the population.
#     nOB: Int
#         The Number of osteoblasts in the population.
#     N: Int
#         The total number of cells in the group excluding the focal cell itself.
#     xOC: Float
#         The frequency of osteoclasts in the population.
#     xOB: Float
#         The frequency of osteoblasts in the population.
#     xMM: Float
#         The frequency of multiple myeloma cells in the population.
#
#     Returns:
#     -----------
#     probability: Float
#         Probability that population contains nOC, nOB and N - nOC - nOB MM cells.
#
#     Example:
#     -----------
#     >>> probability_number_cells(2, 3, 10, 0.3, 0.4, 0.3)
#     0.05878656000000001
#     """
#     # Number of ways to choose nOC OC cells and nOB OB cells from a total of N−1 cells
#     combination_part_1 = math.factorial(N - 1)/ (math.factorial(nOC) * math.factorial(N - 1 - nOC))
#     combination_part_2 = math.factorial(N - 1-nOC)/ (math.factorial(nOB) * math.factorial(N - 1 - nOC - nOB))
#     combination_part = combination_part_1 * combination_part_2
#
#     # Probability of having nOC osteoclasts, nOB osteoblast and N - nOB - nOC - 1
#     # multiple myeloma cells
#     probability_part = (xOC**nOC) * (xOB**nOB) * (xMM**(N - nOB - nOC - 1))
#
#     # Calculate the final probability
#     probability = combination_part * probability_part # (1)
#
#     return probability
#
# """
# In game theory, a "payoff" represents the benefit that a player receives through their
# actions, decisions, or strategies in a given game. For osteoclasts (OC), osteoblasts
# (OB), and multiple myeloma cells (MM), the payoffs are calculated based on the number
# of cells of each type in a group, the effects of beneficial growth factors produced by
# each cell type, and the associated costs.
#
# VOC= bOC,OC(nOC+1)+ bOB,OC(nOB)+ bMM,OC(N−1−nOC−nOB)−cOC
# - Positive terms: positive contributions to the payoff, the effects of growth factors
# - Negative term: the cost of producing growth factors
# ​"""
#
# def payoff_OC(nOC, nOB, N, bOC_OC, bOB_OC, bMM_OC, cOC, cOB, cMM):
#     """Function that calculates the payoff for osteoclasts (2).
#
#      Parameters:
#     -----------
#     nOC: Int
#         The number of osteoclasts in population.
#     nOB: Int
#         The Number of osteoblasts in the population.
#     N: Int
#         The total number of cells in the group excluding the focal cell itself.
#     bOC_OC: Float
#         The benefit on a OC of the growth factors produced by an OC.
#     bOB_OC: Float
#         The benefit on a OC of the growth factors produced by an OB.
#     bMM_OC: Float
#         The benefit on a OC of the growth factors produced by an MM cell.
#     cOC: Float
#         The cost of producing growth factors by OC.
#
#     Returns:
#     -----------
#     VOC: Float
#         Payoff for osteoclasts.
#
#     Example:
#     -----------
#     >>> payoff_OC(2, 3, 15, 0.1, 0.2, 0.3, 0.4)
#     3.1999999999999997
#     """
#
#     VOC = (bOC_OC * (nOC + 1)) + (bOB_OC * nOB) + (bMM_OC * (N - 1 - nOC - nOB)) \
#                                                                         - cOC #(2)
#     # VOC = (((bOC_OC * (nOC + 1) *cOC) + (bOB_OC * nOB*cOB) + (bMM_OC * cMM* (N - 1 - nOC - nOB)))/ N )\
#     #                                                                     - cOC #(2)
#     # VOC = (((bOC_OC * (nOC + 1) ) + (bOB_OC * nOB) + (bMM_OC * (N - 1 - nOC - nOB)))/ N )\
#     #                                                                     - cOC #(2)
#     return VOC
#
# def payoff_OB(nOC, nOB, N, bOC_OB, bOB_OB, bMM_OB, cOC, cOB, cMM):
#     """Function that calculates the payoff for osteoblasts (3).
#
#      Parameters:
#     -----------
#     nOC: Int
#         The number of osteoclasts in population.
#     nOB: Int
#         The Number of osteoblasts in the population.
#     N: Int
#         The total number of cells in the group excluding the focal cell itself.
#     bOC_OB: Float
#         The benefit on a OB of the growth factors produced by an OC.
#     bOB_OB: Float
#         The benefit on a OB of the growth factors produced by an OB.
#     bMM_OB: Float
#         The benefit on a OB of the growth factors produced by an MM cell.
#     cOB: Float
#         The cost of producing growth factors by OB.
#
#     Returns:
#     -----------
#     VOB: Float
#         Payoff for osteoblasts.
#
#     Example:
#     -----------
#     >>> payoff_OB(2, 3, 15, 0.1, 0.2, 0.3, 0.4)
#     3.3
#     """
#
#     VOB = (bOC_OB * nOC) + (bOB_OB * (nOB + 1)) + (bMM_OB * (N - 1 - nOC - nOB)) \
#                                                                         - cOB #(3)
#     # VOB = (((bOC_OB * nOC *cOC) + (bOB_OB * (nOB + 1)* cOB) + (bMM_OB * cMM *(N - 1 - nOC - nOB)))/N) \
#     #                                                                     - cOB #(3)
#     # VOB = (((bOC_OB * nOC) + (bOB_OB * (nOB + 1)) + (bMM_OB * (N - 1 - nOC - nOB)))/N) \
#     #                                                                     - cOB #(3)
#     return VOB
#
# def payoff_MM(nOC, nOB, N, bOC_MM, bOB_MM, bMM_MM, cOC, cOB, cMM):
#     """Function that calculates the payoff for multiple myeloma cells (4).
#
#     Parameters:
#     -----------
#     nOC: Int
#         The number of osteoclasts in population.
#     nOB: Int
#         The Number of osteoblasts in the population.
#     N: Int
#         The total number of cells in the group excluding the focal cell itself.
#     bOC_MM: Float
#         The benefit on a MM cell of the growth factors produced by an OC.
#     bOB_MM: Float
#         The benefit on a MM cell of the growth factors produced by an OB.
#     bMM_MM: Float
#         The benefit on a MM cell of the growth factors produced by an MM cell.
#     cMM: Float
#         The cost of producing growth factors by multiple myeloma cells
#
#     Returns:
#     -----------
#     VMM: Float
#         Payoff for multiple myeloma cells.
#
#     Example:
#     -----------
#     >>> payoff_MM(2, 3, 15, 0.1, 0.2, 0.3, 0.4)
#     3.4
#     """
#     # VMM = (((bOC_MM * nOC* cOC) + (bOB_MM * nOB* cOB) + (bMM_MM * cMM *(N - nOC - nOB)))/N) - cMM
#     VMM = (bOC_MM * nOC) + (bOB_MM * nOB) + (bMM_MM * (N - nOC - nOB)) - cMM #(4)
#     # VMM = (((bOC_MM * nOC) + (bOB_MM * nOB) + (bMM_MM  *(N - nOC - nOB)))/N) - cMM
#     return VMM
#
# """
# Fitness (Wi) is calculated by considering the payoffs obtained in the randomly formed
# groups weighted by the probability that such groups occur.
# N/(N-1) = Normalization factor to ensure that the fitness values are on a comparable
#           scale across different population sizes
# nested summation = the outer sum iterates over values of nOC and the inner sum iterates
#                 over values of nOB. The constraints 1 ≤ nOC ≤ N-1 and 0≤ nOB ≤ N-1-nOC.
#                 In outer ring a value of OC is choosen and in inner ring a value for OB is
#                 choosen
# P(nOC, nOB)= the probability of a group having a particular combination of osteoclasts
#             and osteoblasts and multiply myeloma cells
# Vi = the payoff for type i.
# """
#
# def calculate_fitness(nOC, nOB, N, xOC, xOB, xMM, bOC_OC, bOB_OC, bMM_OC, cOC, bOC_OB, bOB_OB,
#                                         bMM_OB, cOB, bOC_MM, bOB_MM, bMM_MM, cMM):
#     """ Function that calculates the fitness of the osteoblasts, osteoclasts and
#     multiple myeloma cells (5).
#
#     Parameters:
#     -----------
#     N: Int
#        The total number of cells in the group excluding the focal cell itself.
#     xOC: Float
#        The frequency of osteoclasts in the population.
#     xOB: Float
#        The frequency of osteoblasts in the population.
#     xMM: Float
#        The frequency of multiple myeloma cells in the population.
#     bOC_OC: Float
#        The benefit on a OC of the growth factors produced by an OC.
#     bOB_OC: Float
#        The benefit on a OC of the growth factors produced by an OB.
#     bMM_OC: Float
#        The benefit on a OC of the growth factors produced by an MM cell.
#     cOC: Float
#        The cost of producing growth factors by OC.
#     bOC_OB: Float
#        The benefit on a OB of the growth factors produced by an OC.
#     bOB_OB: Float
#        The benefit on a OB of the growth factors produced by an OB.
#     bMM_OB: Float
#        The benefit on a OB of the growth factors produced by an MM cell.
#     cOB: Float
#        The cost of producing growth factors by OB.
#     bOC_MM: Float
#        The benefit on an MM cell of the growth factors produced by an OC.
#     bOB_MM: Float
#        The benefit on an MM cell of the growth factors produced by an OB.
#     bMM_MM: Float
#        The benefit on an MM cell of the growth factors produced by an MM cell.
#     cMM: Float
#        The cost of producing growth factors by MM cells.
#
#     Returns:
#     -----------
#     normalized_fitness_OC: Float
#         The normalized fitness of the osteoclasts.
#     normalized_fitness_OB: Float
#         The normalized fitness of the osteoblasts.
#     normalized_fitness_MM: Float
#         The normalized fitness of the multiple myeloma.
#
#     Example:
#     -----------
#     >>> calculate_fitness(10, 0.3, 0.4, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
#     ... 0.8, 0.9, 1.0, 1.1, 1.2)
#     (0.15821162519999998, 0.5527329200999999, 0.9472542150000003)
#     """
#     fitness_OC = 0
#     fitness_OB = 0
#     fitness_MM = 0
#
#     # Loop over the range of nOC values. (-1 is left out of the range because then the
#     # range goes to N-1 if you have range(1, N-1) Then the range goes to N-2)
#     for nOC in range(1, N):
#
#         # Loop over the range of nOB values
#         for nOB in range(0, N- nOC):
#
#             # Calculate the probability of nOC, nOB and (N-nOC-nOB) MM cells
#             probability_value = probability_number_cells(nOC, nOB, N, xOC, xOB, xMM)
#
#             # Determine the fitness of the OC, OB and MM cells
#             payoff_OC_value = payoff_OC(nOC, nOB, N, bOC_OC, bOB_OC, bMM_OC, cOC, cOB, cMM)
#
#             fitness_OC += probability_value * payoff_OC_value
#             payoff_OB_value = payoff_OB(nOC, nOB, N, bOC_OB, bOB_OB, bMM_OB, cOC, cOB, cMM)
#
#             fitness_OB += probability_value * payoff_OB_value
#             payoff_MM_value = payoff_MM(nOC, nOB, N, bOC_MM, bOB_MM, bMM_MM, cOC, cOB, cMM)
#
#             fitness_MM += probability_value * payoff_MM_value
#
#     # Normalize the fitness values
#     normalization_factor = 1
#     normalized_fitness_OC = fitness_OC/ normalization_factor
#     normalized_fitness_OB = fitness_OB/ normalization_factor
#     normalized_fitness_MM = fitness_MM/ normalization_factor
#
#     return normalized_fitness_OC, normalized_fitness_OB, normalized_fitness_MM
#
# def sigmoid(n_i, h, B_max, s, N):
#     """ Functionthat calculates the sigmoid value.
#
#     Parameters:
#     -----------
#     n_i : Int
#         The number of cells of type i.
#     h : Float
#         The position of the inflection point.
#     B_max : Float
#         The maximum benefit.
#     s : Float
#         The  steepness of the function.
#     N : Int
#         The total number of cell in the group.
#
#     Returns:
#     -----------
#     sigmoid_value: Float
#         The output of the sigmoid function.
#
#     Example:
#     -----------
#     >>> sigmoid(2, 0.5, 10, 2, 20)
#     3.1002551887238754
#     """
#     sigmoid_value = B_max / (1 + np.exp(s * (h - n_i/ N)))
#     return sigmoid_value
#
# def benefit_function(n_i, h, B_max, s, N):
#     """ Function that calculates the benefit value of the growth factors produced
#     by cell type i on cell type j (9).
#
#     Parameters:
#     -----------
#     n_i : Int
#         The number of cells of type i.
#     h : Float
#         The position of the inflection point.
#     B_max : Float
#         The maximum benefit.
#     s : Float
#         The steepness of the function.
#     N : Int
#         The total number of cell in the group.
#
#     Returns:
#     -----------
#     benefit_value: Float
#         Value that indicates the effect of the growth factors produced by cell
#         type i on cell type j
#
#     Example:
#     -----------
#     >>> benefit_function(4, 0.5, 10, 2, 20)
#     0.18480653891012727
#     """
#     # Avoid deviding by zero
#     if B_max == 0:
#         benefit_value = 1
#     else:
#         benefit_value = (sigmoid(n_i, h, B_max, s, N) - sigmoid(0, h, B_max, s, N)) / \
#                             (sigmoid(N, h, B_max, s, N) - sigmoid(0, h, B_max, s, N))
#
#     # If the benefit value is nan set it to zero
#     if math.isnan(benefit_value):
#         benefit_value = 1
#
#     return benefit_value
#
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
# BOC_OC = 0
# BOC_OB = 1.0
# BOC_MM = 1.1
# BOB_OC = 1.0
# BOB_OB = 0.0
# BOB_MM = 0.0
# BMM_OC = 1.1
# BMM_OB = -0.3
# BMM_MM = 0
#
# # Steepness and inflection point
# s = 1e-10
# h = 0.3
#
# # # Initial frequencies and values --> are needed to make a plot but are not mentioned
# xOC = 0.2
# xOB = 0.6
# xMM = 0.2
#
# nOC = 2
# nOB = 6
# nMM = 2
#
# S = 7
# m = 2
# c1 = 0.1
# c2= 0.2
# c3= 0.3
# r1_1= 0
# r1_2= 1.0
# r1_3= 1.1
# N= 10
# x= 0.2
# y= 0.6
# z= 0.2
# a = 1.0
# b = 1.1
#
# def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
#     return ((b*c3*z + a*c2*y)*(N - 1)/N - c1)
#
# def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
#     return (a*c1*x - d*c3*z)*(N - 1)/N - c2
#
# def fitness_WMM(x, y, z, N, c1, c2, c3, b):
#     return (b*c1*x*(N - 1)/N) - c3
#
# # Define binomial distribution function
# def B(x, n, p):
#     return comb(n, x) * p**x * (1 - p)**(n - x)
#
# # Define fitness functions f1, f2, f3
# def f1(S, m, c1, c2, c3, r1_1, r1_2, r1_3, N):
#     return (((m+1 ) * c1 * r1_1) / N) + (((S - m -1) * c2 * r1_2) / N) + (((N - S) * c3 * r1_3) / N)
#
# def f2(S, m, c1, c2, c3, r2_1, r2_2, r2_3, N):
#     return (((m + 1) * c2 * r2_2) / N) + (((S - m - 1) * c3 * r2_3) / N) + (((N - S) * c1 * r2_1) / N)
#
# def f3(S, m, c1, c2, c3, r3_1, r3_2, r3_3, N):
#     return (((m + 1) * c3 * r3_3) / N) + (((S - m - 1) * c1 * r3_1) / N) + (((N - S) * c2 * r3_2) / N)
#
# # Define fitness functions W1
# W1_v = 0
# for S in range(1, N+1):
#     value_inner_w1 = 0
#     prob_1 = B(S - 1, N - 1, 1 - z)
#
#     for m in range(0, S):
#         prob_2 = B(m, S - 1, (x / (1 - z)))
#         f1_v = f1(S, m, c1, c2, c3, r1_1, r1_2, r1_3, N)
#         inner_value = prob_2 * f1_v
#         value_inner_w1 += inner_value
#
#     W1_v += (prob_1 *value_inner_w1)
#
# W1_v -= c1
# # w1 = f1(((1- z)*(N - 1)), x*(N -1), c1, c2, c3, r1_1, r1_2, r1_3, N)- c1
#
# print('W1', W1_v, w1)
# print('WOC', fitness_WOC(x, y, z, N, c1, c2, c3, a, b))
#
# S = 4
# m = 6
# c1 = 0.1
# c2= 0.2
# c3= 0.3
# r2_1= 1
# r2_2= 0.0
# r2_3= -0.3
# N= 10
# x= 0.2
# y= 0.6
# z= 0.2
#
# #Define fitness functions W2
# W2_v = 0
# for S in range(1, N+1):
#     value_inner_w2 = 0
#     prob_1 = B(S - 1, N - 1, 1 - x)
#
#     for m in range(0, S):
#         prob_2 = B(m, S - 1, (y / (1 - x)))
#         f1_v = f2(S, m, c1, c2, c3, r1_1, r1_2, r1_3, N)
#         inner_value = prob_2 * f1_v
#         value_inner_w2 += inner_value
#
#     W2_v += (prob_1 *value_inner_w2)
#
# W2_v -= c3
# w2 = f2((1 - x) * (N - 1), y * (N - 1), c1, c2, c3, r2_1, r2_2, r2_3, N) - c2
#
# print('W2', W2_v, w2)
#
#
#
#
# S = 8
# m = 2
# c1 = 0.1
# c2= 0.2
# c3= 0.3
# r3_1= 1.1
# r3_2= 0.0
# r3_3= 0.0
# N= 10
# x= 0.2
# y= 0.6
# z= 0.2
#
# # Define fitness functions W3
# W3_v = 0
# for S in range(1, N+1):
#     value_inner_w3 = 0
#     prob_1 = B(S - 1, N - 1, 1 -y)
#
#     for m in range(0, S):
#         prob_2 = B(m, S - 1, (z / (1 - y)))
#         f1_v = f3(S, m, c1, c2, c3, r1_1, r1_2, r1_3, N)
#         inner_value = prob_2 * f1_v
#         value_inner_w3 += inner_value
#
#     W3_v += (prob_1 *value_inner_w3)
#
# W3_v -= c3
# w3 = f3((1 - y) * (N - 1), z * (N - 1), c1, c2, c3, r3_1, r3_2, r3_3, N) - c3
# print('W3', W3_v, w3)
#
# W_average = x * W1_v + y* W2_v + z * W3_v
#
# # Determine the new frequencies based of replicator dynamics
# xOC_change = x * (W1_v - W_average) # (6)
# xOB_change = y * (W2_v - W_average) # (7)
# xMM_change = z * (W3_v - W_average) # (8)
#
# print('xOC_change, xOB_change, xMM_change', xOC_change, xOB_change, xMM_change)
#
# # Calculate the benefit values
# bOC_OC = benefit_function(nOC, h, BOC_OC, s, N)
# bOB_OC = benefit_function(nOB, h, BOB_OC, s, N)
# bMM_OC = benefit_function(nMM, h, BMM_OC, s, N)
#
# bOC_OB = benefit_function(nOC, h, BOC_OB, s, N)
# bOB_OB = benefit_function(nOB, h, BOB_OB, s, N)
# bMM_OB = benefit_function(nMM, h, BMM_OB, s, N)
#
# bOC_MM = benefit_function(nOC, h, BOC_MM, s, N)
# bOB_MM = benefit_function(nOB, h, BOB_MM, s, N)
# bMM_MM = benefit_function(nMM, h, BMM_MM, s, N)
#
# # Determine the fitness values
# fitness_OC, fitness_OB, fitness_MM = calculate_fitness(nOC, nOB,N, xOC, xOB, xMM, bOC_OC,
#                             bOB_OC, bMM_OC, cOC_value, bOC_OB, bOB_OB, bMM_OB,
#                             cOB_value, bOC_MM, bOB_MM, bMM_MM, cMM_value)
#
# # Determine the change of the xOC, xOB, xMM values and W average value
# xOC_change, xOB_change, xMM_change, W_average = calculate_replicator_dynamics(
#                             xOC, xOB, xMM, fitness_OC, fitness_OB, fitness_MM)
#
# print('xOC_change, xOB_change, xMM_change', xOC_change, xOB_change, xMM_change)


"""figure 2"""
def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
    return (b*c3*z + a*c2*y)*(N - 1)/N - c1

def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
    return (a*c1*x - d*c3*z)*(N - 1)/N - c2

def fitness_WMM(x, y, z, N, c1, c2, c3, b):
    return (b*c1*x*(N - 1)/N) - c3

a = 1
b = 2.5
d = -0.3

N = 10
c3 = 1.4
c2 = 1.2
c1 = 1

xOC = 0.5
xOB = 0.45
xMM = 0.05

generations = 50

column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
df_figure_2_first_line = pd.DataFrame(columns=column_names)

for generation in range(generations):

    WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
    WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
    WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
    print(WMM, WOB, WOC)

    # Determine the average fittness
    W_average = xOC * WOC + xOB * WOB + xMM * WMM
    print(W_average, 'W_average')

    # Determine the new frequencies based of replicator dynamics
    xOC_change = xOC * (WOC - W_average) # (6)
    xOB_change = xOB * (WOB - W_average) # (7)
    xMM_change = xMM * (WMM - W_average)

    # Add row to dataframe (first add row and the update because then also the
    # beginning values get added to the dataframe at generation =0)
    new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
    df_figure_2_first_line = pd.concat([df_figure_2_first_line, new_row], ignore_index=True)

    # Update the xOC, xOB, xMM values
    xOC = max(0, xOC + xOC_change)
    xOB = max(0, xOB + xOB_change)
    xMM = max(0, xMM + xMM_change)

    """# Do the nOC,nOB and nMM need to be updated ?"""
    nOC = xOC * N
    nOB = xOB * N
    nMM = xMM * N

a = 1
b = 2.5
d = -0.3

N = 10
c3 = 1.4
c2 = 1.2
c1 = 1

xOC = 0.4
xOB = 0.3
xMM = 0.3

generations = 50

column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
df_figure_2_second_line = pd.DataFrame(columns=column_names)

for generation in range(generations):

    WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
    WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
    WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
    print(WMM, WOB, WOC)

    # Determine the average fittness
    W_average = xOC * WOC + xOB * WOB + xMM * WMM
    print(W_average, 'W_average')

    # Determine the new frequencies based of replicator dynamics
    xOC_change = xOC * (WOC - W_average) # (6)
    xOB_change = xOB * (WOB - W_average) # (7)
    xMM_change = xMM * (WMM - W_average)

    # Add row to dataframe (first add row and the update because then also the
    # beginning values get added to the dataframe at generation =0)
    new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
                                            'xMM': xMM, 'W_average': W_average}])
    df_figure_2_second_line = pd.concat([df_figure_2_second_line, new_row], ignore_index=True)

    # Update the xOC, xOB, xMM values
    xOC = max(0, xOC + xOC_change)
    xOB = max(0, xOB + xOB_change)
    xMM = max(0, xMM + xMM_change)

    """# Do the nOC,nOB and nMM need to be updated ?"""
    nOC = xOC * N
    nOB = xOB * N
    nMM = xMM * N

# Make a plot
df_figure_2_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
plt.xlabel('Generations')
plt.ylabel('Fitness/ Frequency')
plt.title('Bistability with linear benefits (figure 1)')
plt.legend()

plt.show()

df_figure_2_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
plt.xlabel('Generations')
plt.ylabel('Fitness/ Frequency')
plt.title('Bistability with linear benefits (figure 1)')
plt.legend()

plt.show()

# Make a ternary plot
""" So when i plot it in a ternary plot it does not go to the right point"""
fig1 = px.line_ternary(df_figure_2_first_line, a='xOC', b='xOB', c='xMM')
fig2 = px.line_ternary(df_figure_2_second_line, a='xOC', b='xOB', c='xMM')


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
fig1.update_layout(title_text= 'Dynamics (figure 2)')

fig1.update_layout(
    ternary=dict(
        aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
        baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
        caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
fig1.update_layout(title_text='Dynamics (figure 2)')
fig1.show()

#
# """figure 5"""
# def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
#     return (b*c3*z + a*c2*y)*(N - 1)/N - c1
#
# def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
#     return (a*c1*x - d*c3*z)*(N - 1)/N - c2
#
# def fitness_WMM(x, y, z, N, c1, c2, c3, b):
#     return (b*c1*x*(N - 1)/N) - c3
#
# a = 1
# b = 2.5
# d = -0.3
#
# N = 2
# c3 = 1.4
# c2 = 1.2
# c1 = 1
#
# xOC = 0.8
# xOB = 0.2
# xMM = 0.0
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_5_first_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_5_first_line = pd.concat([df_figure_5_first_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# a = 1
# b = 2.5
# d = -0.3
#
# N = 2
# c3 = 1.4
# c2 = 1.2
# c1 = 1
#
# xOC = 0.4
# xOB = 0.3
# xMM = 0.3
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_5_second_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_5_second_line = pd.concat([df_figure_5_second_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# # Make a plot
# df_figure_5_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 5)')
# plt.legend()
#
# plt.show()
#
# df_figure_5_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 5)')
# plt.legend()
#
# plt.show()
#
# # Make a ternary plot
# """ So when i plot it in a ternary plot it does not go to the right point"""
# fig1 = px.line_ternary(df_figure_5_first_line, a='xOC', b='xOB', c='xMM')
# fig2 = px.line_ternary(df_figure_5_second_line, a='xOC', b='xOB', c='xMM')
#
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
#
# # Add both lines to one ternary plot
# for trace in fig2.data:
#     fig1.add_trace(trace)
# fig1.data[0].update(line=dict(color='red'))
# fig1.data[1].update(line=dict(color='blue'))
# fig1.update_layout(title_text= 'Dynamics (figure 5)')
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#
# fig1.show()

#
# """figure 8A"""
# def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
#     return (b*c3*z + a*c2*y)*(N - 1)/N - c1
#
# def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
#     return (a*c1*x - d*c3*z)*(N - 1)/N - c2
#
# def fitness_WMM(x, y, z, N, c1, c2, c3, b):
#     return (b*c1*x*(N - 1)/N) - c3
#
# a = 1
# b = 1.1
# d = -0.3
#
# N = 10
# c3 = 1.4
# c2 = 1.2
# c1 = 1
#
# xOC = 0.8
# xOB = 0.2
# xMM = 0.0
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_5_first_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_5_first_line = pd.concat([df_figure_5_first_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# a = 1
# b = 1.1
# d = -0.3
#
# N = 10
# c3 = 1.4
# c2 = 1.2
# c1 = 1
#
# xOC = 0.2
# xOB = 0.5
# xMM = 0.3
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_5_second_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_5_second_line = pd.concat([df_figure_5_second_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# # Make a plot
# df_figure_5_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 8A)')
# plt.legend()
#
# plt.show()
#
# df_figure_5_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 8A)')
# plt.legend()
#
# plt.show()
#
# # Make a ternary plot
# """ So when i plot it in a ternary plot it does not go to the right point"""
# fig1 = px.line_ternary(df_figure_5_first_line, a='xOC', b='xOB', c='xMM')
# fig2 = px.line_ternary(df_figure_5_second_line, a='xOC', b='xOB', c='xMM')
#
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
#
# # Add both lines to one ternary plot
# for trace in fig2.data:
#     fig1.add_trace(trace)
# fig1.data[0].update(line=dict(color='red'))
# fig1.data[1].update(line=dict(color='blue'))
# fig1.update_layout(title_text= 'Dynamics (figure 8A)')
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
# fig1.show()

"""figure 8B"""
# def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
#     return (b*c3*z + a*c2*y)*(N - 1)/N - c1
#
# def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
#     return (a*c1*x - d*c3*z)*(N - 1)/N - c2
#
# def fitness_WMM(x, y, z, N, c1, c2, c3, b):
#     return (b*c1*x*(N - 1)/N) - c3
#
# a = 1
# b = 1.8
# d = -0.3
#
# N = 10
# c3 = 1.4
# c2 = 1.2
# c1 = 1
#
# xOC = 0.8
# xOB = 0.2
# xMM = 0.0
#
# generations = 60
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_5_first_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_5_first_line = pd.concat([df_figure_5_first_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# a = 1
# b = 1.8
# d = -0.3
#
# N = 10
# c3 = 1.4
# c2 = 1.2
# c1 = 1
#
# xOC = 0.2
# xOB = 0.5
# xMM = 0.3
#
# generations = 60
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_5_second_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_5_second_line = pd.concat([df_figure_5_second_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# # Make a plot
# df_figure_5_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 8B)')
# plt.legend()
#
# plt.show()
#
# df_figure_5_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 8B)')
# plt.legend()
#
# plt.show()
#
# # Make a ternary plot
# """ So when i plot it in a ternary plot it does not go to the right point"""
# fig1 = px.line_ternary(df_figure_5_first_line, a='xOC', b='xOB', c='xMM')
# fig2 = px.line_ternary(df_figure_5_second_line, a='xOC', b='xOB', c='xMM')
#
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
#
# # Add both lines to one ternary plot
# for trace in fig2.data:
#     fig1.add_trace(trace)
# fig1.data[0].update(line=dict(color='red'))
# fig1.data[1].update(line=dict(color='blue'))
# fig1.update_layout(title_text= 'Dynamics (figure 8B)')
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
# fig1.show()

"""SENARIO 2"""
"""figure 9A"""
# def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
#     return (b*c3*z + a*c2*y)*(N - 1)/N - c1
#
# def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
#     return (a*c1*x - d*c3*z)*(N - 1)/N - c2
#
# def fitness_WMM(x, y, z, N, c1, c2, c3, b):
#     return (b*c1*x*(N - 1)/N) - c3
#
# a = 1
# b = 0.5
# d = -0.3
#
# N = 10
# c3 = 1
# c2 = 1
# c1 = 1
#
# xOC = 0.8
# xOB = 0.2
# xMM = 0.0
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_9A_first_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_9A_first_line = pd.concat([df_figure_9A_first_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# a = 1
# b = 0.5
# d = -0.3
#
# N = 10
# c3 = 1
# c2 = 1
# c1 = 1
#
# xOC = 0.4
# xOB = 0.3
# xMM = 0.3
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_9A_second_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_9A_second_line = pd.concat([df_figure_9A_second_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# # Make a plot
# df_figure_9A_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 9A)')
# plt.legend()
#
# plt.show()
#
# df_figure_9A_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 9A)')
# plt.legend()
#
# plt.show()
#
# # Make a ternary plot
# """ So when i plot it in a ternary plot it does not go to the right point"""
# fig1 = px.line_ternary(df_figure_9A_first_line, a='xOC', b='xOB', c='xMM')
# fig2 = px.line_ternary(df_figure_9A_second_line, a='xOC', b='xOB', c='xMM')
#
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
#
# # Add both lines to one ternary plot
# for trace in fig2.data:
#     fig1.add_trace(trace)
# fig1.data[0].update(line=dict(color='red'))
# fig1.data[1].update(line=dict(color='blue'))
# fig1.update_layout(title_text= 'Dynamics (figure 9A)')
# fig1.show()
#
#
# """figure 9B"""
#
# def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
#     return (b*c3*z + a*c2*y)*(N - 1)/N - c1
#
# def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
#     return (a*c1*x - d*c3*z)*(N - 1)/N - c2
#
# def fitness_WMM(x, y, z, N, c1, c2, c3, b):
#     return (b*c1*x*(N - 1)/N) - c3
#
# a = 1
# b = 0.5
# d = -1
#
# N = 10
# c3 = 1
# c2 = 1
# c1 = 1
#
# xOC = 0.8
# xOB = 0.0
# xMM = 0.2
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_9B_first_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_9B_first_line = pd.concat([df_figure_9B_first_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# a = 1
# b = 0.5
# d = -1
#
# N = 10
# c3 = 1
# c2 = 1
# c1 = 1
#
# xOC = 0.2
# xOB = 0.5
# xMM = 0.3
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_9B_second_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_9B_second_line = pd.concat([df_figure_9B_second_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# # Make a plot
# df_figure_9B_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 9B)')
# plt.legend()
#
# plt.show()
#
# df_figure_9B_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 9B)')
# plt.legend()
#
# plt.show()
#
# # Make a ternary plot
# """ So when i plot it in a ternary plot it does not go to the right point"""
# fig1 = px.line_ternary(df_figure_9B_first_line, a='xOC', b='xOB', c='xMM')
# fig2 = px.line_ternary(df_figure_9B_second_line, a='xOC', b='xOB', c='xMM')
#
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
#
# # Add both lines to one ternary plot
# for trace in fig2.data:
#     fig1.add_trace(trace)
# fig1.data[0].update(line=dict(color='red'))
# fig1.data[1].update(line=dict(color='blue'))
# fig1.update_layout(title_text= 'Dynamics (figure 9B)')
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
# fig1.show()
#
# """Figure 9C"""
#
# def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
#     return (b*c3*z + a*c2*y)*(N - 1)/N - c1
#
# def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
#     return (a*c1*x - d*c3*z)*(N - 1)/N - c2
#
# def fitness_WMM(x, y, z, N, c1, c2, c3, b):
#     return (b*c1*x*(N - 1)/N) - c3
#
# a = 1
# b = 2
# d = 0
#
# N = 10
# c3 = 1
# c2 = 1
# c1 = 1
#
# xOC = 0.8
# xOB = 0.2
# xMM = 0.0
#
# generations = 60
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_9C_first_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_9C_first_line = pd.concat([df_figure_9C_first_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# a = 1
# b = 2
# d = 0
#
# N = 10
# c3 = 1
# c2 = 1
# c1 = 1
#
# xOC = 0.2
# xOB = 0.5
# xMM = 0.3
#
# generations = 60
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_9C_second_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_9C_second_line = pd.concat([df_figure_9C_second_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# # Make a plot
# df_figure_9C_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 9C)')
# plt.legend()
#
# plt.show()
#
# df_figure_9C_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 9C)')
# plt.legend()
#
# plt.show()
#
# # Make a ternary plot
# """ So when i plot it in a ternary plot it does not go to the right point"""
# fig1 = px.line_ternary(df_figure_9C_first_line, a='xOC', b='xOB', c='xMM')
# fig2 = px.line_ternary(df_figure_9C_second_line, a='xOC', b='xOB', c='xMM')
#
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
#
# # Add both lines to one ternary plot
# for trace in fig2.data:
#     fig1.add_trace(trace)
# fig1.data[0].update(line=dict(color='red'))
# fig1.data[1].update(line=dict(color='blue'))
# fig1.update_layout(title_text= 'Dynamics (figure 9C)')
#
# fig1.show()

#
# """SENARIO 3"""
# """figure 10A"""
# def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
#     return (b*c3*z + a*c2*y)*(N - 1)/N - c1
#
# def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
#     return (a*c1*x - d*c3*z)*(N - 1)/N - c2
#
# def fitness_WMM(x, y, z, N, c1, c2, c3, b):
#     return (b*c1*x*(N - 1)/N) - c3
#
# a = 1
# b = 0.5
# d = -0.3
#
# N = 10
# c3 = 0.8
# c2 = 1.2
# c1 = 1
#
# xOC = 0.1
# xOB = 0.2
# xMM = 0.7
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_10A_first_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_10A_first_line = pd.concat([df_figure_10A_first_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# a = 1
# b = 0.5
# d = -0.3
#
# N = 10
# c3 = 0.8
# c2 = 1.2
# c1 = 1
#
# xOC = 0.4
# xOB = 0.3
# xMM = 0.3
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_10A_second_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_10A_second_line = pd.concat([df_figure_10A_second_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# # Make a plot
# df_figure_10A_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 10A)')
# plt.legend()
#
# plt.show()
#
# df_figure_10A_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 10A)')
# plt.legend()
#
# plt.show()
#
# # Make a ternary plot
# """ So when i plot it in a ternary plot it does not go to the right point"""
# fig1 = px.line_ternary(df_figure_10A_first_line, a='xOC', b='xOB', c='xMM')
# fig2 = px.line_ternary(df_figure_10A_second_line, a='xOC', b='xOB', c='xMM')
#
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
#
# # Add both lines to one ternary plot
# for trace in fig2.data:
#     fig1.add_trace(trace)
# fig1.data[0].update(line=dict(color='red'))
# fig1.data[1].update(line=dict(color='blue'))
# fig1.update_layout(title_text= 'Dynamics (figure 10A)')
# fig1.show()
#
#
# """figure 10B"""
#
# def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
#     return (b*c3*z + a*c2*y)*(N - 1)/N - c1
#
# def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
#     return (a*c1*x - d*c3*z)*(N - 1)/N - c2
#
# def fitness_WMM(x, y, z, N, c1, c2, c3, b):
#     return (b*c1*x*(N - 1)/N) - c3
#
# a = 1
# b = 0.5
# d = 0.3
#
# N = 50
# c3 = 0.8
# c2 = 1.2
# c1 = 1
#
# xOC = 0.8
# xOB = 0.2
# xMM = 0.0
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_10B_first_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_10B_first_line = pd.concat([df_figure_10B_first_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# a = 1
# b = 0.5
# d = 0.3
#
# N = 50
# c3 = 0.8
# c2 = 1.2
# c1 = 1
#
# xOC = 0.2
# xOB = 0.5
# xMM = 0.3
#
# generations = 50
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_10B_second_line = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#     print(W_average, 'W_average')
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_10B_second_line = pd.concat([df_figure_10B_second_line, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC = max(0, xOC + xOC_change)
#     xOB = max(0, xOB + xOB_change)
#     xMM = max(0, xMM + xMM_change)
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# # Make a plot
# df_figure_10B_first_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 10B)')
# plt.legend()
#
# plt.show()
#
# df_figure_10B_second_line.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM', 'W_average'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Dynamics (figure 10B)')
# plt.legend()
#
# plt.show()
#
# # Make a ternary plot
# """ So when i plot it in a ternary plot it does not go to the right point"""
# fig1 = px.line_ternary(df_figure_10B_first_line, a='xOC', b='xOB', c='xMM')
# fig2 = px.line_ternary(df_figure_10B_second_line, a='xOC', b='xOB', c='xMM')
#
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
#
# # Add both lines to one ternary plot
# for trace in fig2.data:
#     fig1.add_trace(trace)
# fig1.data[0].update(line=dict(color='red'))
# fig1.data[1].update(line=dict(color='blue'))
# fig1.update_layout(title_text= 'Dynamics (figure 10B)')
#
# fig1.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
#
# fig1.show()


#
# """
# Author:       Eva Nieuwenhuis
# University:   UvA
# Student id':  13717405
# Description:  Code with the model that simulates the dynamics in the multiple myeloma
#               (MM) microenvironment with three cell types: MM cells, osteoblasts (OBs)
#               and osteoclasts (OCs). The model is a public goods game in the framework
#               of evolutionary game theory with collective interactions and nonlinear
#               benefits.
#               The model is based on a in the paper of Sartakhti et al., 2018.
#
# Sartakhti, J. S., Manshaei, M. H., & Archetti, M. (2018). Game Theory of Tumor–Stroma
# Interactions in Multiple Myeloma: Effect of nonlinear benefits. Games, 9(2), 32.
# https://doi.org/10.3390/g9020032
# """
#
# import math
# import numpy as np
# import os
# import pandas as pd
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import ternary
# import plotly.graph_objects as go
# import plotly.express as px
# from cancer_model import *
# import plotly.io as pio
#
# import numpy as np
#
# # Function to calculate the probability B(x,n,p)
# def B(x, n, p):
#     return np.math.comb(n, x) * (p**x) * ((1 - p)**(n - x))
#
# # Function to calculate f1(S,m), f2(S,m), f3(S,m)
# def f1(S, m, N, c1, c2, c3, r11, r12, r13):
#     print('hi', ((S-m-1)*c2*r12 + (N-S)*c3*r13), ((m+1)*c1*r11 + (S-m-1)*c2*r12 + (N-S)*c3*r13))
#     return ((m+1)*c1*r11 + (S-m-1)*c2*r12 + (N-S)*c3*r13)/ N
#
# def f2(S, m, N, c1, c2, c3, r22, r23, r21):
#     return ((m+1)*c2*r22 + (S-m-1)*c3*r23 + (N-S)*c1*r21)/ N
#
# def f3(S, m, N, c1, c2, c3, r33, r31, r32):
#     print(S-m-1, 'S-m-1')
#
#     return ((m+1)*c3*r33 + (S-m-1)*c1*r31 + (N-S)*c2*r32)/ N
#
# def fitness_WOC(N, z, x, c1, c2, c3, r11, r12, r13):
#     return f1((1 - z)*(N - 1), x*(N - 1), N, c1, c2, c3, r11, r12, r13) - c1
#
# # Function to calculate the fitness function W2
# def fitness_WOB(N, x, y, c1, c2, c3, r22, r23, r21):
#     return f2((1 - x)*(N - 1), y*(N - 1), N, c1, c2, c3, r22, r23, r21) - c2
#
# # Function to calculate the fitness function W3
# def fitness_WMM(N, y, z, c1, c2, c3, r33, r31, r32):
#     return f3((1 - y)*(N - 1), z*(N - 1), N, c1, c2, c3, r33, r31, r32) - c3
#
# # r i,j effect Gj op cell type P i waarvan je fitnes bepaald
# a = 1
# b = 2.5
# d = -0.3
#
# r11 = 0
# r12= a
# r13= b
# r21= a
# r22=0
# r23= -d
# r31= b
# r32= 0
# r33 =0
#
# N = 10
# c3 = 1.4
# c2 = 1.2
# c1 = 1
#
# xOC, x = 0.2, 0.2
# xOB, y = 0.2, 0.2
# xMM, z = 0.6, 0.6
#
# generations = 100
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_1 = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#
#     WOC = fitness_WOC(N, z, x, c1, c2, c3, r11, r12, r13)
#     WOB = fitness_WOB(N, x, y, c1, c2, c3, r22, r23, r21)
#     WMM = fitness_WMM(N, y, z, c1, c2, c3, r33, r31, r32)
#     print(WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_1 = pd.concat([df_figure_1, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC =  xOC + xOC_change
#     xOB =  xOB + xOB_change
#     xMM = xMM + xMM_change
#     print('change', xOC_change, xOB_change, xMM_change)
#
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# # Make a plot
# df_figure_1.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Bistability with linear benefits (figure 1)')
# plt.legend()
#
# plt.show()
#
# # Make a ternary plot
# """ So when i plot it in a ternary plot it does not go to the right point"""
# fig = px.line_ternary(df_figure_1, a='xOC', b='xOB', c='xMM')
#
# fig.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
# fig.update_layout(title_text='Bistability with linear benefits (figure 1)')
# fig.show()
# # Function to calculate the fitness functions W1, W2, W3
# # def fitness_functions(x, y, z, N, c1, c2, c3, r1, r2, r3):
# #     W1 = sum(B(S-1, N-1, 1-z) * sum(B(m, S-1, x/(1-z)) * f1(S, m, N, c1, c2, c3, r1, r2, r3) for m in range(S)) for S in range(1, N+1)) - c1
# #     W2 = sum(B(S-1, N-1, 1-x) * sum(B(m, S-1, y/(1-x)) * f2(S, m, N, c1, c2, c3, r1, r2, r3) for m in range(S)) for S in range(1, N+1)) - c2
# #     W3 = sum(B(S-1, N-1, 1-y) * sum(B(m, S-1, z/(1-y)) * f3(S, m, N, c1, c2, c3, r1, r2, r3) for m in range(S)) for S in range(1, N+1)) - c3
# #     return W1, W2, W3
#
#
#
# def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
#     return (b*c3*z + a*c2*y)*(N - 1)/N - c1
#
# def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
#     return (a*c1*x - d*c3*z)*(N - 1)/N - c2
#
# def fitness_WMM(x, y, z, N, c1, c2, c3, b):
#     return (b*c1*x*(N - 1)/N) - c3
#
# a = 1
# b = 2.5
# d = -0.3
#
# r11 = 0
# r12= a
# r13= b
# r21= a
# r22=0
# r23= -d
# r31= b
# r32= 0
# r33 =0
#
# N = 10
# c3 = 1.4
# c2 = 1.2
# c1 = 1
#
# xOC, x = 0.3, 0.3
# xOB, y = 0.4, 0.4
# xMM, z = 0.3, 0.3
# # xOC = 0.0
# # xOB = 1.0
# # xMM = 0.0
#
# generations = 10
#
# column_names = ['Generation', 'xOC', 'xOB', 'xMM', 'W_average']
# df_figure_1 = pd.DataFrame(columns=column_names)
#
# for generation in range(generations):
#     def fitness_WOC(x, y, z, N, c1, c2, c3, a, b):
#         return (b*c3*z + a*c2*y)*(N - 1)/N - c1
#
#     def fitness_WOB(x, y, z, N, c1, c2, c3, a, d):
#         return (a*c1*x - d*c3*z)*(N - 1)/N - c2
#
#     def fitness_WMM(x, y, z, N, c1, c2, c3, b):
#         print(x*(N - 1), 'x*(N - 1)')
#         return (b*c1*x*(N - 1)/N) - c3
#
#     WOC = fitness_WOC(xOC, xOB, xMM, N, c1, c2, c3, a, b)
#     WOB = fitness_WOB(xOC, xOB, xMM, N, c1, c2, c3, a, d)
#     WMM = fitness_WMM(xOC, xOB, xMM, N, c1, c2, c3, b)
#     print('1', WMM, WOB, WOC)
#
#     def fitness_WOC(N, z, x, c1, c2, c3, r11, r12, r13):
#         return f1((1 - z)*(N - 1), x*(N - 1), N, c1, c2, c3, r11, r12, r13) - c1
#
#     # Function to calculate the fitness function W2
#     def fitness_WOB(N, x, y, c1, c2, c3, r22, r23, r21):
#         return f2((1 - x)*(N - 1), y*(N - 1), N, c1, c2, c3, r22, r23, r21) - c2
#
#     # Function to calculate the fitness function W3
#     def fitness_WMM(N, y, z, c1, c2, c3, r33, r31, r32):
#         return f3((1 - y)*(N - 1), z*(N - 1), N, c1, c2, c3, r33, r31, r32) - c3
#
#     WOC = fitness_WOC(N, z, x, c1, c2, c3, r11, r12, r13)
#     WOB = fitness_WOB(N, x, y, c1, c2, c3, r22, r23, r21)
#     WMM = fitness_WMM(N, y, z, c1, c2, c3, r33, r31, r32)
#     print('2', WMM, WOB, WOC)
#
#     # Determine the average fittness
#     W_average = xOC * WOC + xOB * WOB + xMM * WMM
#
#     # Determine the new frequencies based of replicator dynamics
#     xOC_change = xOC * (WOC - W_average) # (6)
#     xOB_change = xOB * (WOB - W_average) # (7)
#     xMM_change = xMM * (WMM - W_average)
#
#     # Add row to dataframe (first add row and the update because then also the
#     # beginning values get added to the dataframe at generation =0)
#     new_row = pd.DataFrame([{'Generation':generation, 'xOC': xOC, 'xOB': xOB,
#                                             'xMM': xMM, 'W_average': W_average}])
#     df_figure_1 = pd.concat([df_figure_1, new_row], ignore_index=True)
#
#     # Update the xOC, xOB, xMM values
#     xOC =  xOC + xOC_change
#     xOB =  xOB + xOB_change
#     xMM = xMM + xMM_change
#     print('change', xOC_change, xOB_change, xMM_change)
#
#
#     """# Do the nOC,nOB and nMM need to be updated ?"""
#     nOC = xOC * N
#     nOB = xOB * N
#     nMM = xMM * N
#
# # Make a plot
# df_figure_1.plot(x= 'Generation', y= ['xOC', 'xOB', 'xMM'])
# plt.legend(['Frequency OC', 'Frequency OB', 'Frequency MM', 'Average fitness'])
# plt.xlabel('Generations')
# plt.ylabel('Fitness/ Frequency')
# plt.title('Bistability with linear benefits (figure 1)')
# plt.legend()
#
# plt.show()
#
# # Make a ternary plot
# """ So when i plot it in a ternary plot it does not go to the right point"""
# fig = px.line_ternary(df_figure_1, a='xOC', b='xOB', c='xMM')
#
# fig.update_layout(
#     ternary=dict(
#         aaxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         baxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),
#         caxis=dict(ticks='outside', tickvals=[0, 0.25, 0.5, 0.75, 1]),))
# fig.update_layout(title_text='Bistability with linear benefits (figure 1)')
# fig.show()
