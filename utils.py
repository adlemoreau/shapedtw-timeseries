import numpy as np
import pandas as pd
from numpy.random import randn
from matplotlib import pyplot as plt
import math
from scipy.signal import savgol_filter
import random
 
def linear_scale(s, a, b):
    """
    Linearly scales a vector s into [a,b]
    """
    c = np.min(s)
    d = np.max(s)
    scaled_array = a + (s - c) * (b - a) / (d - c)
    return scaled_array

def sim_scale_vector(L, a, b):
    """
    Simulates a smooth scale vector in [a,b]
    """
    s = np.zeros(L)
    s[0] = np.random.randn()
    for t in range(1, L):
        s[t] = s[t - 1] + np.sin(np.pi * np.random.randn())
    s = savgol_filter(s, L, 6)
    s = linear_scale(s, a, b)
    return(s)
    
def sim_aligned(ts, tau_values=[1,2,3], alpha=0.15, plot_scale=True, plot_series=True, a=0.75, b=1.25):
    """
    Creates aligned series to ts, with stretching pourcentage alpha
    Stretching values are uniformly taken from tau_values
    We also output the aligmnent path
    """
    L = ts.shape[0]
    scale_vector = sim_scale_vector(L, a, b)
    sim_ts = ts*scale_vector
    #then we need to stretch our data according to our stretching parameters
    stretch_amount = int(L*alpha)
    stretch_indexes = sorted(random.sample(range(L), stretch_amount), reverse=True)
    tau_list = random.choices(tau_values, k=stretch_amount)
    tau_listcopy = list(tau_list)
    i1, i2 = 0, 0
    index1, index2 = [], []
    # in the while loop, we first create the ground truth aligments
    while i1 < L :
        if i1 in stretch_indexes :
            tau = tau_listcopy.pop()
            for t in range(tau+1):
                index1.append(i1)
                index2.append(i2+t)
            i1 += 1
            i2 += tau+1
        else :
            index1.append(i1)
            index2.append(i2)
            i1 += 1
            i2 += 1
    index1 = np.array(index1)
    index2 = np.array(index2)
    # then we create the simulated aligned serie
    for index in stretch_indexes:
        tau = tau_list.pop()
        value_to_insert = sim_ts[index]
        insert_values = np.full(tau, value_to_insert)
        sim_ts = np.insert(sim_ts, index + 1, insert_values)
    
    #plot the scale vector
    if plot_scale:
        plt.plot(scale_vector)
        plt.title("Scale vector")
        plt.show()
    
    if plot_series:
        plt.plot(ts, label="original")
        plt.plot(sim_ts, label="simulated")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    return(sim_ts, index1, index2)

def align_dist_area(align1, align2):
    """
    Computes are between two aligments (Mean Absolute Deviation)
    """
    # Make sure the alignment path is increasing for align1
    p1, q1 = align1[:, 0], align1[:, 1]
    idx1 = np.argsort(p1)
    p1, q1 = p1[idx1], q1[idx1]

    len_p_1, len_q_1 = p1[-1], q1[-1]

    # Make sure the alignment path is increasing for align2
    p2, q2 = align2[:, 0], align2[:, 1]
    idx2 = np.argsort(p2)
    p2, q2 = p2[idx2], q2[idx2]

    len_p_2, len_q_2 = p2[-1], q2[-1]

    if len_p_1 != len_p_2:
        p2, q2 = q2, p2
        len_p_2, len_q_2 = len_p_1, len_q_1

    if p1[-1] != p2[-1] or q1[-1] != q2[-1]:
        raise ValueError('Make sure two alignments are between the same pair of time series')

    len_p, len_q = len_p_1, len_q_1

    loss = np.zeros(len_p)
    for i in range(len_p):
        idx_q1 = q1[p1 == i + 1] 
        idx_q2 = q2[p2 == i + 1]
        tmp = np.abs(np.min(idx_q1) - np.min(idx_q2))
        loss[i] = tmp

    d = np.sum(loss)
    return d
