#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:36:19 2023

@author: jean
"""

import numpy as np

N = 100000  # Number of data points

def f(x):
    return (1 - x**2)**1.5

def g(x):
    return np.exp(x + x**2)

aleatorios = np.random.random(N)

def int_mc_01(f):
    integral = 0
    for i in aleatorios:
        integral += f(i)
    return 1 / N * integral

def int_mc_ab(a, b, f):
    integral = 0
    for i in aleatorios:
        integral += f((b-a)*i +a)*(b-a)
    return 1/ N * integral

print(f'El valor de la primera integral es: {int_mc_01(f)}')
print(f'El valor de la segunda integral es: {int_mc_ab(-2, 2, g)}')
