# Recovering libraries

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic Libraries
import numpy as np
import pandas as pd

# Libraries for Powerlaw 
import powerlaw


# Function for evaluating and visualizing the powerlaw of a graph
# input: a grapg
# output: a figure and two values R and p where:
# R is the loglikelihood ratio between the two candidate distributions. This number will be positive if the data is 
# more likely in the first distribution, and negative if the data is more likely in the second distribution
# p-value is the significance value for that direction. It should be > 0.05
# The normalized_ratio option normalizes R by its std
# https://arxiv.org/pdf/1305.0215.pdf

def free_scale(g):
    degree_sequence = sorted([d for d in g.degree()], reverse=True)
    model = powerlaw.Fit(degree_sequence, disdrete = True)
    fig2 = model.plot_pdf(color='b', linewidth=2)
    model.power_law.plot_pdf(color='b', linestyle='--', ax=fig2)
    model.plot_ccdf(color='r', linewidth=2, ax=fig2)
    model.power_law.plot_ccdf(color='r', linestyle='--', ax=fig2)
    R, p = model.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    return R, p