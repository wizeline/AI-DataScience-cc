import collections
import os, requests, zipfile, io
import numpy as np
import pandas as pd
from itertools import islice
import plotly.graph_objs as go
import plotly.plotly as py
import pandas as pd
import matplotlib.pyplot as plt
import pprint as pp
from pandas.tools.plotting import scatter_matrix
import eda_functions
from IPython.display import Image

def get_key_value_pairs(filename):
    with open(filename, 'r') as file:
        for line in file.readlines():
            key, value = line.strip().split(' ')
            yield key, value

def hist_plot(data, feature, bins, log=False):
    plt.figure()
    plt.hist(data, bins=bins)
    if log:
        plt.yscale('log')
    plt.title(feature)
    plt.show()

def scattermatrixplot(dataframe,column,act_filters,feature):
    active = dataframe[dataframe[column].isin(act_filters)]
    correlations =active[[feature+'-X',feature+'-Y',feature+'-Z']]
    colors = [act_color[dataframe.activities.iloc[idx]] for idx in correlations.index.values]
    axes = scatter_matrix(correlations,c=colors, alpha=0.5, diagonal='kde')
    plt.show()            

act_color = {'WALKING':'black',
             'WALKING_UPSTAIRS':'gold',
             'WALKING_DOWNSTAIRS':'magenta',
             'SITTING':'red',
             'STANDING':'green',
             'LAYING':'blue'}