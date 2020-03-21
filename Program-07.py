#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 00:39:41 2020
Miriam Stevens
@author: steve276

This program generates 6 plots for graphical analysis of USGS earthquake data
from the file "all_month.csv".
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# gen_data = np.genfromtxt("all_month.csv", names=True)

# read data from dataset
data = pd.read_table('all_month.csv', delimiter=',')
# or pd.read_csv()


# create DataFrame
df = pd.DataFrame(data)

# simplify calling parameters needed for plots
mag = data['mag']
lat = data['latitude']
long = data['longitude']
depth = data['depth']


# Fig 1, histogram of magnitude
plt.hist(mag, bins=range(0,10), range=[0,10])    # bin size=1/bin edges are 1 unit apart, instead of 1 value per bin 
plt.xlabel('magnitude')
plt.ylabel('occurences')
#plt.show()
plt.savefig('mag_hist.png')


# Fig 2, KDE of magnitude
import seaborn as sns                 # seaborn used to create a tidy plot with default settings

sns.kdeplot(mag, kernel='gau', bw='scott', legend=False)   # kernel type: Gaussian, kernel width: Scott's rule
plt.xlabel('magnitude')    
plt.ylabel('density')
#plt.show()
plt.savefig('mag_KDE.png')


# Fig 3, Scatter of latitude vs longitude
plt.scatter(long, lat, facecolors='none', edgecolors='k')  # lat vs long to follow common map orientation
plt.xlabel('longitude')    
plt.ylabel('latitude')
#plt.show()
plt.savefig('lat-long_scatter.png')


# Fig 4, Normalized CDF of depth
n_bins = len(depth)              # make number of bins = number of data points to smooth CDF
print(n_bins)      

plt.hist(depth, n_bins, density=True, histtype='step',\
         cumulative=True)
plt.xlabel('earthquake depth')
plt.ylabel('likelihood of occurrence')
#plt.show()
plt.savefig('depth_CDF.png')


# Fig 5, Scatter of depth vs magnitude
plt.scatter(mag, depth, facecolors='none', edgecolors='k')
plt.xlabel('magnitude')    
plt.ylabel('depth')
#plt.show()
plt.savefig('depth-mag_scatter.png')


# Fig 6, Q-Q plot of magnitudes
import statsmodels.api as sm

sm.qqplot(mag)                      # assumes normal distribution by default
#plt.show()
plt.savefig('mag_Q-Q.png')


