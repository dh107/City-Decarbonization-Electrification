# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:22:39 2021

@author: Da Huo -- Aug 2021
"""
import numpy as np
import pandas as pd
import csv
import sys
import copy
import os
import calendar
import xlrd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime as dt
import time
from datetime import datetime, timedelta

country = 'United States'
city_name = 'Houston'
filename = '/Users/carbon-monitor-cities-%s-v0105.csv' %(country) 

def plot_CM_cities(df, days, city_name, x_lim, year):    
    ax = plt.figure(figsize=(18,12))
    for i in range(0,len(city_name)):     
        ax = ax.add_subplot(2,2,i+1)
        ax.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Power'), "value"],'r', label='Power')
        ax.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Industry'), "value"],'k', label='Industry')
        ax.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Residential'), "value"],'b', label='Residential')
        ax.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Ground Transport'), "value"],'g', label='Ground Transport')
        ax.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Aviation'), "value"],'y', label='Aviation')
        ax.set_xlim(0, x_lim)  
        ax.set_ylabel('Daily $CO_2$ Emissions (kt $CO_2$)', fontsize = 14)
        ax.set_title((city_name +'  ' + year), fontsize = 14)
        # Define the date format
        date_form = DateFormatter("%b-%d")
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.legend(prop={'size': 10},loc = 'upper right')  
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)    

#============ Load data ==================
data = pd.read_csv(filename)
year = '(Jan 2019-Jun 2021)'
days = np.arange(1, 365+366+181+1, 1)
x_lim = 365+366+181

#============ if formatted ================== 
plot_CM_cities(data, days, city_name, x_lim, year) 
