# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:32:37 2021

@author: Da Huo
"""
import numpy as np
import pandas as pd
import csv
import os, fnmatch
import sys
import math
import string
import copy
import calendar
import xlrd
import json
import codecs
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.ticker as plticker
from matplotlib.dates import DateFormatter
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from numpy.random import rand

emission_factors = {
    "Raw Coal": 1.991/0.7143,
    "Fuel Oil": 3.181,
    "Natural Gas": 21.671    
    }

# plant lifecycle emission intensity kgCO2e/kWh
# http://www.world-nuclear.org/uploadedFiles/org/WNA/Publications/Working_Group_Reports/comparison_of_lifecycle.pdf
plant_emission_intensity = {
    "Lignite": 1.054,
    "Coal": 0.888,
    "Oil": 0.733,
    "Natural Gas": 0.499,
    "Solar": 0.085,
    "Biomass": 0.045,
    "Nuclear": 0.029,
    "Hydroelectric": 0.026,
    "Wind": 0.026
    }


path = '/Users/danie/Desktop/Beijing_energy_systems/data_processing_python/'
os.chdir(path)
listOfFiles = os.listdir(path)
data = pd.read_excel("power_plants.xlsx" , sheet_name= "2000", index_col=0)
plant_names = data.index


years = ['1990','1997','2000','2002','2004','2005','2006','2007','2008',
               '2009','2010','2011','2012','2015','2016','2017',
               '1996-2000','2001-2005','2006-2010','2011-2015','2016-2020']

plant1 = "北京京能热电股份有限公司"
#plant1 = "郑常庄电厂"

# dataframe for saving results
# result = {'year':years,
#        'Power Generation':np.zeros((len(years),1)).tolist(),
#        'Self Consumption (%)':np.zeros((len(years),1)).tolist(),
#        'Consumption-Raw Coal':np.zeros((len(years),1)).tolist(),
#        'Consumption-Fuel Oil':np.zeros((len(years),1)).tolist(),
#        'Consumption-Natural Gas':np.zeros((len(years),1)).tolist(),
#        'Coal Consumption for Power Supply (g/kwh)':np.zeros((len(years),1)).tolist(),
#        'Emission Intensity (kgCO2/kwh)':np.zeros((len(years),1)).tolist(),
#        }

# result = pd.DataFrame(result)
cols=['Power Generation','Coal Consumption for Power Supply (g/kwh)','Emission Intensity (kgCO2/kwh)']
result = pd.DataFrame([[0, 0, 0] for x in range(len(years))], columns = cols)
#result.insert(loc=0, column='A', value=new_col)

result[cols] = result[cols].apply(pd.to_numeric, errors='coerce', axis=1)

result['year'] = years
result = result.set_index('year')
#result = result.astype(float)
#result.replace(np.nan,0)

for year in years[0:16]:
    df = pd.read_excel("power_plants.xlsx", sheet_name=year, index_col=0)
    # check if a specific plant exists in dataframe
    if plant1 in df.index:
        result.loc[year, 'Power Generation'] = df.loc[plant1, 'Power Generation (1e4kwh)']
        result.loc[year, 'Coal Consumption for Power Supply (g/kwh)'] = df.loc[plant1, 'Coal Consumption for Power Supply (g/kwh)']
        result.loc[year,'Emission Intensity (kgCO2/kwh)'] = result.loc[year,'Coal Consumption for Power Supply (g/kwh)']*emission_factors["Raw Coal"]/1e3
    
#result = result.drop(['1990', '1997', '2016', '2017'])
for year in years[0:16]:
    if 1996<=int(year)<=2000:
        result.loc['1996-2000', 'Power Generation'] += result.loc[year, 'Power Generation']
        result.loc['1996-2000', 'Emission Intensity (kgCO2/kwh)'] += result.loc[year, 'Emission Intensity (kgCO2/kwh)']
    elif 2001<=int(year)<=2005:
        result.loc['2001-2005', 'Power Generation'] += result.loc[year, 'Power Generation']
        result.loc['2001-2005', 'Emission Intensity (kgCO2/kwh)'] += result.loc[year, 'Emission Intensity (kgCO2/kwh)']
    elif 2006<=int(year)<=2010:
        result.loc['2006-2010', 'Power Generation'] += result.loc[year, 'Power Generation']
        result.loc['2006-2010', 'Emission Intensity (kgCO2/kwh)'] += result.loc[year, 'Emission Intensity (kgCO2/kwh)']
    elif 2011<=int(year)<=2015:
        result.loc['2011-2015', 'Power Generation'] += result.loc[year, 'Power Generation']
        result.loc['2011-2015', 'Emission Intensity (kgCO2/kwh)'] += result.loc[year, 'Emission Intensity (kgCO2/kwh)']
    # if 2016<=int(year)<=2020:
    #     result.loc['2016-2020', 'Power Generation'] += result.loc[year, 'Power Generation']
    #     result.loc['2016-2020', 'Emission Intensity (kgCO2/kwh)'] += result.loc[year, 'Emission Intensity (kgCO2/kwh)']
       
#result.loc['1996-2000', 'Emission Intensity (kgCO2/kwh)'] /= 1
result.loc['2001-2005', 'Emission Intensity (kgCO2/kwh)'] /= 3
result.loc['2006-2010', 'Emission Intensity (kgCO2/kwh)'] /= 5
result.loc['2011-2015', 'Emission Intensity (kgCO2/kwh)'] /= 3
#result.loc['2016-2020', 'Emission Intensity (kgCO2/kwh)'] /= 3


result_5year = result.drop(years[0:17])

# plot as bar plots
plt.bar(result_5year.index, height = result_5year['Emission Intensity (kgCO2/kwh)'],
        width = result_5year['Power Generation']/5e6)

plt.ylabel('Emission Intensity (kgCO2/kwh)')
plt.legend()
plt.show()

#%% Plot all **Thermal+ renewable**  plants at 2000, 2005, 2010, 2015
data = pd.read_excel("power_plants3.xlsx" , sheet_name= "all_renewable_plants", index_col=0)
renewable_plants = data.index

data = pd.read_excel("power_plants3.xlsx" , sheet_name= "all_thermal_plants", index_col=0)
thermal_plants = data.index
thermal_plants_index = np.arange(0,len(thermal_plants))

#data = data.drop("Shougang Power Plant")
#data = data.drop("Petro Qianjin Plant")

years = ['2000','2005','2010','2015']
#data['Power Generation' + str(years[1])] = np.zeros((len(thermal_plants),1))

# data2000 = pd.read_excel("power_plants2.xlsx" , sheet_name= "2000", index_col=0)
# data2005 = pd.read_excel("power_plants2.xlsx" , sheet_name= "2005", index_col=0)
# data2010 = pd.read_excel("power_plants2.xlsx" , sheet_name= "2010", index_col=0)
# data2015 = pd.read_excel("power_plants2.xlsx" , sheet_name= "2015", index_col=0)
for year in years:
    data.loc['renewable', 'Power Generation'+ str(year)] = 0


for year in years:
    df = pd.read_excel("power_plants3.xlsx", sheet_name=year, index_col=0)
    for plant1 in thermal_plants:
    # check if a specific plant exists in dataframe
        if plant1 in df.index:
            data.loc[plant1, 'Power Generation'+ str(year)] = df.loc[plant1, 'Power Generation (1e4kwh)']
            data.loc[plant1, 'Coal Consumption for Power Supply (g/kwh)'+ str(year)] = df.loc[plant1, 'Coal Consumption for Power Supply (g/kwh)']
            data.loc[plant1,'Emission Intensity (kgCO2/kwh)'+ str(year)] = data.loc[plant1,'Coal Consumption for Power Supply (g/kwh)'+ str(year)]*emission_factors["Raw Coal"]/1e3
    for plant2 in renewable_plants:
        if plant2 in df.index:
            #data.loc['renewable', 'Power Generation'+ str(year)] += df.loc[plant2, 'Power Generation (1e4kwh)']
            data.loc['renewable', 'Power Generation'+ str(year)] = df.loc['All Hydropower Plants', 'Power Generation (1e4kwh)']+df.loc['All Wind Power Plants', 'Power Generation (1e4kwh)']+df.loc['All Solar Power Plants', 'Power Generation (1e4kwh)']
            data.loc['renewable','Emission Intensity (kgCO2/kwh)'+ str(year)] = plant_emission_intensity['Hydroelectric']
        
        
ax = plt.figure(figsize=(12,24))

# Get a color for each bar
#rgb = rand(len(thermal_plants_index),3)
rgb=np.load('/Users/danie/Desktop/Beijing_energy_systems/data_processing_python/colorbar_for_plants.npy')

plant_names = []
for i in range(0,len(thermal_plants_index)):
    plant_names.append('T' + str(i))

for i in range(0,len(years)):
    ax1 = ax.add_subplot(4,1,i+1) 
    ax1.bar(plant_names, height = data['Emission Intensity (kgCO2/kwh)'+str(years[i])],
            width = data['Power Generation'+str(years[i])]/4e5, color=rgb)
   
    #y_avg = [np.nanmean(data['Emission Intensity (kgCO2/kwh)'+str(years[i])])] * len(thermal_plants_index)
    y_avg = [np.nansum(data['Emission Intensity (kgCO2/kwh)'+str(years[i])]*data['Power Generation'+str(years[i])])/np.nansum(data['Power Generation'+str(years[i])])] * len(thermal_plants_index)
    data2 = data.drop(['renewable'])
    y_avg_thermal = [np.nansum(data2['Emission Intensity (kgCO2/kwh)'+str(years[i])]*data2['Power Generation'+str(years[i])])/np.nansum(data2['Power Generation'+str(years[i])])] * len(thermal_plants_index)
    
    ax1.plot(thermal_plants_index, y_avg, color='green', lw=3, ls='--', label="average plot")
    ax1.plot(thermal_plants_index, y_avg_thermal, color='red', lw=3, ls='--', label="average plot")
    ax1.set(
        ylabel='Emission Intensity (kgCO2/kwh)',
        xlim=(thermal_plants_index.min(), thermal_plants_index.max()),
        ylim=(0,2.5),
        xticks=(thermal_plants_index),
        title=("year " + str(years[i])))    
      
    ax1.yaxis.label.set_size(18)
    ax1.title.set_size(20)
    
#%% Plot all **Renewable** plants at 2000, 2005, 2010, 2015
data = pd.read_excel("power_plants3.xlsx" , sheet_name= "all_renewable_plants2", index_col=0)
thermal_plants = data.index
thermal_plants_index = np.arange(0,len(thermal_plants))

#data = data.drop("Shougang Power Plant")
#data = data.drop("Petro Qianjin Plant")

years = ['2000','2005','2010','2015']
#data['Power Generation' + str(years[1])] = np.zeros((len(thermal_plants),1))

# data2000 = pd.read_excel("power_plants2.xlsx" , sheet_name= "2000", index_col=0)
# data2005 = pd.read_excel("power_plants2.xlsx" , sheet_name= "2005", index_col=0)
# data2010 = pd.read_excel("power_plants2.xlsx" , sheet_name= "2010", index_col=0)
# data2015 = pd.read_excel("power_plants2.xlsx" , sheet_name= "2015", index_col=0)

for year in years:
    df = pd.read_excel("power_plants3.xlsx", sheet_name=year, index_col=0)
    for plant1 in thermal_plants:
    # check if a specific plant exists in dataframe
        if plant1 in df.index:
            data.loc[plant1, 'Power Generation'+ str(year)] = df.loc[plant1, 'Power Generation (1e4kwh)']
            data.loc[plant1,'Emission Intensity (kgCO2/kwh)'+ str(year)] = plant_emission_intensity['Hydroelectric']
        
ax = plt.figure(figsize=(8,24))

# Get a color for each bar
#rgb = rand(len(thermal_plants_index),3)
rgb=np.load('/Users/danie/Desktop/Beijing_energy_systems/data_processing_python/colorbar_for_renewable_plants.npy')

# plant_names = []
# for i in range(0,len(thermal_plants_index)):
#     plant_names.append('R' + str(i))

plant_names = ['Hydropower Plants', 'Wind Power Plants','Solar Power Plants', ' ']

for i in range(0,len(years)):
    ax1 = ax.add_subplot(4,1,i+1) 
    ax1.bar(plant_names, height = data['Emission Intensity (kgCO2/kwh)'+str(years[i])],
            width = data['Power Generation'+str(years[i])]/0.7e5, color=rgb,align='center')
   
    #y_avg = [np.nanmean(data['Emission Intensity (kgCO2/kwh)'+str(years[i])])] * len(thermal_plants_index)
    y_avg = [np.nansum(data['Emission Intensity (kgCO2/kwh)'+str(years[i])]*data['Power Generation'+str(years[i])])/np.nansum(data['Power Generation'+str(years[i])])] * len(thermal_plants_index)
    
    ax1.plot(thermal_plants_index, y_avg, color='red', lw=3, ls='--', label="average plot")
    ax1.set(
        ylabel='Emission Intensity (kgCO2/kwh)',
        xlim=(thermal_plants_index.min(), thermal_plants_index.max()),
        ylim=(0,0.03),
        xticks=(thermal_plants_index),
        title=("year " + str(years[i]))) 
    ax1.yaxis.label.set_size(18)
    ax1.xaxis.label.set_size(18)
    ax1.title.set_size(20)    
    

# fig, (ax1, ax2) = plt.subplots(2)
# #fig = plt.figure(figsize=(8, 6))
# ax1.bar(thermal_plants_index, height = data['Emission Intensity (kgCO2/kwh)2000'],
#         width = data['Power Generation2000']/4e5)

# ax1.ylabel('Emission Intensity (kgCO2/kwh)')
# ax1.xlim(thermal_plants_index.min(), thermal_plants_index.max())
# ax1.xticks(thermal_plants_index)
# #plt.legend()
# #plt.show()

# #fig = plt.figure(figsize=(8, 6))
# ax2.bar(thermal_plants_index, height = data['Emission Intensity (kgCO2/kwh)2005'],
#         width = data['Power Generation2005']/4e5)

# ax2.ylabel('Emission Intensity (kgCO2/kwh)')
# ax2.xlim(thermal_plants_index.min(), thermal_plants_index.max())
# ax2.xticks(thermal_plants_index)
# #plt.legend()
# #plt.show()

# fig = plt.figure(figsize=(8, 6))
# plt.bar(thermal_plants_index, height = data['Emission Intensity (kgCO2/kwh)2010'],
#         width = data['Power Generation2010']/4e5)

# plt.ylabel('Emission Intensity (kgCO2/kwh)')
# plt.xlim(thermal_plants_index.min(), thermal_plants_index.max())
# plt.xticks(thermal_plants_index)
# #plt.legend()
# plt.show()

# fig = plt.figure(figsize=(8, 6))
# plt.bar(thermal_plants_index, height = data['Emission Intensity (kgCO2/kwh)2015'],
#         width = data['Power Generation2015']/4e5)

# plt.ylabel('Emission Intensity (kgCO2/kwh)')
# plt.xlim(thermal_plants_index.min(), thermal_plants_index.max())
# plt.xticks(thermal_plants_index)
# #plt.legend()
# plt.show()


