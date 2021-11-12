# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:21:12 2021
@author: DH
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
# import datetime as dt
# import time
# from datetime import datetime, timedelta
# import re

path = '/Users/danie/Desktop/Beijing_energy_systems/data_processing_python/'
os.chdir(path)
listOfFiles = os.listdir(path)
#for entry in listOfFiles:
data_GHG = pd.read_excel('Beijing-data-0412.xlsx', sheet_name='GHG Intensity', index_col=0)  
data_EE = pd.read_excel('Beijing-data-0412.xlsx', sheet_name='Efficiency&Electrification', index_col=0)  
data_Decomp = pd.read_excel('Beijing-data-0412.xlsx', sheet_name='Decomposition', index_col=0)  
data_LMDI = pd.read_excel('Beijing-data-0412.xlsx', sheet_name='LMDI decomposition', index_col=0)

data_EE_v2 = pd.read_excel('Beijing-data-0412a.xlsx', sheet_name='Efficiency&Electrification', index_col=0)
# convertion constants from CEADS physical quantity to tce
CEADs_tce_convertion = {
    "Raw Coal": 0.7143,
    "Cleaned Coal":	0.9,
    "Other Washed Coal":0.285,
    "Briquettes":0.6,
    "Coke":	0.9714,
    "Coke Oven Gas":6.143,
    "Other Gas":3.5701,
    "Other Coking Products":1.3,
    "Crude Oil": 1.4286,
    "Gasoline":	1.4714,
    "Kerosene":	1.4714,
    "Diesel Oil":1.4571,
    "Fuel Oil":	1.4286,
    "LPG": 1.7143,
    "Refinery Gas": 1.5714,
    "Other Petroleum Products": 1.2,
    "Natural Gas": 13.3,
    "Heat": 0.0341,
    "Electricity": 1.229,
    "Other Energy": 1
}

conv2tce = list(CEADs_tce_convertion.values())

emission_factors = {
    "Raw Coal": 1.991,
    "Fuel Oil": 3.181,
    "Natural Gas": 21.671    
    }

#print(data.iloc[0,2])

#print(data_EE.columns)

# rename columns
#df.columns = df.columns.str.replace('A', 'a')
#decomp = pd.DataFrame([[, '', '', '',''] for x in range(data.shape[0]*num_sect)], columns=['city','country','date','sector','value (KtCO2 per day)','timestamp'])

# constants for converting units
tce2mj = 29307 # 1 tce = 29307 MJ
# ============== GHG Intensity ==================
# compute from raw data
GHG_stats= list(data_GHG.index)
year = data_GHG.columns

# get first value
#data_GHG[years[0]][GHG_stats[0]]

# BJ local elec supply (without power plants)
data_GHG.iloc[2] = data_GHG.iloc[0]*(100-data_GHG.iloc[1])/100
data_GHG.iloc[4] = data_GHG.iloc[3] - data_GHG.iloc[5]
data_GHG.iloc[5] = data_GHG.iloc[3] * data_GHG.iloc[9]/100
data_GHG.iloc[7] = data_GHG.iloc[2] + data_GHG.iloc[3] - data_GHG.iloc[6]
data_GHG.iloc[8] = data_GHG.iloc[2] / data_GHG.iloc[7]*100
data_GHG.iloc[12] = data_GHG.iloc[10] * (100 - data_GHG.iloc[11])/100
data_GHG.iloc[14] = data_GHG.iloc[13]/data_GHG.iloc[12]/10
data_GHG.iloc[17] = data_GHG.iloc[15]*(100-data_GHG.iloc[16])/100
data_GHG.iloc[19] = data_GHG.iloc[18]/data_GHG.iloc[17]/10
data_GHG.iloc[21] = data_GHG.iloc[20]/data_GHG.iloc[2]/10
data_GHG.iloc[22] = data_GHG.iloc[4]/data_GHG.iloc[12]*data_GHG.iloc[13]
data_GHG.iloc[23] = data_GHG.iloc[5]/data_GHG.iloc[17]*data_GHG.iloc[18]
data_GHG.iloc[24] = (data_GHG.iloc[20]+data_GHG.iloc[22]+data_GHG.iloc[23])/(data_GHG.iloc[7]+data_GHG.iloc[6])/10
data_GHG.iloc[25] = (data_GHG.iloc[20]+data_GHG.iloc[22]+data_GHG.iloc[23])/100

# ========= Efficiency and Electrification ======
#data_EE.iloc[0] = data_EE.iloc[1] / 1.0935
#data_EE.iloc[2] = data_EE.iloc[1] / 1.0935
data_EE.iloc[3] = data_GHG.iloc[7]
data_EE.iloc[4] = data_EE.iloc[3]*3600*100/(data_EE.iloc[0]*29307*10)
data_EE.iloc[11] = (data_EE.iloc[9]/data_EE.iloc[10]) / (data_EE.iloc[9,0]/data_EE.iloc[10,0])
data_EE.iloc[12] = (data_EE.iloc[0]/data_EE.iloc[10]) / (data_EE.iloc[0,0]/data_EE.iloc[10,0])
data_EE.iloc[14] = data_EE.iloc[3]/data_EE.iloc[13]*10000
data_EE.iloc[15] = data_EE.iloc[14]*3.6/1000
data_EE.iloc[16] = data_EE.iloc[0]/data_EE.iloc[13]*tce2mj/1000
data_EE.iloc[17] = data_EE.iloc[9]/data_EE.iloc[13]/data_EE.iloc[11]*10000

# ========== Decomposition basic data ==========
data_Decomp.loc['GHGs, Mt'] = data_GHG.iloc[20]/100#data_GHG.iloc[25] #* (data_GHG.iloc[0]/(data_GHG.iloc[0]+data_GHG.iloc[3]))
data_Decomp.loc['Energy, 10k tce'] = data_EE.iloc[0]
data_Decomp.loc['GDP/POP'] = data_Decomp.loc['GDP index (1978=100)']/data_Decomp.loc['POP, 1000 persons']
data_Decomp.loc['Energy/GDP'] = data_EE.iloc[12]
data_Decomp.loc['Electricity/Energy, %'] = data_EE.iloc[4]
data_Decomp.loc['GHG Intensity, ton/MWh'] = data_GHG.iloc[21]#data_GHG.iloc[24]


# total energy based on coal+oil+gas from CEADs
# data_Decomp.loc['Energy, 10k tce'] = CEADs_tce_convertion["Raw Coal"]*data_Decomp.loc['Tot coal consump'] + \
#     CEADs_tce_convertion["Fuel Oil"]*data_Decomp.loc['Tot oil consump'] +\
#         CEADs_tce_convertion["Natural Gas"]*data_Decomp.loc['Tot gas consump']

data_Decomp.loc['GHG from non-elec coal, Mt'] = data_Decomp.loc['Coal consump non elec, 10kton']*emission_factors["Raw Coal"]/100
data_Decomp.loc['GHG from non-elec oil, Mt'] = data_Decomp.loc['Oil consump non elec, 10kton']*emission_factors["Fuel Oil"]/100
data_Decomp.loc['GHG from non-elec gas, Mt'] = data_Decomp.loc['Gas consump non elec, 1e8m3']*emission_factors["Natural Gas"]/100

data_Decomp.loc['Non-elec coal/Energy'] = data_Decomp.loc['Coal consump non elec, 10k tce']/data_Decomp.loc['Energy, 10k tce']
data_Decomp.loc['Non-elec oil/Energy'] = data_Decomp.loc['Oil consump non elec, 10k tce']/data_Decomp.loc['Energy, 10k tce']
data_Decomp.loc['Non-elec gas/Energy'] = data_Decomp.loc['Gas consump non elec, 10k tce']/data_Decomp.loc['Energy, 10k tce']
data_Decomp.loc['Elec coal oil gas/Energy'] = 1-data_Decomp.loc['Non-elec coal/Energy']-data_Decomp.loc['Non-elec oil/Energy']-data_Decomp.loc['Non-elec gas/Energy']

#data_Decomp.loc['Elec coal oil gas/Energy'] = (data_Decomp.loc['Coal for Elec, 10k tce']+data_Decomp.loc['Oil for Elec, 10k tce']+data_Decomp.loc['Gas for Elec, 10k tce'])/data_Decomp.loc['Energy, 10k tce']

#data_Decomp.loc['GHG Intensity non-elec coal']  = data_Decomp.loc['GHG Intensity, ton/MWh']

data_Decomp.loc['GHG Intensity non-elec coal'] = 100*data_Decomp.loc['GHG from non-elec coal, Mt']/data_Decomp.loc['Coal consump non elec, 10kton']
data_Decomp.loc['GHG Intensity non-elec oil']  = 100*data_Decomp.loc['GHG from non-elec oil, Mt']/data_Decomp.loc['Oil consump non elec, 10kton']
data_Decomp.loc['GHG Intensity non-elec gas'] = 10*data_Decomp.loc['GHG from non-elec gas, Mt']/data_Decomp.loc['Gas consump non elec, 1e8m3']

#data_Decomp.loc['GHG Intensity elec'] = 100*data_Decomp.loc['GHGs, Mt']/data_Decomp.loc['Energy, 10k tce']

# ========== LMDI Decomposition ==========
#Modified Kaya Identity
#GHG from electricity = POP * GDP/POP * Energy/GDP * Electricity/Energy * GHG/Electricity
years = data_LMDI.columns
step = 1
for i in range(0, len(years)-step+1):    
    data_LMDI.loc['delta GHGs', years[i]] = data_Decomp.iloc[0, i+step] - data_Decomp.iloc[0, i]
    data_LMDI.loc['delta ln(GHGs)', years[i]] = math.log(data_Decomp.iloc[0, i+step]) - math.log(data_Decomp.iloc[0, i])
    data_LMDI.loc['delta ln(POP)', years[i]] = math.log(data_Decomp.iloc[1, i+step]) - math.log(data_Decomp.iloc[1, i])
    data_LMDI.loc['delta ln(GDP/POP)', years[i]] = math.log(data_Decomp.iloc[4, i+step]) - math.log(data_Decomp.iloc[4, i])
    data_LMDI.loc['delta ln(Energy/GDP)', years[i]] = math.log(data_Decomp.iloc[5, i+step]) - math.log(data_Decomp.iloc[5, i])
    data_LMDI.loc['delta ln(Electrification)', years[i]] = math.log(data_Decomp.iloc[6, i+step]) - math.log(data_Decomp.iloc[6, i])
        
    data_LMDI.loc['delta ln(GHG Intensity)', years[i]] = math.log(data_Decomp.iloc[7, i+step]) - math.log(data_Decomp.iloc[7, i])
    data_LMDI.loc['delta ln(GHG Intensity non-elec coal)'] = math.log(data_Decomp.loc['GHG Intensity non-elec coal', year[i+step]]) - math.log(data_Decomp.loc['GHG Intensity non-elec coal', year[i]])
    data_LMDI.loc['delta ln(GHG Intensity non-elec oil)'] = math.log(data_Decomp.loc['GHG Intensity non-elec oil', year[i+step]]) - math.log(data_Decomp.loc['GHG Intensity non-elec oil', year[i]])
    data_LMDI.loc['delta ln(GHG Intensity non-elec gas)'] = math.log(data_Decomp.loc['GHG Intensity non-elec gas', year[i+step]]) - math.log(data_Decomp.loc['GHG Intensity non-elec gas', year[i]])
    
    data_LMDI.loc['delta ln(Elec coal oil gas/Energy)', years[i]] = math.log(data_Decomp.loc['Elec coal oil gas/Energy',year[i+step]]) - math.log(data_Decomp.loc['Elec coal oil gas/Energy', year[i]])
    data_LMDI.loc['delta ln(Non-elec coal/Energy)', years[i]] = math.log(data_Decomp.loc['Non-elec coal/Energy', year[i+step]]) - math.log(data_Decomp.loc['Non-elec coal/Energy', year[i]])
    data_LMDI.loc['delta ln(Non-elec oil/Energy)', years[i]] = math.log(data_Decomp.loc['Non-elec oil/Energy', year[i+step]]) - math.log(data_Decomp.loc['Non-elec oil/Energy', year[i]])
    data_LMDI.loc['delta ln(Non-elec gas/Energy)', years[i]] = math.log(data_Decomp.loc['Non-elec gas/Energy', year[i+step]]) - math.log(data_Decomp.loc['Non-elec gas/Energy', year[i]])

    data_LMDI.loc['delta GHG from non-elec coal', years[i]] = data_Decomp.loc['GHG from non-elec coal, Mt', year[i+step]] - data_Decomp.loc['GHG from non-elec coal, Mt', year[i]]
    data_LMDI.loc['delta ln(GHG from non-elec coal)', years[i]] = math.log(data_Decomp.loc['GHG from non-elec coal, Mt', year[i+step]]) - math.log(data_Decomp.loc['GHG from non-elec coal, Mt', year[i]])
    data_LMDI.loc['delta GHG from non-elec oil', years[i]] = data_Decomp.loc['GHG from non-elec oil, Mt', year[i+step]] - data_Decomp.loc['GHG from non-elec oil, Mt', year[i]]
    data_LMDI.loc['delta ln(GHG from non-elec oil)', years[i]] = math.log(data_Decomp.loc['GHG from non-elec oil, Mt', year[i+step]]) - math.log(data_Decomp.loc['GHG from non-elec oil, Mt', year[i]])
    data_LMDI.loc['delta GHG from non-elec gas', years[i]] = data_Decomp.loc['GHG from non-elec gas, Mt', year[i+step]] - data_Decomp.loc['GHG from non-elec gas, Mt', year[i]]
    data_LMDI.loc['delta ln(GHG from non-elec gas)', years[i]] = math.log(data_Decomp.loc['GHG from non-elec gas, Mt', year[i+step]]) - math.log(data_Decomp.loc['GHG from non-elec gas, Mt', year[i]])
            
data_LMDI.loc['L elec'] =  data_LMDI.loc['delta GHGs'] / data_LMDI.loc['delta ln(GHGs)']
data_LMDI.loc['L non elec coal'] =  data_LMDI.loc['delta GHG from non-elec coal'] / data_LMDI.loc['delta ln(GHG from non-elec coal)'] 
data_LMDI.loc['L non elec oil'] =  data_LMDI.loc['delta GHG from non-elec oil'] / data_LMDI.loc['delta ln(GHG from non-elec oil)']
data_LMDI.loc['L non elec gas'] =  data_LMDI.loc['delta GHG from non-elec gas'] / data_LMDI.loc['delta ln(GHG from non-elec gas)'] 

data_LMDI.loc['Contribution(POP)'] = data_LMDI.loc['L elec'] * data_LMDI.loc['delta ln(POP)'] + \
    data_LMDI.loc['L non elec coal'] * data_LMDI.loc['delta ln(POP)'] + \
        data_LMDI.loc['L non elec oil'] * data_LMDI.loc['delta ln(POP)'] +\
            data_LMDI.loc['L non elec gas'] * data_LMDI.loc['delta ln(POP)']

data_LMDI.loc['Contribution(GDP/POP)'] = data_LMDI.loc['L elec'] * data_LMDI.loc['delta ln(GDP/POP)'] + \
    data_LMDI.loc['L non elec coal'] * data_LMDI.loc['delta ln(GDP/POP)'] + \
        data_LMDI.loc['L non elec oil'] * data_LMDI.loc['delta ln(GDP/POP)'] +\
            data_LMDI.loc['L non elec gas'] * data_LMDI.loc['delta ln(GDP/POP)']

data_LMDI.loc['Contribution(Energy/GDP)'] = data_LMDI.loc['L elec'] * data_LMDI.loc['delta ln(Energy/GDP)'] + \
    data_LMDI.loc['L non elec coal'] * data_LMDI.loc['delta ln(Energy/GDP)'] + \
        data_LMDI.loc['L non elec oil'] * data_LMDI.loc['delta ln(Energy/GDP)'] +\
            data_LMDI.loc['L non elec gas'] * data_LMDI.loc['delta ln(Energy/GDP)']           

data_LMDI.loc['Contribution(Energy mix)'] = data_LMDI.loc['L elec'] * data_LMDI.loc['delta ln(Elec coal oil gas/Energy)'] + \
    data_LMDI.loc['L non elec coal'] * data_LMDI.loc['delta ln(Non-elec coal/Energy)'] + \
        data_LMDI.loc['L non elec oil'] * data_LMDI.loc['delta ln(Non-elec oil/Energy)'] +\
            data_LMDI.loc['L non elec gas'] * data_LMDI.loc['delta ln(Non-elec gas/Energy)'] 

data_LMDI.loc['Contribution(GHG intensity)'] = data_LMDI.loc['L elec'] * data_LMDI.loc['delta ln(GHG Intensity)'] + \
    data_LMDI.loc['L non elec coal'] * data_LMDI.loc['delta ln(GHG Intensity non-elec coal)'] + \
        data_LMDI.loc['L non elec oil'] * data_LMDI.loc['delta ln(GHG Intensity non-elec oil)'] +\
            data_LMDI.loc['L non elec gas'] * data_LMDI.loc['delta ln(GHG Intensity non-elec gas)'] 

            
#data_LMDI.loc['Contribution(GDP/POP)'] = data_LMDI.loc['L'] * data_LMDI.loc['delta ln(GDP/POP)']
#data_LMDI.loc['Contribution(Energy/GDP)'] = data_LMDI.loc['L'] * data_LMDI.loc['delta ln(Energy/GDP)']
#data_LMDI.loc['Contribution(Electrification)'] = data_LMDI.loc['L'] * data_LMDI.loc['delta ln(Electrification)']
#data_LMDI.loc['Contribution(GHG intensity)'] = data_LMDI.loc['L'] * data_LMDI.loc['delta ln(GHG intensity)']

#data_LMDI.loc['Contribution(Elec from coal/Energy)'] = data_LMDI.loc['L'] * data_LMDI.loc['delta ln(Elec from coal/Energy)'] 
#data_LMDI.loc['Contribution(Elec from oil/Energy)'] = data_LMDI.loc['L'] * data_LMDI.loc['delta ln(Elec from oil/Energy)']
#data_LMDI.loc['Contribution(Elec from gas/Energy)'] = data_LMDI.loc['L'] * data_LMDI.loc['delta ln(Elec from gas/Energy)']
#data_LMDI.loc['Contribution(Energy Mix Structure)'] = data_LMDI.loc['Contribution(Elec from coal/Energy)'] + data_LMDI.loc['Contribution(Elec from oil/Energy)']+ data_LMDI.loc['Contribution(Elec from gas/Energy)']

data_LMDI.loc['delta tot GHGs'] = data_LMDI.loc['delta GHGs']+data_LMDI.loc['delta GHG from non-elec coal']+data_LMDI.loc['delta GHG from non-elec oil']+data_LMDI.loc['delta GHG from non-elec gas'] 
data_LMDI.loc['sum to validate'] = data_LMDI.loc['Contribution(POP)']+data_LMDI.loc['Contribution(GDP/POP)']+data_LMDI.loc['Contribution(Energy/GDP)']+ data_LMDI.loc['Contribution(Energy mix)'] + data_LMDI.loc['Contribution(GHG intensity)']

data_LMDI.loc['(POP)%'] = data_LMDI.loc['Contribution(POP)'] / data_LMDI.loc['delta tot GHGs']
data_LMDI.loc['(GDP/POP)%'] = data_LMDI.loc['Contribution(GDP/POP)'] / data_LMDI.loc['delta tot GHGs']
data_LMDI.loc['(Energy/GDP)%'] = data_LMDI.loc['Contribution(Energy/GDP)'] / data_LMDI.loc['delta tot GHGs']
#data_LMDI.loc['(Electrification)%'] = data_LMDI.loc['Contribution(Electrification)'] / data_LMDI.loc['delta tot GHGs']
data_LMDI.loc['(Energy mix)%'] = data_LMDI.loc['Contribution(Energy mix)'] / data_LMDI.loc['delta tot GHGs']
data_LMDI.loc['(GHG intensity)%'] = data_LMDI.loc['Contribution(GHG intensity)'] / data_LMDI.loc['delta tot GHGs']

# ============== Decomposition results for each 5-year range==================
stats = ['Delta Total GHGs, Mt',
 'Contribution(POP)',
 'Contribution(GDP/POP)',
 'Contribution(Energy/GDP)',
 'Contribution(Energy mix)',
 'Contribution(GHG intensity)']

decomp = {'statistics':stats,
       '1995-2000':np.zeros((len(stats),1)).tolist(),
       '2000-2005':np.zeros((len(stats),1)).tolist(),
       '2005-2010':np.zeros((len(stats),1)).tolist(),
       '2010-2015':np.zeros((len(stats),1)).tolist(),
       '2015-2017':np.zeros((len(stats),1)).tolist(),
       'Cumulative impacts':np.zeros((len(stats),1)).tolist(),}

decomp = pd.DataFrame(decomp)
decomp = decomp.set_index('statistics')

#print(decomp['statistics'][0])
for i, col in enumerate(decomp.columns): 
    decomp[col][0] = sum(data_LMDI.iloc[37,5*i:5*(i+1)])
    decomp[col][1] = sum(data_LMDI.iloc[8,5*i:5*(i+1)])
    decomp[col][2] = sum(data_LMDI.iloc[9,5*i:5*(i+1)])
    decomp[col][3] = sum(data_LMDI.iloc[10,5*i:5*(i+1)])
    decomp[col][4] = sum(data_LMDI.iloc[36,5*i:5*(i+1)])
    decomp[col][5] = sum(data_LMDI.iloc[12,5*i:5*(i+1)])
    
    if col == '2015-2017':
        decomp[col][0] = sum(data_LMDI.iloc[37,5*i:5*i+2])
        decomp[col][1] = sum(data_LMDI.iloc[8,5*i:5*i+2])
        decomp[col][2] = sum(data_LMDI.iloc[9,5*i:5*i+2])
        decomp[col][3] = sum(data_LMDI.iloc[10,5*i:5*i+2])
        decomp[col][4] = sum(data_LMDI.iloc[36,5*i:5*i+2])
        decomp[col][5] = sum(data_LMDI.iloc[12,5*i:5*i+2])

decomp['Cumulative impacts'][0] = sum(decomp.iloc[0,0:5])
decomp['Cumulative impacts'][1] = sum(decomp.iloc[1,0:5])/decomp['Cumulative impacts'][0]
decomp['Cumulative impacts'][2] = sum(decomp.iloc[2,0:5])/decomp['Cumulative impacts'][0]
decomp['Cumulative impacts'][3] = sum(decomp.iloc[3,0:5])/decomp['Cumulative impacts'][0]
decomp['Cumulative impacts'][4] = sum(decomp.iloc[4,0:5])/decomp['Cumulative impacts'][0]
decomp['Cumulative impacts'][5] = sum(decomp.iloc[5,0:5])/decomp['Cumulative impacts'][0]
     
#print(sum(data_LMDI.iloc[0,1:5+1]))

data = decomp.drop(columns=['Cumulative impacts']).drop(index=['Delta Total GHGs, Mt'])
data = data.transpose()
#plt.figure(figsize=(800, 600), dpi=600)
data.plot.bar()
plt.ylim((-40, 63))
#plt.tight_layout() 
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(fontsize =  'small')  
plt.xticks(rotation = 360)
plt.ylabel('Decomposition for Emission Changes (MtCO2)')
plt.show()

#%%
#plot: https://matplotlib.org/stable/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py
columns = data.columns
rows = data.index
data = data.values.tolist()

#%%
data2 = {'country':['AUE','USA','UK','USA'],
       'population':[10,12,100,200]}
df = pd.DataFrame(data2)

# get data for line 1
print(data2.iloc[0]+data2.iloc[1])

pop = df["population"]
pop[0:3]
pop.values[0:3]+pop.values[0:3]

dd = df.iloc[0:2, 0:2]

df.loc[df["country"] == "USA", "population"].mean()

# group by
df.groupby("country").sum()

df.insert(2, "Age", [21, 23, 24, 21]) 
df.set_index("country")

df.loc[df["Age"] >20, "country"]

df.groupby('country')['Age'].mean()

# sort
df.sort_values(by=["population"], ascending = [True], inplace=True)

# duplicate
df.drop_duplicates()

# copy first
df2 = df.copy()
#===============

# computation
df['Age'].sum()

# stats
df.corr()
df["Age"].value_counts(ascending = True)
bar = df["Age"].value_counts(ascending = True, bins = 5)
bar = bar.reset_index()
plt.bar(bar["level_0"], bar['Age'])

AgebyC = df.groupby('country')['Age'].mean()
AgebyC = AgebyC.reset_index()
plt.bar(AgebyC["country"], AgebyC['Age'])