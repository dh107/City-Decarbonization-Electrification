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
import re
from datetime import datetime, timedelta

path = '/Users/CN_eFUA_and_addtional_cities_2021_Jun/' # where data to be corrected
output_path ='/Users/CM_Cities_2022/cut/'

os.chdir(path)
listOfFiles = os.listdir(path)

for filename in listOfFiles:
    #============ if not formatted ==================
    year = re.split("[-.]", filename)[-2]
    city = re.split("[-]", filename)[-2]
    country = re.split("[-]", filename)[-3]
       
    if year == 'y2021':
        data = pd.read_csv(filename)
        
        df = data.copy()
        df.loc[(df['timestamp']>1625011200), 'value (KtCO2 per day)'] = ""
        df['value (KtCO2 per day)'].replace("", np.nan, inplace=True)
        df.dropna(subset = ["value (KtCO2 per day)"], inplace=True)
    
        df.to_csv(output_path + filename, index=False, encoding = 'utf_8_sig') 
