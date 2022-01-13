#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import os
import re
import numpy as np
import time
import difflib
######################################################  路径修改 ################################################

Path_1 = 'C:/Users/'  #这里是所有txt文件的路径
Path_2 = 'C:/.csv'  #这里是数据大表的路径(需要被替换数据的文件)
Path_3 = 'C:/Users/'  #这里是输出的路径 result.csv可以改成想要的名字

Sector_Name = 'Ground Transport'                                                                                          #这里是要替换的sector的名字
 
################################################### 读取 所有的需要替换数据的txt ##########################################
filePath = Path_1
country = os.listdir(filePath)
data_name = []
for dbtype in country:
    if os.path.isfile(os.path.join(filePath,dbtype)):
        data_name.append(dbtype)
        
data_name = [data_name[i] for i,x in enumerate(data_name) if x.find('txt')!=-1] #只需要读取txt的数据

#将文件名中的城市名读取出来
City_Name = []
for i in data_name:
    title = re.compile(r'.*?_(?P<Name>.*?)[.]', re.S)

    City_Name_result = title.finditer(i) 
    for it in City_Name_result:
        City_Name.append(it.group("Name"))

result = []
for x,y in zip(data_name,City_Name):
    df = pd.read_csv(filePath+x, sep=',')
    df['city'] = y
    result.append(df)
df_all = pd.DataFrame(np.concatenate(result), columns = df.columns) 

##################################### 读取需要被替换数据的文件 并处理#########################################################
#要替换的源数据表
df_data = pd.read_csv(Path_2)

#一样统一格式

df_data['city'] = df_data['city'].str.title()
df_data['sector'] = df_data['sector'].str.title()

#统一日期格式
df_data_timestamp = df_data['timestamp'].tolist()
date_time = []
for x in df_data_timestamp:
    timeArray = time.gmtime(x)
    date_time.append(time.strftime("%Y-%m-%d", timeArray))
df_data['date'] = date_time
df_data = df_data.sort_values(by=['city','sector','date'],ascending=True,na_position='first').reset_index(drop = True)

#读取type为float的列名
for x in df_data.columns.tolist():
    if type(df_data[x].tolist()[0]) == float:
        Value_col = x
        

df_all = df_all.set_index(['city','Mon/Day']).stack().reset_index().rename(columns={'level_2':'year', 0:Value_col})

############################################# txt文件数据清洗并填补缺失值 ##################################
#将年份提取出来
year_list = df_all['year'].tolist()

Year_Name = []
for i in year_list:
    title = re.compile(r'\d{4}') #只保留四位数的数字 也就是年份
    Year_Name.append(title.findall(i)) 
Year_Name = [int(x) for item in Year_Name for x in item] #将结果转为int
df_all['year'] = Year_Name
#分离月和日
df_all = pd.concat([df_all[['city','year',Value_col]],df_all['Mon/Day'].str.split('-', expand=True).rename(columns = {0:'month',1:'day'})], axis = 1) 
#将年月日合并为日期格式
df_all['date'] = pd.to_datetime(df_all[['year', 'month','day']].assign(), errors='coerce')

#添加sector
df_all['sector'] = Sector_Name

#因为之前转换为日期格式是 加了coerce 所以可以直接定位date这一列有null值的行并删除
null_list = df_all[df_all[['date']].isnull().T.any()].index.tolist()

df_all = df_all.drop(null_list)

#只需要替换csv文件里的时间范围 这里超级low 记得修改

#第一天
first_date = df_data['date'].drop_duplicates().tolist()[0]
#最后一天
last_date = df_data['date'].drop_duplicates().tolist()[-1]
df_all = df_all[(df_all['date'] >= first_date) & (df_all['date'] <= last_date)].reset_index(drop = True)

df_all = df_all[['city','date','sector',Value_col]]

#将信息列统一为  字母大写
df_all['city'] = df_all['city'].str.title()
df_all['sector'] = df_all['sector'].str.title()

#只保留数字
df_all[Value_col] = df_all[Value_col].astype('float')

#按照城市和日期排序
df_all = df_all.sort_values(by=['city','date'],ascending=True,na_position='first').reset_index(drop = True)

#用前后两日的平均值填充null值
df_all[Value_col]=df_all[Value_col].fillna(df_all[Value_col].interpolate())
#数据单位转换
df_all[Value_col]=df_all[Value_col]/1000

################################################# 数据替换 ##########################################################
# drop empty rows                
df_data[Value_col].replace('', np.nan, inplace=True)
df_data.dropna(subset = [Value_col], inplace=True) 

#将两个文件中的城市名生成list
city_df_data = df_data['city'].drop_duplicates().tolist()
city_df_all = df_all['city'].drop_duplicates().tolist()
#模糊匹配并替换
n = 1.1
city_name_result = []
while len(city_name_result) != len(city_df_all):
    n = n - 0.1
    city_name_result.clear()
    for x in city_df_all:
        if difflib.get_close_matches(x, city_df_data, 1, cutoff=n) == []:
            pass
        else:
            city_name_result.append(difflib.get_close_matches(x, city_df_data, 1, cutoff=n))
    city_name_result = [str(x) for item in city_name_result for x in item]
print('######模糊匹配#######')
for x,y in zip(city_df_all,city_name_result):
    print(f'{x} 被模糊匹配为 {y}')
for x,y in zip(city_df_all,city_name_result):
    df_all['city'] = df_all['city'].str.replace(x,y)

print('#######定为国家######')

#保留相同城市名中的最大值的国家名
country_name_result = []
for x in city_name_result:
    result_max = []
    country_list = df_data[(df_data['sector'] == Sector_Name) & (df_data['city'].str.contains(x))]['country'].drop_duplicates().tolist()
    for y in country_list:
        value_data = df_data[(df_data['sector'] == Sector_Name) & (df_data['city'].str.contains(x)) & (df_data['country'] == y)][Value_col].sum()
        result_max.append(value_data)
    country_name_result.append(country_list[result_max.index(max(result_max))])
    print(f'城市 {x} 被定位到 {country_list[result_max.index(max(result_max))]} 国家')  

prev_x = ''    
#将相同的城市和日期的值替换
for x,y in zip(city_name_result,country_name_result):
    index_df_all = df_all[(df_all['city'] == x) &(df_all['sector'] == Sector_Name) &(df_all['date'] >= first_date) &(df_all['date'] <= last_date)].index.tolist()
    index_df_data = df_data[(df_data['city'].str.contains(x)) & (df_data['country'] == y) &(df_data['sector'] == Sector_Name) &(df_data['date'] >= first_date) &(df_data['date'] <= last_date)].index.tolist()
    for t,h in zip(index_df_data,index_df_all):
        if df_data.loc[t,Value_col] < df_all.loc[h,Value_col]:
            df_data.loc[t,Value_col] = df_all.loc[h,Value_col]
            if x != prev_x:
                print('replaced for', x)
                prev_x = x

#输出结果
df_data.to_csv(Path_3 +'carbon-monitor-London-cities-v0105-TomTom-corrected.csv', index = False, encoding = 'utf_8_sig')

# save by country
list_country = df_data['country'].drop_duplicates().tolist()
for ctry in list_country:   
    df_data_ctry = df_data.loc[df_data['country']==ctry]
    df_data_ctry.to_csv(Path_3 + 'carbon-monitor-cities-%s-v0105.csv' %(ctry), index = False, encoding = 'utf_8_sig')

print('Total number of countires:', len(list_country))
print('Total number of cities:', len(df_data['city'].drop_duplicates().tolist()))


















