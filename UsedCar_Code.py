# %%
import pandas as pd
import numpy as np
#%%

# %%
def dfChkBasics(dframe, valCnt = False): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(f'\n{cnt}: describe(): ')
  cnt+=1
  print(dframe.describe())

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1
# %%
# Dataset
# The data in this project contains 31 variables and more than 200,000 used cars' data. 
# 15 variables are anonymous we will drop them later. 
# 150,000 observations will be train set and 50,000 observation will be test set.
# %%
df=pd.read_csv("used_car_train_20200313.csv")
#%%
# Data Processing
#
# Drop meaningless variables
df=df.iloc[:,:16]
df=df.drop(columns='name')
# 
# %%
dfChkBasics(df, valCnt= True)
# %%
# Q1 Car Attribute
# Research the influence of a vehicle’s 
# Brand, Model, Body Type, Fuel Type and Gearbox Type
# on the value of the vehicle.
# %%
# Q2 Car Damage
# Research the influence of a vehicle’s 
# Milage, Engine power, whether the car has been Damaged
# on the value of the vehicle.

# %%
# EDA
dfQ2 = 

# %%
# Q3 Market Behavior
# Research the influence of 
# Seller and Offer type 
# on the value of the vehicle.

# %%
# Conclusion