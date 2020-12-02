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
df=pd.read_csv("used_car_train_20200313.csv")
#%%
dfChkBasics(df)
#%%
# Data Processing
#
# Drop meaningless variables
df=df.iloc[:,:16]
df=df.drop(columns='name')
# 
# %%
df.shape
# %%
