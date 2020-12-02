
#%%
#os.chdir(r'F:\Github\6103-Final-Project')
# %%
import pandas as pd
import numpy as np
import pylab as py
import scipy.stats as stats
import statsmodels.api as sm
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
# Origin dataset is too large, so we only research on sample have size 20000.
df=df.sample(n=20000,random_state=2020)
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

# Select col and drop NA
dfQ2 = df.iloc[:,[0,7,8,9,14]]
dfQ2.set_index('SaleID')
dfQ2.replace('-',np.nan,inplace=True)
dfQ2=dfQ2.dropna()

#%%
# Change the data type
dfQ2["power"] = dfQ2["power"].astype('float')
dfQ2["kilometer"] = dfQ2["kilometer"].astype('float')
dfQ2["notRepairedDamage"] = dfQ2["notRepairedDamage"].astype('int')
#%%
# Change 'notRepairedDamage' to category that only have 0 or 1.
dfQ2=dfQ2[(dfQ2['notRepairedDamage'] == 0) | (dfQ2['notRepairedDamage'] == 1)]
dfQ2["notRepairedDamage"] = dfQ2["notRepairedDamage"].astype('category')
#%%
# Drop abnormal value in power, kilometer and price.
dfQ2=dfQ2.drop(dfQ2[dfQ2['power']==0].index)
dfQ2=dfQ2.drop(dfQ2[dfQ2['kilometer']==0].index)
dfQ2=dfQ2.drop(dfQ2[dfQ2['price']==float].index)

# %%
# Drop outliers
dfQ2['power'] = dfQ2['power'][dfQ2['power'].between(dfQ2['power'].quantile(.05), dfQ2['power'].quantile(.95))]
dfQ2['kilometer'] = dfQ2['kilometer'][dfQ2['kilometer'].between(dfQ2['kilometer'].quantile(.05), dfQ2['kilometer'].quantile(.95))]
dfQ2['price'] = dfQ2['price'][dfQ2['price'].between(dfQ2['price'].quantile(.05), dfQ2['price'].quantile(.95))]
#%%
# QQPlot    
stats.probplot(dfQ2['power'], dist="norm", plot=py)
py.show()
stats.probplot(dfQ2['kilometer'], dist="norm", plot=py)
py.show()
stats.probplot(dfQ2['price'], dist="norm", plot=py)
py.show()
#%%
# Histogram

#%%
# Boxplot
#%%
dfChkBasics(dfQ2, valCnt= True)
# %%
# Q3 Market Behavior
# Research the influence of 
# Seller and Offer type 
# on the value of the vehicle.

# %%
# Conclusion
dfQ2
# %%
