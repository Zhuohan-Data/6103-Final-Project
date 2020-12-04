# %%
import pandas as pd
import numpy as np
import pylab as py
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("used_car_train_20200313.csv")


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
df=df.iloc[:,:16]
df=df.drop(columns='name')
dfChkBasics(df, valCnt= True)


# %%
# Q1 Car Attribute
# Research the influence of a vehicle’s 
# Brand, Model, Body Type, Fuel Type and Gearbox Type
# on the value of the vehicle.
dfQ1 = df.iloc[:,[0,1,2,3,4,5,6,7,14]]
dfQ1.set_index('SaleID')
dfQ1.replace('-',np.nan,inplace=True)
dfQ1=dfQ1.dropna()
dfChkBasics(dfQ1, valCnt= True)

# %%
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve

# %%
x1 = dfQ1[['model', 'brand', 'bodyType', 'fuelType', 'gearbox']]
y1 = dfQ1['price']
print(x1.head())
print(type(x1))
print(y1.head())
print(type(y1))

x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size = 0.20, random_state=20000)
full_split1 = linear_model.LinearRegression()
full_split1.fit(x_train1, y_train1)
y_pred1 = full_split1.predict(x_test1)
full_split1.score(x_test1, y_test1)

print('score (train):', full_split1.score(x_train1, y_train1)) 
print('score (test):', full_split1.score(x_test1, y_test1))
print('intercept:', full_split1.intercept_)
print('coef_:', full_split1.coef_)

logit = LogisticRegression() 
logit.fit(x_train1, y_train1)
print('Logit model accuracy (with the test set):', logit.score(x_test1, y_test1))
print('Logit model accuracy (with the train set):', logit.score(x_train1, y_train1)) 


knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(x1,y1)
y_pred = knn1.predict(x1)
print(y_pred)
print(knn1.score(x1,y1))
