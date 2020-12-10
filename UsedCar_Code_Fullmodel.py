# %%
import pandas as pd
import numpy as np
import pylab as py
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

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
# Read csv file
df=pd.read_csv("used_car_train_20200313.csv")
df=df.iloc[:,:16]
df=df.drop(columns='name')
dfChkBasics(df, valCnt= True)

# %%
# EDA
# Subsets and drop na
from statsmodels.formula.api import ols
dfFull = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,14]]
dfFull.set_index('SaleID')
dfFull.replace('-',np.nan,inplace=True)
dfFull = dfFull.dropna()
dfChkBasics(dfFull)
# %%
# Change data type and set gearbox to category that only have 0 or 1
dfFull['gearbox'] = dfFull['gearbox'].astype('float')
dfFull = dfFull[(dfFull['gearbox'] == 0) | (dfFull['gearbox'] == 1)]
dfFull['gearbox'] = dfFull['gearbox'].astype('int')

dfFull['power'] = dfFull['power'].astype('float')
dfFull['kilometer'] = dfFull['kilometer'].astype('float')
dfFull['notRepairedDamage'] = dfFull['notRepairedDamage'].astype('int')
dfFull = dfFull[dfFull['fuelType']<=6]
dfFull = dfFull[dfFull['bodyType']<=7]


#%%
# Change 'notRepairedDamage' to category that only have 0 or 1
dfFull=dfFull[(dfFull['notRepairedDamage'] == 0) | (dfFull['notRepairedDamage'] == 1)]

#%%
# Drop abnormal value in power, kilometer and price
dfFull=dfFull.drop(dfFull[dfFull['power']==0].index)
dfFull=dfFull.drop(dfFull[dfFull['kilometer']==0].index)
dfFull=dfFull.drop(dfFull[dfFull['price']==float].index)

#%%
# Remove outliers
cols = ['power', 'kilometer', 'price']
# Q1 = dfFull.quantile(0.25)
# Q3 = dfFull.quantile(0.75)
# IQR = Q3 - Q1
# dfFull = dfFull[~((dfFull[cols] < (Q1 - 1.5 * IQR)) |(dfFull[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
dfFull['model'] = dfFull['model'][dfFull['model'].between(dfFull['model'].quantile(.025), dfFull['model'].quantile(.975))]
dfFull['brand'] = dfFull['brand'][dfFull['brand'].between(dfFull['brand'].quantile(.025), dfFull['brand'].quantile(.975))]
dfFull['power'] = dfFull['power'][dfFull['power'].between(dfFull['power'].quantile(.025), dfFull['power'].quantile(.975))]
dfFull['kilometer'] = dfFull['kilometer'][dfFull['kilometer'].between(dfFull['kilometer'].quantile(.025), dfFull['kilometer'].quantile(.975))]
dfFull['price'] = dfFull['price'][dfFull['price'].between(dfFull['price'].quantile(.025), dfFull['price'].quantile(.975))]
dfFull=dfFull.dropna()
dfFull["model"] = dfFull["model"].astype('int')
dfFull["brand"] = dfFull["brand"].astype('int')
# %%
# Remove data with negative values
dfFull = dfFull[dfFull >= 0]
dfChkBasics(dfFull)

# %%
# Visualization
# QQ-plot
stats.probplot(dfFull['model'], dist="norm", plot=py)
py.show()
stats.probplot(dfFull['brand'], dist="norm", plot=py)
py.show()
stats.probplot(dfFull['bodyType'], dist="norm", plot=py)
py.show()
stats.probplot(dfFull['fuelType'], dist="norm", plot=py)
py.show()
stats.probplot(dfFull['gearbox'], dist="norm", plot=py)
py.show()
stats.probplot(dfFull['power'], dist="norm", plot=py)
py.show()
stats.probplot(dfFull['kilometer'], dist="norm", plot=py)
py.show()
stats.probplot(dfFull['price'], dist="norm", plot=py)
py.show()

#%%
# Histogram
# model histogram
plt.hist(dfFull['model'], density=True, bins=30)
plt.ylabel('Probability')
plt.xlabel('model')
plt.title("model Histogram")
#%%
# brand histogram
plt.hist(dfFull['brand'], density=True, bins=30)
plt.ylabel('Probability')
plt.xlabel('brand')
plt.title("brand Histogram")
#%%
# bodyType histogram
plt.hist(dfFull['bodyType'], density=True, bins=30)
plt.ylabel('Probability')
plt.xlabel('bodyType')
plt.title("bodyType Histogram")
#%%
# fuelType histogram
plt.hist(dfFull['fuelType'], density=True, bins=30)
plt.ylabel('Probability')
plt.xlabel('fuelType')
plt.title("fuelType Histogram")
#%%
# gearbox histogram
plt.hist(dfFull['gearbox'], density=True, bins=30)
plt.ylabel('Probability')
plt.xlabel('gearbox')
plt.title("gearbox Histogram")
#%%
# power histogram
plt.hist(dfFull['power'], density=True, bins=30)
plt.ylabel('Probability')
plt.xlabel('power')
plt.title("power Histogram")
#%%
# kilometer histogram
plt.hist(dfFull['kilometer'], density=True, bins=30)
plt.ylabel('Probability')
plt.xlabel('kilometer')
plt.title("kilometer Histogram")
#%%
# price histogram
plt.hist(dfFull['price'], density=True, bins=30)
plt.ylabel('Probability')
plt.xlabel('price')
plt.title("price Histogram")

# %%
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.formula.api import glm
import statsmodels.api as sm
# %%
# Setup train-test split
X_dfFull = dfFull[['brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', 'notRepairedDamage']]
print(X_dfFull.head())
y_dfFull = dfFull['price']
print(y_dfFull.head())
x_train, x_test, y_train, y_test = train_test_split(X_dfFull, y_dfFull, test_size = 0.25, random_state=2020)
# %%
# Prediction model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

clf1 = SVC()
clf2 = LogisticRegression()
clf3 = DecisionTreeClassifier()
clf4 = KNeighborsClassifier(n_neighbors=3)
clf5 = linear_model.LinearRegression() 
classifiers = [clf1,clf2,clf3,clf4,clf5]

# %%
for c in classifiers:
    c.fit(x_train,y_train)
    print('\n%s\n'%(c))
    print(f'train score:  {c.score(x_train,y_train)}')
    print(f'test score:  {c.score(x_test,y_test)}')

# %%
for c in classifiers:
  print('\n%s\n'%(c))
  print(cross_val_score(c, x_test, y_test, cv= 10))
  print(f'CV mean:  {np.mean(cross_val_score(c, x_test, y_test, cv= 10))}')

# %%
# Linear Model
lmfull = ols(formula = 'price ~ C(brand) + C(bodyType) + fuelType + gearbox + power + kilometer + C(notRepairedDamage)', data=dfFull).fit()
print( lmfull.summary() )
np.mean(lmfull.predict(dfFull))

# %%
# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
dfFulllm = dfFull.iloc[:, [2,3,4,5,6,7,8,9,10]]
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(dfFulllm.values, i) for i in range(dfFulllm.shape[1])]
vif["features"] = dfFulllm.columns
vif.round(1)

# %%
# Modified Linear Model
# The VIF for power is 14.1 > 10, so we want to exclude power from the linear model.
lmfull_adj = ols(formula = 'price ~ C(brand) + C(bodyType) + fuelType + gearbox + kilometer + C(notRepairedDamage)', data=dfFull).fit()
print( lmfull_adj.summary() )
np.mean(lmfull_adj.predict(dfFull))

# %%