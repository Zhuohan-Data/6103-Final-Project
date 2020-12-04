
#%%
#os.chdir(r'F:\Github\6103-Final-Project')
# %%
import pandas as pd
import numpy as np
import pylab as py
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
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
# Read Data
#
# Drop meaningless variables
df=df.iloc[:,:16]
df=df.drop(columns='name')
# %%
# Check Data
dfChkBasics(df, valCnt= True)



# %%
# Q1 Car Attribute
# Research the influence of a vehicle’s 
# Brand, Model, Body Type, Fuel Type and Gearbox Type
# on the value of the vehicle.
dfQ1 = df.iloc[:,[0,1,2,3,4,5,6,14]]
dfQ1.set_index('SaleID')
dfQ1.replace('-',np.nan,inplace=True)
dfQ1=dfQ1.dropna()

#Change datatype
dfQ1["regDate"] = dfQ1["regDate"].astype('float')
dfQ1["model"] = dfQ1["model"].astype('float')
dfQ1["brand"] = dfQ1["brand"].astype('float')
dfQ1["bodyType"] = dfQ1["bodyType"].astype('int')
dfQ1["fuelType"] = dfQ1["fuelType"].astype('int')
dfQ1['gearbox'] = dfQ1['gearbox'].astype('float')

dfQ1 = dfQ1[(dfQ1['gearbox'] == 0) | (dfQ1['gearbox'] == 1)]
dfQ1['gearbox'] = dfQ1['gearbox'].astype('int')
dfQ1 = dfQ1[dfQ1['fuelType']<=6]
dfQ1 = dfQ1[dfQ1['bodyType']<=7]
dfQ1=dfQ1.drop(dfQ1[dfQ1['price']==float].index)


#Delete outliers
dfQ1['model'] = dfQ1['model'][dfQ1['model'].between(dfQ1['model'].quantile(.025), dfQ1['model'].quantile(.975))]
dfQ1['brand'] = dfQ1['brand'][dfQ1['brand'].between(dfQ1['brand'].quantile(.025), dfQ1['brand'].quantile(.975))]
dfQ1['price'] = dfQ1['price'][dfQ1['price'].between(dfQ1['price'].quantile(.025), dfQ1['price'].quantile(.975))]
dfQ1=dfQ1.dropna()


# %%
#QQPlot    
#relationship between model and price
sm.qqplot(dfQ1['model']) 
plt.title("model")
py.show() 
#relationship between brand and price
sm.qqplot(dfQ1['brand']) 
plt.title("brand")
py.show()
#relationship between bodyType and price
sm.qqplot(dfQ1['bodyType']) 
plt.title("bodyType")
py.show()
#relationship between fuelType and price
sm.qqplot(dfQ1['fuelType']) 
plt.title("fuelType")
py.show()
#relationship between gearbox and price
sm.qqplot(dfQ1['gearbox']) 
plt.title("gearbox")
py.show()

# %%
#hist
#relationship between model and price
grouped=dfQ1.groupby('model')
g1=grouped['price'].mean().reset_index('model')
sns.barplot(x=dfQ1['model'],y=dfQ1['price'])
plt.show()

#relationship between brand and price
grouped=dfQ1.groupby('brand')
g1=grouped['price'].mean().reset_index('brand')
sns.barplot(x=dfQ1['brand'],y=dfQ1['price'])
plt.show()

#relationship between bodyType and price
grouped=dfQ1.groupby('bodyType')
g1=grouped['price'].mean().reset_index('bodyType')
sns.barplot(x=dfQ1['bodyType'],y=dfQ1['price'])
plt.xlim(-0.5,7.5)
plt.show()

#relationship between fuelType and price
grouped=dfQ1.groupby('fuelType')
g1=grouped['price'].mean().reset_index('fuelType')
sns.barplot(x=dfQ1['fuelType'],y=dfQ1['price'])
plt.xlim(-0.5,6.5)
plt.show()

#relationship between gearbox and price
grouped=dfQ1.groupby('gearbox')
g1=grouped['price'].mean().reset_index('gearbox')
sns.barplot(x=dfQ1['gearbox'],y=dfQ1['price'])
plt.xlim(-0.5,3)
plt.show()

# %% 
#boxplot
#relationship between model and price
dfQ1['model'].plot(kind='box')
plt.show()

#relationship between brand and price
dfQ1['brand'].plot(kind='box')
plt.show()

#relationship between bodyType and price
dfQ1['bodyType'].plot(kind='box')
plt.show()

#relationship between fuel and price
dfQ1['fuelType'].plot(kind='box')
plt.show()

# %% scatter plot
#relationship between model and price
plt.scatter(dfQ1.model, dfQ1.price, alpha = 0.1)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("model")


#relationship between brand and price
plt.scatter(dfQ1.brand, dfQ1.price, alpha = 0.1)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("brand")


#relationship between bodytype and price
plt.scatter(dfQ1.bodyType, dfQ1.price, alpha = 0.1)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("bodyType")


#relationship between fueltype and price
plt.scatter(dfQ1.fuelType, dfQ1.price, alpha = 0.1)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("fuelType")


# 3-dimention of model with gearbox
sns.lmplot('model', 'price', data=dfQ1, hue="gearbox", x_jitter=3.5, scatter_kws={'alpha': 0.3, 's': 40 } )
plt.xlabel("model")
plt.ylabel("Price")
plt.title("Price vs model")
sns.despine()

# 3-dimention of brand with gearbox
sns.lmplot('brand', 'price', data=dfQ1, hue="gearbox", x_jitter=3.5, scatter_kws={'alpha': 0.3, 's': 40 } )
plt.xlabel("brand")
plt.ylabel("Price")
plt.title("Price vs brand")
sns.despine()

# 3-dimention of bodytype with gearbox
sns.lmplot('bodyType', 'price', data=dfQ1, hue="gearbox", x_jitter=3.5, scatter_kws={'alpha': 0.3, 's': 40 } )
plt.xlabel("bodyType")
plt.ylabel("Price")
plt.title("Price vs bodyType")
sns.despine()

# 3-dimention of fueltype with gearbox
sns.lmplot('fuelType', 'price', data=dfQ1, hue="gearbox", x_jitter=3.5, scatter_kws={'alpha': 0.3, 's': 40 } )
plt.xlabel("fuelType")
plt.ylabel("Price")
plt.title("Price vs fuelType")
sns.despine()



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
from sklearn.tree import DecisionTreeClassifier
# %%
#Model

x1 = dfQ1[['model', 'brand', 'bodyType', 'fuelType', 'gearbox']]
y1 = dfQ1['price']
y1 = y1.astype('int')
print(x1.head())
print(type(x1))
print(y1.head())
print(type(y1))
x1.dtypes
y1.dtypes


x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size = 0.20, random_state=2000)
full_split1 = linear_model.LinearRegression()
full_split1.fit(x_train1, y_train1)
full_split1.fit(x_train1, y_train1)
y_pred1 = full_split1.predict(x_test1)
full_split1.score(x_test1, y_test1)




print('score (train):', full_split1.score(x_train1, y_train1)) 
print('score (test):', full_split1.score(x_test1, y_test1))
print('intercept:', full_split1.intercept_)
print('coef_:', full_split1.coef_)





#%%
#
# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


#%%
clf1 = SVC()
clf2 = LogisticRegression()
clf3 = DecisionTreeClassifier()
clf4 = KNeighborsClassifier(n_neighbors=3)
clf5 = linear_model.LinearRegression() 
classifiers = [clf1,clf2,clf3,clf4,clf5]
# %%
for c in classifiers:
    c.fit(x_train1,y_train1)
    print('\n%s\n'%(c))
    print(f'train score:  {c.score(x_train1,y_train1)}')
    print(f'test score:  {c.score(x_test1,y_test1)}')

#%%
for c in classifiers:
  print('\n%s\n'%(c))
  print(cross_val_score(c, x_test1, y_test1, cv= 10))
  print(f'CV mean:  {np.mean(cross_val_score(c, x_test1, y_test1, cv= 10))}')

#%%
#Choose Linear model
from statsmodels.formula.api import ols

lm = ols(formula='price ~ model + brand + bodyType + fuelType + C(gearbox)', data=dfQ1).fit()
print( lm.summary() )
np.mean(lm.predict(dfQ1))


#%%
#VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
dfQ1=dfQ1.iloc[:,[2,3,4,5,6]]
vif = pd.DataFrame()
vif["variables"] = dfQ1.columns
vif["VIF"] = [ variance_inflation_factor(dfQ1.values, i) for i in range(dfQ1.shape[1]) ]
print(vif)
#So lm is the final model



# %%
# Q2 Car Damage
# Research the influence of a vehicle’s 
# Milage, Engine power, whether the car has been Damaged
# on the value of the vehicle.
# %%
# Q2.1 EDA

# Q2.1.1 EDA

# Select col and drop NA
dfQ2 = df.iloc[:,[0,7,8,9,14]]
dfQ2.set_index('SaleID')
dfQ2.replace('-',np.nan,inplace=True)
dfQ2=dfQ2.dropna()

dfChkBasics(dfQ2, valCnt= True)
#%%
# Change the data type
dfQ2["power"] = dfQ2["power"].astype('float')
dfQ2["kilometer"] = dfQ2["kilometer"].astype('float')
dfQ2["notRepairedDamage"] = dfQ2["notRepairedDamage"].astype('int')
#%%
# Change 'notRepairedDamage' to category that only have 0 or 1.
dfQ2=dfQ2[(dfQ2['notRepairedDamage'] == 0) | (dfQ2['notRepairedDamage'] == 1)]
#%%
# Drop abnormal value in power, kilometer and price.
dfQ2=dfQ2.drop(dfQ2[dfQ2['power']==0].index)
dfQ2=dfQ2.drop(dfQ2[dfQ2['kilometer']==0].index)
dfQ2=dfQ2.drop(dfQ2[dfQ2['price']==float].index)

# %%
# Drop outliers
dfQ2['power'] = dfQ2['power'][dfQ2['power'].between(dfQ2['power'].quantile(.025), dfQ2['power'].quantile(.975))]
dfQ2['kilometer'] = dfQ2['kilometer'][dfQ2['kilometer'].between(dfQ2['kilometer'].quantile(.025), dfQ2['kilometer'].quantile(.975))]
dfQ2['price'] = dfQ2['price'][dfQ2['price'].between(dfQ2['price'].quantile(.025), dfQ2['price'].quantile(.975))]
dfQ2=dfQ2.dropna()

#%%
plt.hist(dfQ2['kilometer'], bins='auto')
plt.xlabel("Kilometer")
plt.ylabel("Frequency")
plt.title("Kilometer Histogram")
plt.show()
#%%
# Kilometer issue
#
# We think this variable's stratification might come from a mutiple choice question.
# And the last choice might be the kilometer that larger than 15km, so we have so many "15km" at here.
# In order to continue our analysis, we choose to drop most of cars' "kilometer" equal to 15.
dfQ2=dfQ2.drop((dfQ2[dfQ2['kilometer']==15]).sample(frac=0.7,random_state=1).index)

dfChkBasics(dfQ2)
#%%
# Q2.1.2 Normality Plot
# QQPlot    
stats.probplot(dfQ2['power'], dist="norm", plot=py)
py.show()
stats.probplot(dfQ2['kilometer'], dist="norm", plot=py)
py.show()

#%%
# Histogram
plt.hist(dfQ2['power'], bins='auto')
plt.xlabel("Power")
plt.ylabel("Frequency")
plt.title("Power Histogram")
plt.show()

plt.hist(dfQ2['kilometer'], bins='auto')
plt.xlabel("Kilometer")
plt.ylabel("Frequency")
plt.title("Kilometer Histogram")
plt.show()
#%%
# Boxplot
dfQ2['power'].plot(kind='box')
plt.show()
dfQ2['kilometer'].plot(kind='box')
plt.show()
#%%
sns.regplot('kilometer', 'power', data=dfQ2, y_jitter=0.75, x_jitter=3.5, scatter_kws={'alpha': 0.3, 's': 40 } )
#%%
# Q2.2 Visualization
# Power vs Damage
# Before, we assume the "power" is the power of after using.
fuzzykilo = dfQ2['kilometer'] + np.random.normal(0,0.75, size=len(dfQ2['kilometer']))
fuzzypower = dfQ2['power'] + np.random.normal(0,3.5, size=len(dfQ2['notRepairedDamage']))
plt.plot(fuzzykilo,fuzzypower, 'o', markersize=8 ,alpha = 0.1)
plt.xlabel("Kilometer")
plt.ylabel("Power")
plt.title("Kilometer vs Power")
plt.show()
#%%
# Price vs Damage
fuzzykilo = dfQ2['notRepairedDamage'] + np.random.normal(0,0.4, size=len(dfQ2['notRepairedDamage']))
plt.plot(fuzzykilo,dfQ2['price'], 'o', markersize=8 ,alpha = 0.1)
plt.xlabel("Damge")
plt.ylabel("Price")
plt.title("Price vs Damage")
plt.show()

#%%
# Price vs Power
plt.plot(fuzzypower,dfQ2['price'], 'o', markersize=8 ,alpha = 0.1)
plt.xlabel("Power")
plt.ylabel("Price")
plt.title("Price vs Power")
plt.show()

#%%
# 3-dimention
sns.lmplot('power', 'price', data=dfQ2, hue="notRepairedDamage", x_jitter=3.5, scatter_kws={'alpha': 0.3, 's': 40 } )
plt.xlabel("Power")
plt.ylabel("Price")
plt.title("Price vs Power")
sns.despine()

#%%
# Q2.3 Model 
#
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.formula.api import glm
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import linear_model 
#%%
# Train-Test Split
xdfQ2 = dfQ2[['power', 'kilometer','notRepairedDamage']]
print(xdfQ2.head())
ydfQ2 = dfQ2['price']
print(ydfQ2.head())

x_train, x_test, y_train, y_test = train_test_split(xdfQ2, ydfQ2, test_size = 0.25, random_state=2020)

#%%
#
# Prediction Model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

clf1 = SVC()
clf2 = SVC(kernel="linear")
clf3 = LinearSVC()
clf4 = LogisticRegression()
clf5 = DecisionTreeClassifier()
clf6 = KNeighborsClassifier(n_neighbors=3)
clf7 = linear_model.LinearRegression() 
classifiers = [clf1,clf2,clf3,clf4,clf5,clf6,clf7]
#classifiers = [clf7]  

# %%
for c in classifiers:
    c.fit(x_train,y_train)
    print('\n%s\n'%(c))
    print(f'train score:  {c.score(x_train,y_train)}')
    print(f'test score:  {c.score(x_test,y_test)}')

#%%
for c in classifiers:
  print('\n%s\n'%(c))
  print(cross_val_score(c, x_test, y_test, cv= 10))
  print(f'CV mean:  {np.mean(cross_val_score(c, x_test, y_test, cv= 10))}')
#%%
#
# ROC-AUC (Canceled) 
#
# Can't draw on numeric response.
#%%
# Model Summary
from statsmodels.formula.api import ols

lm = ols(formula='price ~ power + kilometer + C(notRepairedDamage)', data=dfQ2).fit()
print( lm.summary() )
np.mean(lm.predict(dfQ2))

#%%
# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
dfQ2=dfQ2.iloc[:,[1,2,3,4]]
vif = pd.DataFrame()
vif["variables"] = dfQ2.columns
vif["VIF"] = [ variance_inflation_factor(dfQ2.values, i) for i in range(dfQ2.shape[1]) ]
print(vif)
#%%
# Final Model
lm1 = ols(formula='price ~ + kilometer + C(notRepairedDamage)', data=dfQ2).fit()
print( lm1.summary() )
np.mean(lm1.predict(dfQ2))


# %%
# Q3 Market Behavior (Canceled)
# Research the influence of 
# Seller and Offer type 
# on the value of the vehicle.
dfQ3 = df.iloc[:,[0,11,12]]
dfQ3.set_index('SaleID')
dfQ3.replace('-',np.nan,inplace=True)
dfQ3=dfQ3.dropna()

dfQ3=dfQ3[(dfQ3['seller'] == 0) | (dfQ3['seller'] == 1)]
dfQ3=dfQ3[(dfQ3['offerType'] == 0) | (dfQ3['offerType'] == 1)]
# Since we find all sellers and offerType are 0, we canceled this question.
dfChkBasics(dfQ3, valCnt= True)
#%%

# %%
# Conclusion

# %%
