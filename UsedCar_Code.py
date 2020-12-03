
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
# Brand, Body Type, Fuel Type and Gearbox Type
# on the value of the vehicle.


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
# Kilometer issue
#
# We think this variable's stratification might come from a mutiple choice question.
# And the last choice might be the kilometer that larger than 15km, so we have so many "15km" at here.
# In order to continue our analysis, we choose to drop most of cars' "kilometer" equal to 15.
dfQ2=dfQ2.drop(((dfQ2['kilometer']==15).sample(frac=0.7,random_state=1)).index)
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

lr = linear_model.LinearRegression()

#%%
#print('Logit model accuracy (with the test set):', lr.score(x_test, y_test))
#print('Logit model accuracy (with the train set):', lr.score(x_train, y_train))
#%%
#
# CV Split

#%%
#
# Classification
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
classifiers = [clf1,clf2,clf3] 
classifiers.append(clf4)
classifiers.append(clf5)
classifiers.append(clf6)
#%%
for c in classifiers:
    c.fit(x_train,y_train)
    print(f'train score:  {c.score(x_train,y_train)}')
    print(f'test score:  {c.score(x_test,y_test)}')
    print(confusion_matrix(y_test, c.predict(x_test)))
    print(classification_report(y_test, c.predict(x_test)))

#%%
for c in classifiers:
  print('\n%s\n'%(c))
  print(cross_val_score(c, x_test, y_test, cv= 10, scoring='accuracy'))
#%%
#
# ROC-AUC

#%%
dfChkBasics(dfQ3, valCnt= True)
# %%
# Q3 Market Behavior(Canceled)
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
