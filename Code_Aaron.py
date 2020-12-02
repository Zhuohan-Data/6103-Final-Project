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

# %% scatter plot
#relationship between regDate and price
plt.scatter(dfQ1.regDate, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("regDate")


#relationship between model and price
plt.scatter(dfQ1.model, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("model")


#relationship between brand and price
plt.scatter(dfQ1.brand, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("brand")


#relationship between bodytype and price
dfQ1 = dfQ1[dfQ1['bodyType']<=7]
plt.scatter(dfQ1.bodyType, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("bodyType")


#relationship between fueltype and price
dfQ1 = dfQ1[dfQ1['fuelType']<=6]
plt.scatter(dfQ1.fuelType, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("fuelType")


#relationship between Gearbox and price
#change geartype into category with only 0 and 1
dfQ1=dfQ1[(dfQ1['gearbox'] == 0) | (dfQ1['gearbox'] == 1)]
dfQ1["gearbox"] = dfQ1["gearbox"].astype('category')
plt.scatter(dfQ1.gearbox, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("gearbox")

# %% scatter plot
#relationship between regDate and price
plt.scatter(dfQ1.regDate, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("regDate")


#relationship between model and price
plt.scatter(dfQ1.model, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("model")


#relationship between brand and price
plt.scatter(dfQ1.brand, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("brand")


#relationship between bodytype and price
dfQ1 = dfQ1[dfQ1['bodyType']<=7]
plt.scatter(dfQ1.bodyType, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("bodyType")


#relationship between fueltype and price
dfQ1 = dfQ1[dfQ1['fuelType']<=6]
plt.scatter(dfQ1.fuelType, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("fuelType")


#relationship between Gearbox and price
#change geartype into category with only 0 and 1
dfQ1=dfQ1[(dfQ1['gearbox'] == 0) | (dfQ1['gearbox'] == 1)]
dfQ1["gearbox"] = dfQ1["gearbox"].astype('category')
plt.scatter(dfQ1.gearbox, dfQ1.price)
plt.ylabel("price")
plt.grid(b=True, which='major', axis='y') 
plt.title("gearbox")
# %% boxplot
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


# %% hist
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
# %% qqplot
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
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA
pip install lightgbm
pip install xgboost
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
# %%
#Training
Train_data = pd.read_csv('used_car_train_20200313.csv')
TestB_data = pd.read_csv('used_car_testB_20200421.csv')
Train_data =Train_data.drop(columns='name')
train = Train_data.iloc[:,[0,1,2,3,4,5,6,7,14]]
train.set_index('SaleID')
train.replace('-',np.nan,inplace=True)
train=train.dropna()
dfChkBasics(train, valCnt= True)
test = TestB_data.iloc[:,[0,1,2,3,4,5,6,7,14]]
numerical_cols = train.select_dtypes(exclude = 'object').columns
print(numerical_cols)

feature_cols = [col for col in numerical_cols if col not in ['SaleID','regDate','price','model','brand','seller']]
feature_cols = [col for col in feature_cols if 'Type' not in col]

X_data = train[feature_cols]
Y_data = train['price']

X_test  = test[feature_cols]

print('X train shape:',X_data.shape)
print('X test shape:',X_test.shape)

def Sta_inf(data):
    print('_min',np.min(data))
    print('_max:',np.max(data))
    print('_mean',np.mean(data))
    print('_ptp',np.ptp(data))
    print('_std',np.std(data))
    print('_var',np.var(data))
    
plt.hist(Y_data)
plt.title("Distribution of Price")
plt.show()
plt.close() 


xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=7) 

scores_train = []
scores = []

sk=StratifiedKFold(n_splits=4,shuffle=False,random_state=0)
for train_ind,val_ind in sk.split(X_data,Y_data):
    
    train_x=X_data.iloc[train_ind].values
    train_y=Y_data.iloc[train_ind]
    val_x=X_data.iloc[val_ind].values
    val_y=Y_data.iloc[val_ind]
    
    xgr.fit(train_x,train_y)
    pred_train_xgb=xgr.predict(train_x)
    pred_xgb=xgr.predict(val_x)
    
    score_train = mean_absolute_error(train_y,pred_train_xgb)
    scores_train.append(score_train)
    score = mean_absolute_error(val_y,pred_xgb)
    scores.append(score)

def build_model_xgb(x_train,y_train):
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=7) #, objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model

def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127,n_estimators = 150)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm


x_train,x_val,y_train,y_val = train_test_split(X_data,Y_data,test_size=0.3)
print('Train lgb...')
model_lgb = build_model_lgb(x_train,y_train)
val_lgb = model_lgb.predict(x_val)
MAE_lgb = mean_absolute_error(y_val,val_lgb)
print('MAE of val with lgb:',MAE_lgb)

print('Predict lgb...')
model_lgb_pre = build_model_lgb(X_data,Y_data)
subA_lgb = model_lgb_pre.predict(X_test)
print('Sta of Predict lgb:')
Sta_inf(subA_lgb)


print('Train xgb...')
model_xgb = build_model_xgb(x_train,y_train)
val_xgb = model_xgb.predict(x_val)
MAE_xgb = mean_absolute_error(y_val,val_xgb)
print('MAE of val with xgb:',MAE_xgb)

print('Predict xgb...')
model_xgb_pre = build_model_xgb(X_data,Y_data)
subA_xgb = model_xgb_pre.predict(X_test)
print('Sta of Predict xgb:')
Sta_inf(subA_xgb)


val_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*val_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*val_xgb
val_Weighted[val_Weighted<0]=10
print('MAE of val with Weighted ensemble:',mean_absolute_error(y_val,val_Weighted))