# %%
import pandas as pd
#%%
df=pd.read_csv("used_car_train_20200313.csv")
# %%
df.rename(columns=[df.columns[0].split()])
# %%

# %%
