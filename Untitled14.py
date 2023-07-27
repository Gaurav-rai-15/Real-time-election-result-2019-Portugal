#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LogisticRegression,Lasso,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics
from scipy.stats import zscore
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet


# In[2]:


df=pd.read_csv("ElectionData.csv")
df.head()


# In[3]:


df.info()


# In[4]:


sns.heatmap(df.isnull())
plt.show()


# In[5]:


df.dtypes


# In[6]:


df['Date']=pd.to_datetime(df['time']).dt.date
df['time'] = pd.to_datetime(df['time']).dt.time
df.head()


# In[7]:


cols=["territoryName","Party"]
for i in cols:
    print("Number of unique values in ", i ," are : ",len(df[i].unique()), " : " ,df[i].unique())


# In[8]:


cols = ["Party","territoryName"]
fig,axes=plt.subplots(nrows=2,ncols=1,figsize=[16,12])
for i in range(0,len(cols)):
    axes[i]=sns.countplot(x = cols[i],data = df,ax=axes[i])
    axes[i].set_title("Count plot of "+cols[i])


# In[9]:


plt.figure(figsize=(18,10))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)


# In[10]:


cols=["Hondt","Votes","Mandates","totalVoters"]

fig, axes = plt.subplots(nrows=1,ncols=4,figsize=[20,6])
for i in range(0,len(cols)):
    axes[i]=sns.scatterplot( x= cols[i], y="FinalMandates", data = df,hue="FinalMandates", size = "FinalMandates",    sizes=(50,200), hue_norm=(0, 6),cmap="accent",ax=axes[i])
    axes[i].set_title("FinalMandates vs "+cols[i])


# In[11]:


cols=["territoryName","Party"]

fig, axes = plt.subplots(nrows=2,ncols=1,figsize=[20,12])
for i in range(0,len(cols)):
    axes[i]=sns.scatterplot( x= cols[i], y="FinalMandates", data = df,hue=cols[i], ax=axes[i])
    axes[i].set_title("FinalMandates vs "+cols[i])
    axes[i].set_ylim(0,30)


# In[12]:


cols_grp1=['totalMandates','numParishesApproved','blankVotes','nullVotes','subscribedVoters','totalVoters','pre.blankVotes','pre.nullVotes','pre.subscribedVoters','pre.totalVoters','Percentage','Mandates','pre.blankVotesPercentage','pre.votersPercentage']
cols_grp2=["Votes","Hondt"]
df1 = df[cols_grp1]
df2=df[cols_grp2]
sc = StandardScaler()
a = sc.fit_transform(df1)
df_1 = pd.DataFrame(a,columns=df1.columns)
b = sc.fit_transform(df2)
df_2 = pd.DataFrame(b,columns=df2.columns)


# In[13]:


pca = PCA(n_components=1)
new_var1 = pca.fit_transform(df_1)
new_var2=pca.fit_transform(df_2)

df_new = pd.concat((df, pd.DataFrame(new_var1)), axis=1)
df_new.rename({0: 'PCA_1'}, axis=1, inplace = True)
df_new.drop(cols_grp1, axis=1, inplace=True)
df_new = df_new = pd.concat((df_new, pd.DataFrame(new_var2)), axis=1)
df_new.rename({0: 'PCA_2'}, axis=1, inplace = True)
df_new.drop(cols_grp2, axis=1, inplace=True)
df_new.head()


# In[14]:


df=df_new


# In[15]:


obj_col =[]
for i in df.columns:
    if df[i].dtypes=="O":
        obj_col.append(i)
obj_col


# In[16]:


le = LabelEncoder()
for i in obj_col:
    df[i]=pd.DataFrame(le.fit_transform(df[i]))
df.head()


# In[17]:


df.describe()


# In[18]:


df.plot(kind="box",subplots=True,layout=(5,5),figsize=(15,15))


# In[19]:


z = np.abs(zscore(df))


# In[20]:


threshold = 3
print(np.where(z<3))
print(df.shape)


# In[21]:


filtered_entries= (z < 3).all(axis=1)
df_new = df[filtered_entries]


# In[22]:


print(df.shape)
print(df_new.shape)
df_new.tail()


# In[23]:


df_new = df[(z<3).all(axis = 1)]
df=df_new


# In[24]:


df.plot(kind="box",subplots=True,layout=(5,5),figsize=(15,15))


# In[25]:


df.hist(figsize=(15,15), layout=(4,4), bins=20)


# In[26]:


df.skew()


# In[27]:


cols =["availableMandates","pre.nullVotesPercentage","validVotesPercentage"]

for col in cols:
    df[col]=np.sqrt(df[col])
            
df.skew()


# In[28]:


df.hist(figsize=(15,15), layout=(4,4), bins=20)


# In[29]:


x = df.drop(columns=['FinalMandates'])
y = df[["FinalMandates"]]


# In[30]:


sc = StandardScaler()
a = sc.fit_transform(x)
df_x = pd.DataFrame(a,columns=x.columns)

df_x.head()


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df_x,y,test_size=0.20,random_state=45)


# In[32]:


model=[LinearRegression(),SVR(),DecisionTreeRegressor(),KNeighborsRegressor(),Lasso(),Ridge(),ElasticNet(),RandomForestRegressor(),AdaBoostRegressor()]

for m in model:
    m.fit(x_train,y_train)
    score=m.score(x_train,y_train)
    predm=m.predict(x_test)
    print('Score of',m,'is:',score)
    print('MAE:',mean_absolute_error(y_test,predm))
    print('MSE:',mean_squared_error(y_test,predm))
    print('RMSE:',np.sqrt(mean_squared_error(y_test,predm)))
    print('R2 score:',r2_score(y_test,predm))
    print('*'*100)
    print('\n') 


# In[33]:


dtr_rg = DecisionTreeRegressor()
parameters={'criterion':['mse'],'max_depth': np.arange(3, 15)}
clf = GridSearchCV(dtr_rg, parameters, cv=5)
clf.fit(df_x,y)
clf.best_params_


# In[34]:


from sklearn.ensemble import RandomForestRegressor
dtr_rg = DecisionTreeRegressor(criterion="mse",max_depth=6)
dtr_rg.fit(x_train,y_train)
print('Score:',dtr_rg.score(x_train,y_train))
y_pred_dtr=dtr_rg.predict(x_test)
print('Mean absolute error:',mean_absolute_error(y_test,y_pred_dtr))
print('Mean squared error:',mean_squared_error(y_test,y_pred_dtr))
print('Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test,y_pred_dtr)))
print('R2 score:',r2_score(y_test,y_pred_dtr))


# In[ ]:


from sklearn.model_selection import GridSearchCV

parameters={'n_estimators':[200,400,500,600] ,'max_depth':[4,5,10,15,20]}
rfr=RandomForestRegressor()

clf=GridSearchCV(rfr,parameters)
clf.fit(x,y)
print(clf.best_params_)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=600,max_features='auto',max_depth=5)
rf.fit(x_train,y_train)
print('Score:',rf.score(x_train,y_train))
y_pred_rf=rf.predict(x_test)
print('Mean absolute error:',mean_absolute_error(y_test,y_pred_rf))
print('Mean squared error:',mean_squared_error(y_test,y_pred_rf))
print('Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test,y_pred_rf)))
print('R2 score:',r2_score(y_test,y_pred_rf))


# In[ ]:


plt.scatter(x=y_test,y=y_pred_rf,marker= "o",color="red",alpha=0.2)
plt.plot(x,x,"g--",alpha=0.1)
plt.xlim(0,20)
plt.ylim(0,20)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

