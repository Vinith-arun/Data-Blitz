#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import missingno
import seaborn as sns


# In[3]:


train_df = pd.read_csv(r"C:\Users\Tringapps\Downloads\housing.csv")
test_df = pd.read_csv(r"C:\Users\Tringapps\Downloads\housing_test.csv")
model = LinearRegression()


# In[4]:


train_df=train_df.fillna(train_df.quantile(0.5))
test_df=test_df.fillna(test_df.quantile(0.5))


# In[5]:


# df1 = df[(df['median_house_value']>df.median_house_value.quantile(0.04))&(df['median_house_value']<df.median_house_value.quantile(0.96))]
filtered_df=train_df.drop(['households','total_rooms','ocean_proximity'],axis='columns')
print("Filtered Training Data")
filtered_df.head()


# In[6]:


print("Test Data")
test_df.head()


# In[7]:


list(train_df.columns)


# In[8]:


train_df.describe()


# In[9]:


#Plot graphic of missing
import missingno
missingno.matrix(train_df, figsize = (30,5))


# In[10]:


#Plot graphic of missing
import missingno
missingno.matrix(test_df, figsize = (30,5))


# In[11]:


#finding number of missing values 
train_df.isnull().sum()


# In[12]:


#finding number of missing values 
test_df.isnull().sum()


# In[13]:


dummies = pd.get_dummies(train_df.ocean_proximity)
filtered_df1=[]
filtered_df1 = pd.concat([filtered_df,dummies], axis='columns')
# Filtered the 'NEAR OCEAN','<1H OCEAN','NEAR BAY' by grouping them into one
x = filtered_df1.drop(['median_house_value','NEAR OCEAN','<1H OCEAN','NEAR BAY'],axis='columns')
y = train_df.median_house_value
print("Seperating the data for the Plot")
x.head()


# In[14]:


sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()


# In[17]:



g = sns.pairplot(data=train_df, hue='median_house_value', palette = 'seismic',
                 height=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
     


# In[18]:


model.fit(x,y)


# In[19]:


model.score(x,y)


# In[20]:


model.predict([[-122.23,37.88,41.0,129.0,322.0,8.3252,1,0]])


# In[50]:


plt.plot(x,y)
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
n_estim=range(100,1000,100)

## Search grid for optimal parameters
param_grid = {"n_estimators" :n_estim}


model_rf = GridSearchCV(model2,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)

model_rf.fit(x,y)



# Best score
print(model_rf.best_score_)

#best estimator
model_rf.best_estimator_


# In[ ]:




