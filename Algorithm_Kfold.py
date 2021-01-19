#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''Importing Data Manipulation Modules'''
import numpy as np                 # Linear Algebra
import pandas as pd                # Data Processing, CSV file I/O (e.g. pd.read_csv)

'''Seaborn and Matplotlib Visualization'''
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import seaborn as sns              # Python Data Visualization Library based on matplotlib
import geopandas as gpd            # Python Geospatial Data Library
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

'''Plotly Visualizations'''
import plotly as plotly                # Interactive Graphing Library for Python
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)


'''Machine Learning'''
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor


# In[5]:


airbnb= pd.read_csv('/Users/shilp/Downloads/new-york-city-airbnb-open-data_final/AB_NYC_2019.csv')

airbnb.drop(['name','id','host_name','last_review'],axis=1,inplace=True)
airbnb['reviews_per_month']=airbnb['reviews_per_month'].replace(np.nan, 0)


# In[6]:


'''Encode labels with value between 0 and n_classes-1.'''
le = preprocessing.LabelEncoder()                                            # Fit label encoder
le.fit(airbnb['neighbourhood_group'])
airbnb['neighbourhood_group']=le.transform(airbnb['neighbourhood_group'])    # Transform labels to normalized encoding.

le = preprocessing.LabelEncoder()
le.fit(airbnb['neighbourhood'])
airbnb['neighbourhood']=le.transform(airbnb['neighbourhood'])

le = preprocessing.LabelEncoder()
le.fit(airbnb['room_type'])
airbnb['room_type']=le.transform(airbnb['room_type'])

airbnb.sort_values(by='price',ascending=True,inplace=True)

airbnb.head()


# In[7]:


'''Reversing Labeling Transform'''
list(le.inverse_transform(airbnb['room_type']))[:10]


# In[8]:


'''Train LRM'''
lm = LinearRegression()

X = airbnb[['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]
y = airbnb['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

lm.fit(X_train,y_train)


# In[9]:


'''Get Predictions & Print Metrics'''
predicts = lm.predict(X_test)

print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
        np.sqrt(metrics.mean_squared_error(y_test, predicts)),
        r2_score(y_test,predicts) * 100,
        mean_absolute_error(y_test,predicts)
        ))


# In[13]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(lm, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
lm_rmse_scores = np.sqrt(-scores)
print("""
        Scores: {}
        Mean: {}
        Standard deviation: {}
        """.format(
        lm_rmse_scores,
        lm_rmse_scores.mean(),
        lm_rmse_scores.std())
     )


# In[10]:


'''Gradient Boosted Regressor'''
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01)
GBoost.fit(X_train,y_train)


# In[14]:


'''Get Predictions & Metrics'''
predicts2 = GBoost.predict(X_test)

print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
        np.sqrt(metrics.mean_squared_error(y_test, predicts2)),
        r2_score(y_test,predicts2) * 100,
        mean_absolute_error(y_test,predicts2)
        ))


# In[16]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(GBoost, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
GBoost_rmse_scores = np.sqrt(-scores)
print("""
        Scores: {}
        Mean: {}
        Standard deviation: {}
        """.format(
        GBoost_rmse_scores,
        GBoost_rmse_scores.mean(),
        GBoost_rmse_scores.std())
     )


# In[17]:


error_airbnb = pd.DataFrame({
        'Actual Values': np.array(y_test).flatten(),
        'Predicted Values': predicts.flatten()}).head(20)

error_airbnb.head(5)


# In[18]:


title=['Pred vs Actual']
fig = go.Figure(data=[
    go.Bar(name='Predicted', x=error_airbnb.index, y=error_airbnb['Predicted Values']),
    go.Bar(name='Actual', x=error_airbnb.index, y=error_airbnb['Actual Values'])
])

fig.update_layout(barmode='group')
fig.show()


# In[19]:


plt.figure(figsize=(16,8))
sns.regplot(predicts,y_test)
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.title("Linear Model Predictions")
plt.grid(False)
plt.show()


# In[20]:


error_airbnb = pd.DataFrame({
        'Actual Values': np.array(y_test).flatten(),
        'Predicted Values': predicts2.flatten()}).head(20)

error_airbnb.head(5)


# In[21]:


title=['Pred vs Actual']
fig = go.Figure(data=[
    go.Bar(name='Predicted', x=error_airbnb.index, y=error_airbnb['Predicted Values']),
    go.Bar(name='Actual', x=error_airbnb.index, y=error_airbnb['Actual Values'])
])

fig.update_layout(barmode='group')
fig.show()


# In[22]:


plt.figure(figsize=(16,8))
sns.regplot(predicts2,y_test)
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.title("Gradient Boosted Regressor model Predictions")
plt.show()


# In[30]:


df= pd.read_csv('/Users/shilp/Downloads/new-york-city-airbnb-open-data_final/AB_NYC_2019.csv')


# In[31]:


df.neighbourhood.unique()


# In[32]:


drop_elements = ['last_review', 'host_name','id']
df.drop(drop_elements, axis = 1, inplace= True)
df.fillna({'reviews_per_month':0}, inplace=True)
df.reviews_per_month.isnull().sum()


# In[33]:


df_onehot = pd.get_dummies(df[['price']], prefix = "", prefix_sep = "")
df_onehot['neighbourhood'] = df['neighbourhood']
fixed_columns = [df_onehot.columns[-1]] + list(df_onehot.columns[:-1])
df_grouped = df_onehot.groupby('neighbourhood').mean().reset_index()
df_grouped.head(20)


# In[34]:


# Import libraries
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


# In[35]:


df_clustering = df_grouped.drop('neighbourhood',1)
df_clustering.head()


# In[36]:


K = range(1,10)
distortions = []
for k in K:
    kmeans = KMeans(init = 'k-means++', n_clusters = k, n_init = 12, random_state = 0)
    kmeans.fit(df_clustering.values.reshape(-1,1))
    distortions.append(sum(np.min(cdist(df_clustering.values.reshape(-1, 1),kmeans.cluster_centers_, 'euclidean'), axis = 1)) / df_clustering.shape [0])

import matplotlib.pyplot as plt
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal K')
plt.show()  


# In[37]:


num_clusters = 3

kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(df_clustering)
kmeans.labels_


# In[38]:


df_grouped.insert(0, 'Cluster', kmeans.labels_)


# In[39]:


df_grouped.head()


# In[40]:


df_co = df.copy()
df_co.drop_duplicates(subset = ['neighbourhood'])
drop_element = 'price'
df_co.drop(drop_element, axis = 1)
df_co.head()


# In[41]:


df_merge = pd.merge(df_co, df_grouped[['Cluster','neighbourhood','price']],on = 'neighbourhood')
df_merge.head()


# In[42]:


cluster_1 = df_merge.loc[df_merge['Cluster'] == 1]

cluster_0 = df_merge.loc[df_merge['Cluster'] == 0]

cluster_2 = df_merge.loc[df_merge['Cluster'] == 2]


# In[43]:


print('Cluster_0 mean price: ',cluster_0.price_x.mean())
print('Reviews in Cluster0:',len(cluster_0.number_of_reviews))
print('Cluster_1 mean price: ',cluster_1.price_x.mean())
print('Reviews in Cluster1:',len(cluster_1.number_of_reviews))
print('Cluster_2 mean price: ',cluster_2.price_x.mean())
print('Reviews in Cluster2:',len(cluster_2.number_of_reviews))


# In[44]:


import matplotlib.pyplot as plt
plt.figure(figsize = (8,8))
sns.countplot(x = 'neighbourhood_group', hue = 'Cluster', data = df_merge)


# In[45]:


df_clu=pd.DataFrame(np.arange(3).reshape((1,3)),index=['0'],columns=['cluster_0','cluster_1','cluster_2'])
df_clu.cluster_0 = len(cluster_0)
df_clu.cluster_1 = len(cluster_1)
df_clu.cluster_2 = len(cluster_2)

plt.figure(figsize = (5,5))
sns.barplot(data = df_clu)
plt.xlabel('Cluster Number',fontsize =12)
plt.ylabel('No. of Shared Room', fontsize = 12)
plt.title('Shared Room data',fontsize = 12 )
plt.show()


# In[46]:


plt.figure(figsize = (16,12))
sns.countplot(x = 'room_type', hue = 'Cluster', data = df_merge)


# In[48]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scores = cross_val_score(kmeans, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
kmeans_rmse_scores = np.sqrt(-scores)
print("""
        Scores: {}
        Mean: {}
        Standard deviation: {}
        """.format(
        kmeans_rmse_scores,
        kmeans_rmse_scores.mean(),
        kmeans_rmse_scores.std())
     )


# In[ ]:




