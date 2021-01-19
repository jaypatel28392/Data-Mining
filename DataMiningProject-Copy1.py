#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df = pd.read_csv("/Users/shilp/Downloads/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head()


# In[5]:


df.shape


# In[3]:



df.info()


# In[4]:


df.isnull().sum()


# In[6]:


for column in df.columns:
    if df[column].isnull().sum() != 0:
        print("=======================================================")
        print(f"{column} ==> Missing Values : {df[column].isnull().sum()}, dtypes : {df[column].dtypes}")


# In[7]:


df['name'].unique()


# In[8]:


df['name'].fillna('', inplace=True)
df['name'].unique()


# In[9]:


df.isnull().sum()


# In[10]:


df['host_name'].unique()


# In[11]:


df['host_name'].fillna('', inplace=True)
df['host_name'].unique()


# In[12]:


df.isnull().sum()


# In[13]:


df['last_review'].unique()


# In[14]:


df['last_review'].fillna('0', inplace=True)


# In[16]:


df['reviews_per_month'].fillna((df['reviews_per_month'].mean()), inplace=True)
df['reviews_per_month'].unique()


# In[17]:


for column in df.columns:
    if df[column].isnull().sum() != 0:
        df[column] = df[column].fillna(df[column].mode()[0])


# In[18]:


df.isnull().sum()


# In[19]:


pd.options.display.float_format = "{:.2f}".format
df.describe()


# In[13]:


categorical_col = []
for column in df.columns:
    if len(df[column].unique()) <= 10:
        print("===============================================================================")
        print(f"{column} : {df[column].unique()}")
        categorical_col.append(column)


# In[14]:


df.last_review.isnull().sum()


# In[20]:



len(df.index)


# In[21]:


df.shape


# In[22]:


timeit df.shape


# In[23]:


df.loc[0]


# In[24]:


df.info(verbose=True)


# In[25]:


df.room_type.value_counts()


# In[26]:


#Number of Room Types Available
room_type_plot = sns.countplot(x="room_type", order = df.room_type.value_counts().index, data=df)
room_type_plot.set(xlabel='Room Types', ylabel='', title='Room Type Count Bar')
for bar in room_type_plot.patches:
    h = bar.get_height()
    room_type_plot.text(
        bar.get_x() + bar.get_width()/2.,  # bar index (x coordinate of text)
        h,                                 # y coordinate of text
        '%d' % int(h),                     # y label
        ha='center', 
        va='bottom',
        color='black',
        fontweight='bold',
        size=14)
    
plt.show()


# In[27]:


#Percentage Representation of Neighbourhood Group in Pie
df.neighbourhood_group.value_counts(dropna = False, normalize = True)


# In[28]:


labels = df.neighbourhood_group.value_counts().index
sizes = df.neighbourhood_group.value_counts().values
explode = (0.1, 0.2, 0.3, 0.4, 0.6)

fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                   shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set(title="Most Rented Neighbourhood Group Pie Plot")
ax.legend(wedges, labels,
          title="Neighbourhood Groups",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=8, weight="bold")
plt.show()


# In[29]:


#Density and Distribution of Prices for each Neighbourhood Group
sns.set(style="whitegrid")

FILTER_PRICE_VALUE = 800

sub_df_price = df[df.price < FILTER_PRICE_VALUE]

fig, ax = plt.subplots(figsize=(12, 12))
density_neigh_price_plot = sns.violinplot(ax=ax, x="neighbourhood_group", y="price", 
                                          hue="neighbourhood_group", data=sub_df_price, 
                                          palette="muted", dodge=False)
density_neigh_price_plot.set(xlabel='Neighberhood Group', ylabel='Price ($)', 
                             title='Density and Distribution of prices for each Neighberhood Group')
ylabels = ['${}'.format(x) for x in density_neigh_price_plot.get_yticks()]
density_neigh_price_plot.set_yticklabels(ylabels)
plt.show()


# In[30]:


df.hist(edgecolor="Black", linewidth=1.5, figsize=(25, 25));


# In[32]:


figure, subplots = plt.subplots(
                len(df.neighbourhood_group.unique()), 
                figsize=(15, 15)
            )
for i, neighbourhood_group in enumerate(df.neighbourhood_group.unique()):
    neighbourhoods = df[df.neighbourhood_group == neighbourhood_group]['price']
    ax = subplots[i]
    dist_plot = sns.distplot(neighbourhoods, ax=ax)
    dist_plot.set_title(neighbourhood_group)
plt.tight_layout(h_pad=1)
plt.show()


# In[33]:


#Number of reviews grouped by Host Id
serie_df = df.groupby("host_id")["number_of_reviews"].agg("sum")
frame = { 'host_id': serie_df.index, 'number_of_reviews': serie_df.values }
df_df = pd.DataFrame(frame).sort_values('number_of_reviews', ascending=False).head(50)

f, ax = plt.subplots(figsize=(12, 12))
sns.barplot(x="number_of_reviews", y="host_id", 
            data=df_df, color="b", ax=ax, orient="h")

plt.show()


# In[34]:


#top 10 host
df1 = df.host_id.value_counts()[:10]
f,ax = plt.subplots(figsize=(10,10))
ax = sns.barplot(x = df1.index,y=df1.values,palette="rocket")
plt.show()


# In[31]:


corr = df.corr()
plt.figure(figsize=(12,12))
ax = sns.heatmap(df.corr(),annot=True, cmap = 'viridis')


# In[35]:


fig = px.scatter(df, x="availability_365", y="host_id",
	         size="number_of_reviews", color="neighbourhood_group",
                 hover_name="host_name", title="Number of reviews/Availability 365 per Host ID/ Host Name")

fig.update_layout(legend_orientation="h")
fig.update_layout(
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Availability 365",
            font=dict(
                family="Courier New, monospace",
                size=13,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Host Id",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)

fig.show()


# In[36]:


fig, ax = plt.subplots(figsize=(12,12))

img=plt.imread("/Users/shilp/Downloads/new-york-city-airbnb-open-data/New_York_City_.png", 0)
coordenates_to_extent = [-74.258, -73.7, 40.49, 40.92]
ax.imshow(img, zorder=0, extent=coordenates_to_extent)

scatter_map = sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group',s=20, ax=ax, data=df)
ax.grid(True)
plt.legend(title='Neighbourhood Groups')
plt.show()


# In[37]:


FILTER_PRICE_VALUE = 600
sub_df = df[df.price < FILTER_PRICE_VALUE]

fig, ax = plt.subplots(figsize=(10, 10))

cmap = plt.get_cmap('hot')
c = sub_df.price           
alpha = 0.5                
label = "airbnb"
price_heatmap = ax.scatter(sub_df.longitude, sub_df.latitude, label=label, c=c, cmap=cmap, alpha=0.4)
plt.title("Heatmap by Price $")
plt.colorbar(price_heatmap)
plt.grid(True)
plt.show()


# In[38]:


plt.figure(figsize=(25, 25))
sns.pairplot(df, height=5, diag_kind="hist")


# In[ ]:


print("Density plots")
df.plot(kind='Density', subplots=True, layout=(16,16), sharex=False, figsize=(150,150))
plt.show()


# In[41]:


categorical_col = []
for column in df.columns:
    if len(df[column].unique()) <= 10:
        print("===============================================================================")
        print(f"{column} : {df[column].unique()}")
        categorical_col.append(column)
categorical_col


# In[42]:


dataset = pd.get_dummies(df, columns=categorical_col)
dataset.head()


# In[43]:


print(df.columns)
print(dataset.columns)


# In[44]:


print(dataset.describe().loc["mean", :])
print("====================================")
print(dataset.describe().loc["std", :])


# In[45]:


from sklearn.preprocessing import StandardScaler

col_to_scale = ['host_id', 'latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews',
                'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
s_sc = StandardScaler()
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])


# In[46]:


print(dataset.describe().loc["mean", :])
print("====================================")
print(dataset.describe().loc["std", :])


# In[47]:


fig = sns.FacetGrid(df, hue="room_type", aspect=4, height=10)
fig.map(sns.kdeplot, 'host_id', shade=True)
oldest = df['host_id'].max()
fig.set(xlim=(0, oldest))
sns.set(font_scale=5)
fig.add_legend()


# In[48]:


sns.set(font_scale=1.5)
plt.figure(figsize=(12, 8))
df.host_id.hist(bins=100)


# In[49]:


df.neighbourhood.hist(bins=100)


# In[50]:


plt.style.use("fivethirtyeight")

data = df.neighbourhood.value_counts()[:10]
plt.figure(figsize=(12, 8))
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("Most Popular Neighbourhood")
plt.ylabel("Neighbourhood Area")
plt.xlabel("Number of guest Who host in this Area")

plt.barh(x, y)


# In[51]:



airbnb=pd.read_csv('/Users/shilp/Downloads/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

airbnb.drop(['name','id','host_name','last_review'],axis=1,inplace=True)
airbnb['reviews_per_month']=airbnb['reviews_per_month'].replace(np.nan, 0)


# In[52]:



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


# In[53]:



list(le.inverse_transform(airbnb['room_type']))[:10]


# In[54]:



lm = LinearRegression()

X = airbnb[['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]
y = airbnb['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

lm.fit(X_train,y_train)


# In[55]:



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


# #kfold
# 
# Input: from sklearn.model_selection import cross_val_score
#        scores = cross_val_score(lm, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
#        lm_rmse_scores = np.sqrt(-scores)
#        print("""
#         Scores: {}
#         Mean: {}
#         Standard deviation: {}
#         """.format(
#         lm_rmse_scores,
#         lm_rmse_scores.mean(),
#         lm_rmse_scores.std())
#         )
# 
# Output:
#         Scores: [237.22555275 282.19382235 186.82124459 234.60652944 247.47577519]
#         Mean: 237.66458486316475
#         Standard deviation: 30.57352498208825

# In[61]:


error_airbnb = pd.DataFrame({
        'Actual Values': np.array(y_test).flatten(),
        'Predicted Values': predicts.flatten()}).head(20)

error_airbnb.head(5)


# In[63]:


title=['Pred vs Actual']
fig = go.Figure(data=[
    go.Bar(name='Predicted', x=error_airbnb.index, y=error_airbnb['Predicted Values']),
    go.Bar(name='Actual', x=error_airbnb.index, y=error_airbnb['Actual Values'])
])

fig.update_layout(barmode='group')
fig.show()


# In[64]:


plt.figure(figsize=(16,8))
sns.regplot(predicts,y_test)
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.title("Linear Model Predictions")
plt.grid(False)
plt.show()


# In[65]:


#cluster


# In[66]:


df.neighbourhood.unique()


# In[67]:


df_onehot = pd.get_dummies(df[['price']], prefix = "", prefix_sep = "")
df_onehot['neighbourhood'] = df['neighbourhood']
fixed_columns = [df_onehot.columns[-1]] + list(df_onehot.columns[:-1])
df_grouped = df_onehot.groupby('neighbourhood').mean().reset_index()
df_grouped.head(20)


# In[68]:


# Import libraries
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


# In[69]:


df_clustering = df_grouped.drop('neighbourhood',1)
df_clustering.head()


# In[70]:


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


# In[71]:


num_clusters = 3

kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(df_clustering)
kmeans.labels_


# In[72]:


df_grouped.insert(0, 'Cluster', kmeans.labels_)


# In[73]:


df_grouped.head()


# In[74]:


df_co = df.copy()
df_co.drop_duplicates(subset = ['neighbourhood'])
drop_element = 'price'
df_co.drop(drop_element, axis = 1)
df_co.head()


# In[75]:


df_merge = pd.merge(df_co, df_grouped[['Cluster','neighbourhood','price']],on = 'neighbourhood')
df_merge.head()


# In[76]:


df_merge.head()


# In[80]:


cluster_1 = df_merge.loc[df_merge['Cluster'] == 1]
cluster_1.head(3)
cluster_0 = df_merge.loc[df_merge['Cluster'] == 0]
cluster_0.head(3)
cluster_2 = df_merge.loc[df_merge['Cluster'] == 2]
cluster_0.head(3)


# In[81]:


cluster_1.shape


# In[82]:


print('Cluster_0 mean price: ',cluster_0.price_x.mean())
print('Reviews in Cluster0:',len(cluster_0.number_of_reviews))
print('Cluster_1 mean price: ',cluster_1.price_x.mean())
print('Reviews in Cluster1:',len(cluster_1.number_of_reviews))
print('Cluster_2 mean price: ',cluster_2.price_x.mean())
print('Reviews in Cluster2:',len(cluster_2.number_of_reviews))


# In[83]:


import matplotlib.pyplot as plt
plt.figure(figsize = (8,8))
sns.countplot(x = 'neighbourhood_group', hue = 'Cluster', data = df_merge)


# In[84]:


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


# In[ ]:





# In[94]:


plt.figure(figsize = (16,12))
sns.countplot(x = 'room_type', hue = 'Cluster', data = df_merge)


# # For algorithm and kfold, I made a new file becuse this worked when I made them, as of now when i ran the codes in this file I got errors, In other file everything worked effeciently.
