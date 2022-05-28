#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install chart-studio')
import chart_studio.plotly as py
get_ipython().system(' pip install ipython')


# In[2]:


# imports
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 40)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
import plotly.express as px
from IPython import get_ipython
get_ipython().system('pip install chart-studio')
get_ipython().run_line_magic('matplotlib','inline')
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore');


# In[3]:


df = pd.read_csv('cars_engage_2022.csv')


# In[4]:


df.head()


# In[5]:


l_D = len(df)
c_m = len(df.Make.unique())
c_c = len(df.Model.unique())
n_f = len(df.columns)
fig = px.bar(x=['Observations',"Makers",'Models','Features'],y=[l_D,c_m,c_c,n_f], width=900,height=500)
fig.update_layout(
    title="Dataset Statistics",
    xaxis_title="",
    yaxis_title="Counts",
    font=dict(
        size=16,
    )
)

fig.show()


# In[6]:


df['car'] = df.Make + ' ' + df.Model
c = ['Make','Model','car','Variant','Body_Type','Fuel_Type','Fuel_System','Type','Drivetrain','Ex-Showroom_Price','Displacement','Cylinders',
     'ARAI_Certified_Mileage','Power','Torque','Fuel_Tank_Capacity','Height','Length','Width','Doors','Seating_Capacity','Wheelbase','Number_of_Airbags']
df_full = df.copy()
df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].str.replace('Rs. ','',regex=False)
df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].str.replace(',','',regex=False)
df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].astype(int)
df = df[c]
df = df[~df.ARAI_Certified_Mileage.isnull()]
df = df[~df.Make.isnull()]
df = df[~df.Width.isnull()]
df = df[~df.Cylinders.isnull()]
df = df[~df.Wheelbase.isnull()]
df = df[~df['Fuel_Tank_Capacity'].isnull()]
df = df[~df['Seating_Capacity'].isnull()]
df = df[~df['Torque'].isnull()]
df['Height'] = df['Height'].str.replace(' mm','',regex=False).astype(float)
df['Length'] = df['Length'].str.replace(' mm','',regex=False).astype(float)
df['Width'] = df['Width'].str.replace(' mm','',regex=False).astype(float)
df['Wheelbase'] = df['Wheelbase'].str.replace(' mm','',regex=False).astype(float)
df['Fuel_Tank_Capacity'] = df['Fuel_Tank_Capacity'].str.replace(' litres','',regex=False).astype(float)
df['Displacement'] = df['Displacement'].str.replace(' cc','',regex=False)
df.loc[df.ARAI_Certified_Mileage == '9.8-10.0 km/litre','ARAI_Certified_Mileage'] = '10'
df.loc[df.ARAI_Certified_Mileage == '10kmpl km/litre','ARAI_Certified_Mileage'] = '10'
df['ARAI_Certified_Mileage'] = df['ARAI_Certified_Mileage'].str.replace(' km/litre','',regex=False).astype(float)
df.Number_of_Airbags.fillna(0,inplace= True)
df['price'] = df['Ex-Showroom_Price'] * 0.014
df.drop(columns='Ex-Showroom_Price', inplace= True)
df.price = df.price.astype(int)
HP = df.Power.str.extract(r'(\d{1,4}).*').astype(int) * 0.98632
HP = HP.apply(lambda x: round(x,2))
TQ = df.Torque.str.extract(r'(\d{1,4}).*').astype(int)
TQ = TQ.apply(lambda x: round(x,2))
df.Torque = TQ
df.Power = HP
df.Doors = df.Doors.astype(int)
df.Seating_Capacity = df.Seating_Capacity.astype(int)
df.Number_of_Airbags = df.Number_of_Airbags.astype(int)
df.Displacement = df.Displacement.astype(int)
df.Cylinders = df.Cylinders.astype(int)
df.columns = ['make', 'model','car', 'variant', 'body_type', 'fuel_type', 'fuel_system','type', 'drivetrain', 'displacement', 'cylinders',
              'mileage', 'power', 'torque', 'fuel_tank','height', 'length', 'width', 'doors', 'seats', 'wheelbase','airbags', 'price']


# In[7]:


df.sample(6)


# In[8]:


df.columns = df.columns.str.strip()


# In[9]:


df[df.model =='Corolla Altis']


# In[10]:


fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,11))
sns.histplot(data=df, x='price',bins=50, alpha=.6, color='pink', ax=ax1)
ax12 = ax1.twinx()
sns.kdeplot(data=df, x='price', alpha=.2,fill= True,color="#254b7f",ax=ax12,linewidth=0)
ax12.grid()
ax1.set_title('Histogram of cars price data',fontsize=16)
ax1.set_xlabel('')
logbins = np.logspace(np.log10(3000),np.log10(744944.578),50)
sns.histplot(data=df, x='price',bins=logbins,alpha=.6, color='pink',ax=ax2)
ax2.set_title('Histogram of cars price data (log scale)',fontsize=16)
ax2.set_xscale('log')
ax22 = ax2.twinx()
ax22.grid()
sns.kdeplot(data=df, x='price', alpha=.2,fill= True,color="#254b7f",ax=ax22,log_scale=True,linewidth=0)
ax2.set_xlabel('Price (log)', fontsize=14)
ax22.set_xticks((800,1000,10000,100000,1000000))
ax2.xaxis.set_tick_params(labelsize=14);
ax1.xaxis.set_tick_params(labelsize=14);


# In[11]:


plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='price',width=.3,color='pink', hue= 'fuel_type')
plt.title('Box plot of Price',fontsize=18)
plt.xticks([i for i in range(0,1000000,100000)],[f'Rs.{i:,}' for i in range(0,1000000,100000)],fontsize=14)
plt.xlabel('price',fontsize=14);


# In[12]:


plt.figure(figsize=(20,7))
sns.countplot(data=df, y='body_type',alpha=.6,color='purple')
plt.title('Cars by car body type',fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('')
plt.ylabel('');


# In[13]:


#SUV's Sedans and hatchbacks seems to be the dominating car types
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='price', y='body_type', palette='viridis')
plt.title('Box plot of Price of every body type',fontsize=18)
plt.ylabel('')
plt.yticks(fontsize=14)
plt.xticks([i for i in range(0,1000000,100000)],[f'Rs.{i:,}' for i in range(0,1000000,100000)],fontsize=14);


# In[14]:


'''It's Clear that Car body type strongly affect the price
Now we check cars by Fuel type'''

plt.figure(figsize=(11,6))
sns.countplot(data=df, x='fuel_type',alpha=.6, color='purple')
plt.title('Cars count by engine fuel type',fontsize=18)
plt.xlabel('Fuel Type', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('');


# In[15]:


#Now we check car by engine size
plt.figure(figsize=(14,6))
sns.histplot(data=df, x='displacement',alpha=.6, color='purple',bins=10)
plt.title('Cars by engine size (in CC)',fontsize=18)
plt.xticks(fontsize=13);
plt.yticks(fontsize=13);


# In[16]:


#Now We check the Horsepower of cars
plt.figure(figsize=(14,6))
sns.histplot(data=df, x='power',alpha=.6, color='purple')
plt.title('Cars by engine size (in CC)',fontsize=18);
plt.xticks(fontsize=13);
plt.yticks(fontsize=13);


# In[17]:


#Now we check the relation horsepower and price considering diffreent body type
plt.figure(figsize=(10,8))
sns.scatterplot(data=df, x='power', y='price',hue='body_type',palette='viridis',alpha=.89, s=120 );
plt.xticks(fontsize=13);
plt.yticks(fontsize=13)
plt.xlabel('power',fontsize=14)
plt.ylabel('price',fontsize=14)
plt.title('Relation between power and price',fontsize=20);


# In[18]:


'''Horsepower of car seems to be highly related to car price but car body type seems a little bit blurry but hatchbacks 
seems to be the body type with the least horsepower and price'''


# In[19]:


#We can also look into the relation between Mileage and price
plt.figure(figsize=(10,8))
ax= fig.add_subplot()
sns.jointplot(data=df, x='mileage', y='price',kind= 'reg',ax=ax, palette='viridis',height=8,  ratio=7)
ax.text(.5,.7,'Relation between Power and price', fontsize=18)
ax.set_xlabel('Power (HP)', fontsize= 15);


# In[20]:


'''It's looks like expensive cars tend to have worse mileage'''


# In[21]:


#We can also check the overall correlation of between variables and each other
#For that first we make a pearson correlation grid
plt.figure(figsize=(22,8))
sns.heatmap(df.corr(), annot=True, fmt='.2%')
plt.title('Correlation between different variable',fontsize=20)
plt.xticks(fontsize=14, rotation=320)
plt.yticks(fontsize=14);


# In[22]:


#Now we check an extensive scatter plot grid of more numerical variable to investigate the realtion in more detail
sns.pairplot(df,vars=[ 'displacement', 'mileage', 'power', 'price'], hue= 'fuel_type',
             palette=sns.color_palette('magma',n_colors=4),diag_kind='kde',height=2, aspect=1.8);


# In[23]:


#Using more interactive plot to show the previous plot and also adding the car manufacturer
fig = px.scatter_3d(df, x='power', z='price', y='mileage',color='make',width=800,height=750)
fig.update_layout(showlegend=True)
fig.show();


# In[24]:


#Clustering the given data
df = df[df.price < 1000000]


# In[25]:


num_cols = [ i for i in df.columns if df[i].dtype != 'object']


# In[26]:


km = KMeans(n_clusters=8, n_init=20, max_iter=400, random_state=0)
clusters = km.fit_predict(df[num_cols])
df['cluster'] = clusters
df.cluster = (df.cluster + 1).astype('object')
df.sample(6)


# # price vs power

# In[27]:


plt.figure(figsize=(15,9))
sns.scatterplot(data=df, y='price', x='power',s=120,hue='cluster',palette='viridis')
plt.legend(ncol=3)
plt.title('Scatter plot of price and horsepower with adding clusters column', fontsize=18)
plt.xlabel('power',fontsize=16)
plt.ylabel('price',fontsize=16);


# # power vs mileage

# In[28]:


plt.figure(figsize=(15,9))
sns.scatterplot(data=df, x='power', y='mileage',s=120,hue='cluster',palette='viridis')
plt.legend(ncol=4)
plt.title('Scatter plot of milage and horsepower with clusters column', fontsize=18);
plt.xlabel('power',fontsize=16)
plt.ylabel('mileage',fontsize=16);


# # engine size vs fuel tank

# In[29]:


plt.figure(figsize=(15,9))
sns.scatterplot(data=df, x='fuel_tank', y='displacement',s=120,hue='cluster',palette='viridis')
plt.legend(ncol=4)
plt.title('Scatter plot of milage and horsepower with clusters column', fontsize=18);
plt.xlabel('Fuel Tank Capacity ',fontsize=16)
plt.ylabel('Engine size',fontsize=16);


# In[30]:


fig = px.scatter_3d(df, x='power', z='price', y='mileage',color='cluster',
                    height=700, width=800,color_discrete_sequence=sns.color_palette('colorblind',n_colors=8,desat=1).as_hex(),
                   title='price power, and mileage')
fig.show()


# In[31]:


plt.figure(figsize=(10,7))
sns.barplot(data=df, x= 'cluster', ci= 'sd', y= 'price', palette='viridis',order=df.groupby('cluster')['price'].mean().sort_values(ascending=False).index);
plt.yticks([i for i in range(0,650000,100000)])
plt.title('Average price of each cluster',fontsize=20)
plt.xlabel('Cluster',fontsize=16)
plt.ylabel('Avg car price', fontsize=16)
plt.xticks(fontsize=14);


# In[32]:


plt.figure(figsize=(14,6))
sns.countplot(data=df, x= 'cluster', palette='viridis',order=df.cluster.value_counts().index);
# plt.yticks([i for i in range(0,650000,100000)])
plt.title('Number of cars in each cluster',fontsize=18)
plt.xlabel('Cluster',fontsize=16)
plt.ylabel('Number of cars', fontsize=16)
plt.xticks(fontsize=14);


# In[33]:


df[df.model == 'Corolla Altis']


# In[34]:


df_c = df[df.cluster.isin([1,8])]
p_dic = {'Mahindra':'#46327e', 'Tata':'#46327e', 'Toyota':'orange',
        'Jeep':'#46327e', 'Honda':'#46327e', 'Kia':'#46327e',
        'Hyundai':'#46327e','Skoda':'#46327e'}
c_dic = {'Mahindra Scorpio':'#481769', 'Mahindra Xuv500':'#481769', 'Tata Hexa':'#481769',
       'Toyota Innova Crysta':'#481769', 'Jeep Compass':'#481769', 'Toyota Corolla Altis':'orange',
       'Honda Civic':'#481769', 'Kia Seltos':'#481769', 'Tata Safari Storme':'#481769',
       'Hyundai Elantra':'#481769', 'Hyundai Tucson':'#481769', 'Hyundai Creta':'#481769',
       'Tata Harrier':'#481769', 'Skoda Octavia':'#481769'}


# In[54]:


df_c.sample(100)


# In[45]:


px.box(data_frame=df_c,x='car',y='price',height=500,  width=800,color='car', color_discrete_sequence=list(c_dic.values()))
# fig.update_layout(show)


# In[46]:


plt.figure(figsize=(20,6))
sns.countplot(data=df_c,x='body_type',palette='viridis')
plt.xlabel('Body type',fontsize=16)
plt.ylabel('Count of variants',fontsize=16)
plt.title('count of each body type in the targeted clusters (including variants)',fontsize=14);


# # seems like there are more number of hatchbacks in the toyota cluster

# In[47]:


get_ipython().system('pip install scikit-learn')


# In[48]:


from sklearn.linear_model import LinearRegression


# In[49]:


df_c['price'] = df_c['price'].astype(float)
df_c['cluster'] = df_c['cluster'].astype(float)


# In[50]:


df.dtypes


# In[51]:


#Increased mileage how the price increses 
model = LinearRegression()
model.fit(df_c[['mileage']],df_c[['price']])
print(model.predict([[10]]))



