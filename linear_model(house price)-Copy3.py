#!/usr/bin/env python
# coding: utf-8

# In[18]:


# find the house price of 33,000 and 5000 area


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[20]:


df = pd.read_csv ( 'D:\houseprice.csv')
df


# In[22]:


# Plot the scatter plot to get the general idea
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.xlabel('Area')
#plt.ylabel('Price')
plt.scatter(df.Area,df.Price,color='red',marker='+')
#plt.plot(df.Area,reg.predict[['Area']]),color='blue')


# In[5]:


reg = linear_model.LinearRegression()
reg.fit(df[['Area']], df.Price)


# In[6]:


reg.predict([[3300]]) #predicted the price of this area


# In[7]:


reg.predict([[5000]]) #predicted the price of this area


# In[17]:


# Plot the scatter plot to get the general idea
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.Area,df.Price,color='red',marker='+')
plt.plot(df.Area,reg.predict(df[['Area']]),color='blue')


# In[8]:


# now we got a new list to pridinct the price. we wre importing the file


# In[9]:


d = pd.read_csv('D:/areaofhome1.csv') # back slash was throwing the error so i put right slash
d


# In[10]:


p=reg.predict(d) # we have preicted the price of new area
p


# In[15]:


d ['Price']= p # we have added a column Price to d pata frame
d


# In[16]:


d.to_csv('D:\prediction',index=False) # created the new csv file in excel work book


# In[13]:


d


# In[ ]:




