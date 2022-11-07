#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[1]:


import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model,model_selection
from sklearn.model_selection import train_test_split


# # import dataset

# In[2]:


dataset = pd.read_csv('adm_data.csv')


# # Look dataset

# In[3]:


dataset


# # In ra các column của dataset

# In[4]:


dataset.columns


# # Kiểm tra thông tin dataset có cột nào null không, có 

# In[5]:


dataset.info()


# # Kiểm tra cột nào có giá trị null

# In[6]:


dataset.isnull().sum()
dataset['Research'].value_counts()


# # Kiểm tra mối liên hệ giữa các biến với nhau

# In[7]:


dataset.corr()


# # Trực quan hóa mối liên hệ giữa các biến với nhau

# In[8]:


plt.figure(figsize=(10,7))
sns.heatmap(dataset.corr(),annot=True,cmap=plt.cm.Blues)
plt.show()


# # Trực quan hóa từng column của dataset dưới dạng đồ thị cột

# In[9]:


dataset.hist(bins=50,figsize=(15,20))
plt.show()


# # Tạo feature và label cho linear regression model

# In[10]:


feature = dataset[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP','LOR ', 'CGPA', 'Research']]
label = dataset['Chance of Admit ']


# # Tạo data train và test model

# In[11]:


dataX_train,dataX_test,dataY_train,dataY_test = train_test_split(feature,label,test_size=1/3, random_state=0)


# # Tạo linear regression model và train model

# In[12]:


linear_regression = linear_model.LinearRegression()
linear_regression.fit(dataX_train,dataY_train)


# # Tìm ma trận weight và giá trị b trong linear regression model

# In[13]:


b = linear_regression.intercept_
w = linear_regression.coef_
w,b


# # Test model

# In[14]:


linear_regression.predict([[0,0,0,0,0,0,0]])


# In[15]:


b + np.dot(w,[337,118,4,4.5,4.5,9.65,1])


# In[16]:


linear_regression.score(feature,label)


# # Thêm cột Mô hình dự đoán và cột loss

# In[17]:


dataset['Chance of Admit Predict'] = linear_regression.predict(feature)
dataset['loss'] = abs(dataset['Chance of Admit '] - dataset['Chance of Admit Predict'])
dataset


# In[18]:


feature = dataset[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP','LOR ', 'CGPA', 'Research']]
label = dataset['Chance of Admit ']
dataX_train,dataX_test,dataY_train,dataY_test = train_test_split(feature,label,test_size=1/3, random_state=0)
linear_regression.fit(dataX_train,dataY_train)
linear_regression.score(feature,label)


# In[19]:





# In[ ]:




