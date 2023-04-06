#!/usr/bin/env python
# coding: utf-8

# # Medical Cost Personal Insurance

# Importing the dependencies

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


# **Data collection and analysis**

# In[4]:



medical_insurance=pd.read_csv(r'C:\Users\Lewnovo\medical_cost_insurance.csv')
medical_insurance


# In[5]:


# top 5  rows of the datasets
medical_insurance.head()


# In[6]:


medical_insurance.shape


# In[7]:


#getting the information from the datasets
medical_insurance.info()


# In[8]:


# checking for the missing values
medical_insurance.isnull().sum()


# **Data Analysis and Visulaization**

# **statistical measures of the dataset**

# In[9]:


medical_insurance.describe()


# In[10]:


#distribution  of age value
sns.set()
plt.figure(figsize=(5,5))
sns.distplot(medical_insurance['age'])# distplot tells us the distribution of the dataset throghtout
plt.title('age distribution')
plt.show()


# In[11]:


# we cannot use categoreical  on the characters
plt.figure(figsize=(8,4))
sns.countplot(x='sex',data=medical_insurance)
plt.title('gender distribution')
plt.show()


# In[12]:


# value count is helped to see how many values are their in the dataset
medical_insurance['sex'].value_counts()


# In[13]:


#bmi distribution( body mass index)
plt.figure(figsize=(5,5))
sns.distplot(medical_insurance['bmi'])# distplot tells us the distribution of the dataset throghtout
plt.title('bmi distribution')
plt.show()


# Normal BMI Range    18.5  to 24.9 

# In[14]:


# we can use countplot for children column 
plt.figure(figsize=(3,4))
sns.countplot(x='children',data=medical_insurance)
plt.title('children distribution')
plt.show()


# In[15]:


# we checking the  number of childrens with the help of the value counts for the accurate numbers
medical_insurance['children'].value_counts()


# In[16]:


sns.set()
plt.figure(figsize=(3,5))
sns.countplot(medical_insurance['smoker'])
plt.title('smoker distribution')
plt.show()


# In[17]:


medical_insurance['smoker'].value_counts()


# In[18]:


# we can use countplot for children column 
plt.figure(figsize=(3,4))
sns.countplot(x='region',data=medical_insurance)
plt.title('region distribution')
plt.show()


# In[19]:


medical_insurance['region'].value_counts()


# In[20]:


#bmi distribution( body mass index)
plt.figure(figsize=(6,6))
sns.distplot(medical_insurance['charges'])# distplot tells us the distribution of the dataset throghtout
plt.title('charges distribution')
plt.show()


# # Data pre processing

# Encoding the categorical features

# In[21]:


#Encoding sex column
medical_insurance.replace({'sex':{'male':0,'female':1}},inplace=True)

#Encoding smoker column
medical_insurance.replace({'smoker':{'yes':0,'no':1}},inplace=True)

#Encoding region column
medical_insurance.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)


# In[20]:


medical_insurance


# Splitting the features ans the target

# In[22]:


x=medical_insurance.drop(columns='charges',axis=1)# we are dropping the charges column and swaving in x
y=medical_insurance['charges']# we are saving the charges column in y


# In[22]:


x


# In[23]:


y


# Spliting the data into the train and test

# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[25]:


print(x.shape,x_train.shape,x_test.shape)
print(y.shape,y_train.shape, y_test.shape)


# Training the model
Linear Regression
# In[26]:


# loading the linesar regression model
reg=LinearRegression()
reg


# In[27]:


reg.fit(x_train,y_train)


# # model evaluation

# In[28]:


# prediction on training the dta 


# In[29]:


train_pred=reg.predict(x_train)


# In[38]:


# R Squared value ( it lies in the range of 0 to 1 )
r2_train= metrics.r2_score(y_train,train_pred)
print('R  Squared value :',r2_train)


# In[39]:


test_pred=reg.predict(x_test)
r2_test=metrics.r2_score(y_test,test_pred)
print('R  Squared value :',r2_test)


# # Building the preductive system

# In[41]:


input_data=(31,1,25.74,0,1,0)
# changing the input data into the numpy array
num_array=np.asarray(input_data)

#reshaping the array
reshaped_num_array=num_array.reshape(1,-1)

prediction=reg.predict(reshaped_num_array)
print(prediction)
print('The insurance cost is USD ',prediction[0])


# Conclusion:
# The insurance cost is USD  3760.0805764960496
#     

# In[ ]:




