#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import libraries
import pandas as pd ##data processing, dataframe (read,drop,replace..etc)
import numpy as np  ##linear algebra
import matplotlib.pyplot as plt  ##..Data visualization
##statement
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns  ##Statistical Data Visualization

#import warning
import  warnings
warnings.filterwarnings('ignore')  ##ignore unwanted warning


# In[2]:


## import the dataset
hc_data=pd.read_csv('health_care.csv')


# In[3]:


# Read the data
hc_data


# In[4]:


hc_data1=hc_data.copy()


# In[5]:


#check shape of dataset
hc_data.shape


# In[6]:


#check top 5 rows by using head
hc_data.head()


# In[7]:


hc_data.tail()


# In[8]:


## CHeck null values
hc_data.isnull().sum().any()


# In[9]:


hc_data.isnull().sum()


# In[10]:


#check info
hc_data.info()


# In[11]:


#check all description from dataset
hc_data.describe()


# In[12]:


# Histogram of the Heart Dataset

fig = plt.figure(figsize = (40,30))
hc_data.hist(ax = fig.gca());


# From the above histogram plots, we can see that the features are skewed and not normally distributed. Also, the scales are different between one and another.

# # Understanding the Data

# we can see the correlation between different features in below heat mat 

# In[13]:


# Creating a correlation heatmap
sns.heatmap(hc_data.corr(),annot=True, cmap='terrain', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(20,20)
plt.show()


# From the above HeatMap, we can see that cp and thalach are the features with highest positive correlation whereas exang, oldpeak and ca are negatively correlated.While other features do not hold much correlation with the response variable "target"

# # Outlier Detection

# Since the dataset is not large, we cannot discard the outliers. We will treat the outliers as potential observations.

# In[14]:


# Boxplots
fig_dims = (15,8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.boxplot(data=hc_data, ax=ax);


# # Handling Imbalance
# 
# Imbalance in a dataset leads to inaccuracy and high precision, recall scores. There are certain resampling techniques such as undersampling and oversampling to handle these issues.
# 
# Considering our dataset, the response variable target has two outcomes "Patients with Heart Disease" and "Patients without Heart Disease". Let us now observe their distribution in the dataset.

# In[15]:


hc_data["target"].value_counts()


# From the above chart, we can conclude even when the distribution is not exactly 50:50, but still the data is good enough to use on machine learning algorithms and to predict standard metrics like Accuracy and AUC scores. So, we do not need to resample this dataset.

# # Train-Test Split

# Divide the data into training and test datasets using the train_test_split() function

# In[16]:


#Extracting Variable  
X = hc_data.drop("target",axis=1)
y = hc_data["target"]


# In[17]:


X.shape


# In[18]:


y.shape


# In[19]:


# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,stratify=y,random_state=7)


# # Logistic Regression

# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# In[22]:


pred = classifier.predict(X_test)


# In[23]:


pred


# # Test Accuracy of the result

# In[24]:


#Creating the Confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[25]:


# Accuracy on Test data
accuracy_score(y_test, pred)


# In[26]:


# Accuracy on Train data
accuracy_score(y_train, classifier.predict(X_train))


# In[27]:


# Building a predictive system
import warnings
in_data = (57,0,0,140,241,0,1,123,1,0.2,1,0,3)

# Changing the input data into a numpy array
in_data_as_numpy_array = np.array(in_data)

# Reshaping the numpy array as we predict it
in_data_reshape = in_data_as_numpy_array.reshape(1,-1)
pred = classifier.predict(in_data_reshape)
print(pred)

if(pred[0] == 0):
    print('The person does not have heart disease.')
else:
    print('The person has heart disease.')


# In[ ]:


**End**..


# In[ ]:


#Thank you!!...


# In[ ]:




