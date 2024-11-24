#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings 
warnings.filterwarnings("ignore")


# In[92]:


data = pd.read_csv(r"C:\Users\Hazel\Downloads\DiabetesDatasets\diabetes.csv")
# Filter to keep only rows with all values intact (no NaNs)
data_complete = data.dropna()

# Display the shape of the filtered dataset
print("Shape of dataset with all values intact:", data_complete.shape)
data_complete


# In[93]:


sns.heatmap(data_complete.isnull())


# Co-relation matrix - done to find out to which columns the values of the outcome is dependent
# - Inorder to find which columns are relevant and which are not in predicting the outcome of diabetes

# In[94]:


correlation = data_complete.corr()
print(correlation)


# To make the correlation visiual, heat map is created
# Least correlation BloodPressure and Family History
# 

# In[95]:


plt.figure(figsize=(8, 8))
sns.heatmap(data_complete.corr(), vmin=-1.0, center=0, cmap='RdBu_r', annot=True)
plt.show()


# **Standardizing the data set**

# In[96]:


sc_x = StandardScaler()
X = pd.DataFrame(sc_x.fit_transform(data_complete.drop(["Outcome"], axis=1)),
                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'WaistToHipRatio', 'FamilyHistory', 'Age'])


# In[97]:


X.head()


# **Train Test Split**

# Train test split
# Variable X will contain all the independent variables and Y will contain the outcome

# In[98]:


X = data_complete.drop("Outcome", axis=1)
Y = data_complete['Outcome']
X


# In[99]:


X = data_complete.drop("Outcome", axis=1)
Y = data_complete['Outcome']
Y


# Now we are going to split the data

# In[100]:


X = data_complete.drop("Outcome", axis=1)
Y = data_complete['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) #20% test data
X_train


# Training the model

# In[101]:


model = LogisticRegression()
model.fit(X_train, Y_train)


# **Making predictions**

# In[102]:


predictions = model.predict(X_test) #should be Test here because we already trained our model


# In[103]:


print(predictions)


# Evaluation

# In[104]:


accuracy = accuracy_score(predictions, Y_test)


# In[105]:


print(accuracy) 


# It will have atleast 96% chance of predicting well, model well fitted

# In[106]:


# Calculate training accuracy
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

# Calculate test accuracy
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)
print("Test Accuracy:", test_accuracy)

# Determine if the model is overfitting, underfitting, or well-fitted
if train_accuracy > test_accuracy + 0.1:
    print("The model may be overfitting. Consider regularization or simplifying the model.")
elif train_accuracy < 0.7 and test_accuracy < 0.7:
    print("The model may be underfitting. Consider making the model more complex.")
else:
    print("The model seems well-fitted.")


# In[ ]:





# In[ ]:




