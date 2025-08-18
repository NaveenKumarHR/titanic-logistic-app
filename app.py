#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report


# In[2]:


# Add title
st.title("Titanic Survival Prediction")

# Load data
train_df = pd.read_csv("Titanic_train.csv")
test_df = pd.read_csv("Titanic_test.csv")

st.write("Training Data Shape:", train_df.shape)
st.write("Testing Data Shape:", test_df.shape)

# Data Info and Description
st.markdown("## Data Overview")
st.write(train_df.info())
st.write(train_df.describe())

# Visualizations
st.markdown("## Data Visualizations")

# Survival Distribution
fig1, ax1 = plt.subplots()
sns.countplot(data=train_df, x='Survived', ax=ax1)
plt.title("Survival Distribution")
st.pyplot(fig1)

# Survival by Gender
fig2, ax2 = plt.subplots()
sns.countplot(data=train_df, x='Sex', hue='Survived', ax=ax2)
plt.title("Survival by Gender")
st.pyplot(fig2)

# Age Distribution
fig3, ax3 = plt.subplots()
sns.histplot(train_df['Age'].dropna(), bins=30, kde=True, ax=ax3)
plt.title("Age Distribution")
st.pyplot(fig3)

# Correlation Heatmap
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax4)
plt.title("Correlation Heatmap")
st.pyplot(fig4)


# In[4]:


# Fill missing Age with median
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

# Fill missing Embarked with mode
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

# Encode categorical variables
label_enc = LabelEncoder()
for col in ['Sex', 'Embarked']:
    train_df[col] = label_enc.fit_transform(train_df[col])
    test_df[col] = label_enc.transform(test_df[col])

# Drop irrelevant columns
drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
train_df = train_df.drop(columns=drop_cols, errors='ignore')
test_df = test_df.drop(columns=drop_cols, errors='ignore')

# Separate features & target
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]


# In[5]:


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[6]:


log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)


# In[7]:


# Model Results
st.markdown("## Model Performance")
y_pred = log_reg.predict(X_val)
y_prob = log_reg.predict_proba(X_val)[:,1]

st.write("Accuracy:", accuracy_score(y_val, y_pred))
st.write("Precision:", precision_score(y_val, y_pred))
st.write("Recall:", recall_score(y_val, y_pred))
st.write("F1-score:", f1_score(y_val, y_pred))
st.write("ROC-AUC:", roc_auc_score(y_val, y_prob))

st.write("\nClassification Report:\n", classification_report(y_val, y_pred))

# ROC Curve
fig5, ax5 = plt.subplots()
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_val, y_prob):.2f})")
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
st.pyplot(fig5)

# Feature Importance
st.markdown("## Feature Importance")
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": log_reg.coef_[0]
}).sort_values(by="Coefficient", ascending=False)
st.write(coef_df)


# ## Interview Questions

# In[ ]:


#get_ipython().run_line_magic('pinfo', 'recall')
#Metric	Meaning	Formula	Used When You Care About...
#Precision	Out of all predicted positives, how many were actually correct?	Precision = TP / (TP + FP)	Avoiding false positives
#Recall	Out of all actual positives, how many did we correctly identify?	Recall = TP / (TP + FN)	Avoiding false negatives

#Example (Medical test):

#Precision: If a test says you're sick, how likely is it correct?

#Recall: If you're sick, how likely is the test to catch it?


# In[ ]:


#get_ipython().run_line_magic('pinfo', 'classification')
#Cross-validation is a technique to evaluate model performance by splitting the dataset into multiple parts (folds), training on some, and testing on the rest — multiple times.

#Most common type: k-Fold Cross-Validation
#Split data into k subsets (folds)

#For each fold: use it as test data, and the remaining as training data

#Repeat k times, then average the performance metrics

#Importance in Binary Classification:
#Reduces overfitting risk

#Ensures model performance is not biased by a lucky or unlucky train-test split

#Gives a more reliable estimate of real-world performance

#Helps in hyperparameter tuning



# In[11]:


import joblib
joblib.dump(log_reg, "logistic_model.pkl")


# In[ ]:


#https://titanic-logistic-app-4njm5yvyqixbznwrhhze4k.streamlit.app/

