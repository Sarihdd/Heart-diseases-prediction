#!/usr/bin/env python
# coding: utf-8

# # Heart Diseases Prediction Using Different Algorithm

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


df =pd.read_csv(r"C:\Users\Hp\Downloads\heart-disease.csv")
df.head(5)


# In[3]:


df.describe()


# In[10]:






# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='RdYlBu', annot=True)
plt.title('Correlation Matrix')
plt.show()


# In[6]:


import matplotlib.pyplot as plt
# Create a count plot
sns.countplot(x='target', data=df)

# Add labels and title
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.title('Distribution of Heart Disease')


# In[17]:


# Create separate dataframes for heart disease and non-disease cases
heart_disease = df[df['target'] == 1]
non_disease = df[df['target'] == 0]

# Set up the figure and axes
fig, ax = plt.subplots()

# Plot histogram for heart disease cases
sns.histplot(data=heart_disease, x='age', kde=True, color='red', label='Heart Disease')

# Plot histogram for non-disease cases
sns.histplot(data=non_disease, x='age', kde=True, color='blue', label='Non-Disease')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age for Heart Disease and Non-Disease Cases')

# Add legend
plt.legend()

# Display the plot
plt.show()


# In[19]:


# Set up the figure and axes
fig, ax = plt.subplots()

# Create the box plot
sns.boxplot(x='target', y='age', data=df)

# Add labels and title
plt.xlabel('Heart Disease')
plt.ylabel('Age')
plt.title('Comparison of Age between Heart Disease and Non-Disease Cases')

# Display the plot
plt.show()


# In[ ]:





# In[20]:


df.info()


# In[21]:


df.isnull().sum()


# In[ ]:





# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset

df.head(5)


# In[23]:


import warnings
warnings.filterwarnings("ignore")


# Split the dataset into features (X) and target variable (y)
X = df.drop('target', axis=1)
y = df['target']  # Replace 'target_column_name' with the correct column name for the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the logistic regression model
model = LogisticRegression()

# Define the hyperparameters to tune and their possible values
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Perform grid search cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[24]:


from sklearn.metrics import roc_curve, auc
# Compute the false positive rate (FPR), true positive rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Compute the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[25]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[26]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# Plot Precision-Recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()


# In[27]:


# Assuming you have a new data point stored in a variable called "new_data"
new_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]

# Make predictions on new data using the best model
prediction = best_model.predict(new_data)

# Print the prediction
if prediction[0] == 1:
    print("You have heart disease.")
else:
    print("You do not have heart disease.")


# # Now lets explore more ml algorithm lets start....

# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score


# Split the dataset into features (X) and target variable (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
dt_grid_search = GridSearchCV(dt_model, dt_param_grid, cv=5)
dt_grid_search.fit(X_train, y_train)
dt_best_model = dt_grid_search.best_estimator_
dt_y_pred = dt_best_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print("Decision Tree Accuracy:", dt_accuracy)

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5)
rf_grid_search.fit(X_train, y_train)
rf_best_model = rf_grid_search.best_estimator_
rf_y_pred = rf_best_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Accuracy:", rf_accuracy)

# Support Vector Machine (SVM)
svm_model = SVC()
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_grid_search = GridSearchCV(svm_model, svm_param_grid, cv=5)
svm_grid_search.fit(X_train, y_train)
svm_best_model = svm_grid_search.best_estimator_
svm_y_pred = svm_best_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print("SVM Accuracy:", svm_accuracy)

# XGBoost Classifier
xgb_model = XGBClassifier()
xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'gamma': [0, 0.1, 0.2]
}
xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5)
xgb_grid_search.fit(X_train, y_train)
xgb_best_model = xgb_grid_search.best_estimator_
xgb_y_pred = xgb_best_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
print("XGBoost Accuracy:",rf_accuracy)


# In[ ]:





# In[30]:


pip install flask


# In[31]:


from flask import Flask

app = Flask(__name__)


# In[32]:


@app.route('/')
def home():
    return 'Welcome to my website!'


# In[ ]:


if __name__ == '__main__':
    app.run()


# In[ ]:




