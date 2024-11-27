#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target labels


# In[2]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[3]:


# Create the SVM model with a linear kernel
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Train the model on the training data
svm_model.fit(X_train, y_train)


# In[4]:


# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[5]:


# Limit the dataset to two features for visualization
X_vis = X[:, :2]  # Use only the first two features
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.3, random_state=42)

# Train a new model with the limited feature set
svm_model_vis = SVC(kernel='linear', C=1.0, random_state=42)
svm_model_vis.fit(X_train_vis, y_train_vis)

# Plot decision boundaries
plt.figure(figsize=(10, 6))
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = svm_model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Decision Boundaries on Iris Dataset')
plt.show()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the MNIST dataset
digits = datasets.load_digits()
X = digits.data  # Flattened pixel data (64 pixels per image for 8x8 images)
y = digits.target  # Digit labels (0 to 9)


# In[7]:


# Standardize the feature values
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[9]:


# Create the SVM model with an RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the model on the training data
svm_model.fit(X_train, y_train)


# In[10]:


# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[11]:


# Plot a few test images with their predicted labels
plt.figure(figsize=(10, 6))
for index, (image, prediction) in enumerate(zip(X_test[:8], y_pred[:8])):
    plt.subplot(2, 4, index + 1)
    plt.imshow(image.reshape(8, 8), cmap=plt.cm.gray)
    plt.title(f"Predicted: {prediction}")
    plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




