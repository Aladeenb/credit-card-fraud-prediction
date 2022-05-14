from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Data pre-processing

# Hot Encoding, converting categorical attributes into numerical

# Drop id column and delete rows with missing values (from bmi column)
data = pd.read_csv('creditcard.csv').dropna()

# Allocate inputs
X = data.iloc[:, :30] # except column 10 (stroke result)

# Allocate outputs
y = data.iloc[:, 30]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Build model using Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 1, max_iter=400).fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# print(classification_report(y_test, y_pred_lr))
print("classification report\n", 
      classification_report(y_test, y_pred_lr),
      "\nconfusion matrix\n",
      confusion_matrix(y_test, y_pred_lr),
      "\n\nAccuracy =",
      accuracy_score(y_test, y_pred_lr)
      )
cm = confusion_matrix(y_test, y_pred_lr) 
display = ConfusionMatrixDisplay(
    confusion_matrix = cm, display_labels = lr.classes_)
display.plot()
plt.show()
