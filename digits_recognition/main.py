import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
train_data = pd.read_csv("/kaggle/input/competitions/digit-recognizer/train.csv")
X = np.array(train_data.drop("label", axis=1))
y = np.array(train_data["label"])
image = X[0]

normalized_X = X / 255.0
print(normalized_X.min())
print(normalized_X.max())
X_train, X_val, y_train, y_val = train_test_split(
    normalized_X,
    y,
    test_size=0.2,
    random_state=42
)
W1 = np.random.randn(784,10) * np.sqrt(2 / 784)
W2 = np.random.randn(10,10)* np.sqrt(2 / 10)
b1 = np.zeros((1,10))
b2 = np.zeros((1,10))
Z1 = np.dot(image, W1) + b1
A1 = np.maximum(0, Z1)
Z2 = np.dot(A1, W2) + b2

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))
    return exp_Z / np.sum(exp_Z)

A2 = softmax(Z2)
prediction = np.argmax(A2)
actual_digit = y[0]
print("Predicted Digit:", prediction)
print("Actual Digit:", actual_digit)

correct_probabilty = A2[0, actual_digit]
print(correct_probabilty)

loss = -np.log(correct_probabilty)
print("Loss:", loss)

learning_rate = 0.01
y_true = np.zeros((1,10))
y_true[0, y_train[0]] = 1
dZ2 = A2 - y_true
dW2 = np.dot(A1.T, dZ2)