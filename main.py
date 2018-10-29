# Import Libraries
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load Dataset
dataset = np.loadtxt("ex2data1.txt", delimiter=",")
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :2], dataset[:, 2], test_size=0.2, random_state=1)

Y_train = Y_train.reshape((Y_train.shape[0], 1))
Y_test = Y_test.reshape((Y_test.shape[0], 1))

# Preprocessing the data.
# Mean normalisation and feature scaling.
def preprocess(X, mean=None, std=None):
    m, n = X.shape
    if mean is None and std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        X = (X - mean) / std
        return X, mean, std
    else:
        X = (X - mean) / std
        return X

X_train, mean, std = preprocess(X_train)
X_test = preprocess(X_test, mean, std)

# Display stats
m, n = X_train.shape
print("Number of training examples:", m)
print("Number of training features:", n)
print("Shape of X_train:", X_train.shape)
print("Shape of Y_train:", Y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of Y_test:", Y_test.shape)

# Visualise the data
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 0], cmap=plt.cm.coolwarm, label='Training set')
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test[:, 0], cmap=plt.cm.coolwarm, marker='*', label='Testing set')
plt.legend()
plt.title("Data points")
plt.show()

# Creating the model

# Placeholders and variables
X = tf.placeholder(tf.float32, [None, n], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')

W = tf.Variable(tf.zeros([n, 1]))
b = tf.Variable(tf.zeros([1]))

# Hypothesis function
hypothesis = tf.sigmoid(tf.matmul(X, W) + b, name='hypothesis')
predict = tf.round(hypothesis, name='prediction')

# Cost function
cost = -tf.reduce_mean((Y * tf.log(hypothesis)) + ((1 - Y) * tf.log(1 - hypothesis)), name="cost")
# Optimizer
# Using the simple Gradient Descent algorithm, with learning rate = 0.01.
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Training the model
init = tf.global_variables_initializer()
epochs = 5000
history = []
test_history = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        dummy, c = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
        history.append(c)

        if epoch % 1000 == 0:
            print("Epoch {}: Cost = {}".format(epoch, c))

        test_c = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
        test_history.append(test_c)

    train_predictions = sess.run(predict, feed_dict={X: X_train, Y: Y_train})
    test_predictions = sess.run(predict, feed_dict={X: X_test, Y: Y_test})

    weights = sess.run(W)
    bias = sess.run(b)

# Calculate Accuracy
train_accuracy = np.mean(Y_train == train_predictions) * 100
test_accuracy = np.mean(Y_test == test_predictions) * 100
print("Training set accuracy:", train_accuracy)
print("Testing set accuracy:", test_accuracy)

# Plot Cost vs Iterations
plt.plot(history, label="train cost")
plt.plot(test_history, label="test cost")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost over Iterations")
plt.show()

# Plotting the decision boundary
slope = - weights[0][0] / weights[1][0]
intercept = - (bias[0] - 0.5) / weights[1][0]
x_lim = [X_train[:, 0].min(), X_train[:, 0].max()]
y_lim = [slope * i + intercept for i in x_lim]
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 0], cmap=plt.cm.coolwarm, label="train set")
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test[:, 0], marker='*', cmap=plt.cm.coolwarm, label="test set")
plt.plot(x_lim, y_lim)
plt.title("Decision Boundary")
plt.legend()
plt.show()