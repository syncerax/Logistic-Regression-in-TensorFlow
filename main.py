import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = np.loadtxt("ex2data1.txt",delimiter = ",")

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,:2],dataset[:,2],test_size = 0.2,random_state = 1)

m,n = X_train.shape
print("Number of training examples:",m)
print("Number of training features:",n)

plt.scatter(X_train[:,0],X_train[:,1],c = Y_train)
plt.show()

#TensorFlow time!