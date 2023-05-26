# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Read the given dataset.
2. Fitting the dataset into the training set and test set.
3. Applying the feature scaling method.
4. Fitting the logistic regression into the training set.
5. Prediction of the test and result
6. Making the confusion matrix
7. Visualizing the training set results. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by : REVATHI.D
RegisterNumber :  212221240045
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter = ',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()


plt.show()

    return 1 / (1 + np.exp(-z))
    
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y,np.log(1-h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J,grad
    
X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J
def gradient(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    grad = np.dot(X.T,h-y)/X.shape[0]
    return grad
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta  = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min, x_max = X[:, 0].min() - 1,X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1,X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max, 0.1),np.arange(y_min,y_max, 0.1))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot = np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)),X))
    prob = sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X) == y)

```

## Output:
Array value of x:

![image](https://github.com/Revathi-Dayalan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/96000574/ea282209-1004-4ce8-a662-5ace9e45027f)

Array Value of y:

![image](https://github.com/Revathi-Dayalan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/96000574/f07f6518-3745-45b9-b2dc-ae1a0a9342b3)

Exam 1- Score graph:

![image](https://github.com/Revathi-Dayalan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/96000574/d9bbc8db-713c-4d84-a570-f4d180524774)

Sigmoid Function Graph

![image](https://github.com/Revathi-Dayalan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/96000574/d71ef608-51fc-43e0-9261-5c68328a1130)

X_train_grad value

![image](https://github.com/Revathi-Dayalan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/96000574/7ac6ea58-4df6-4e10-b482-540f8f98ef48)

Y_train_grad value

![image](https://github.com/Revathi-Dayalan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/96000574/978c8e0b-1e0a-4a1d-abdb-8d9343fdd68d)

Print res.x

![image](https://github.com/Revathi-Dayalan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/96000574/c2162722-d3dd-4c0b-aa55-5e0fe573df33)

Decision Boundary grapg for Exam Score

![image](https://github.com/Revathi-Dayalan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/96000574/7dcbcdc9-6813-4f40-a007-af3a3a5a90f9)

Probability value

![image](https://github.com/Revathi-Dayalan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/96000574/5baceb8c-45b7-43d0-9ce4-a3044e963cf9)

Prediction value of mean

![image](https://github.com/Revathi-Dayalan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/96000574/0073224f-be11-4a66-92f1-4dad4673d4e5)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

