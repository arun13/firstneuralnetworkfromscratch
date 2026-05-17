import numpy as np
import matplotlib.pyplot as plt


#Sigmod formula 
#The sigmoid function is defined as: g(z) = 1 / (1 + e^(-z))
#Where: z = w*x+b (the linear combination of input features and parameters)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    
    for i in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient
        
    return theta
#Cost function for logistic regression is evaluated:
# h = sigmoid(z) where z = w*x + b
# J(w,b) = 1/m*sum(log(h)) for y=1 and J(w,b) = 1/m*sum(log(1-h)) for y=0

# The cost function is defined as:
# J(w,b) = 1/m*sum(-y*log(h) - (1-y)*log(1-h))
def  cost_function(X, y, theta):
    m = len(y)
    z = np.dot(X, theta)
    h = sigmoid(z)
    cost = (-1/m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    return cost 


def gradient_descent(X, y, theta, learning_rate=0.01, num_iterations=1000):
    m = len(y)
    for i in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient
    return theta    



def predict(X, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)
    return h >= 0.5

def plot_decision_boundary(X, y, theta):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    Z = predict(np.c_[xx.ravel(), yy.ravel()], theta)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()



    def main():
        # Sample dataset
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)
        
        # Train logistic regression model
        theta = logistic_regression(X, y)
        
        # Plot decision boundary
        plot_decision_boundary(X, y, theta) 

if __name__ == "__main__":
    main()  