import numpy as np


# Regression function 
# yi = w*xi + b

def predictY(x,w,b):
    return w*x + b;


def drawLine(w,b):
    x=np.array([0,10])
    y=predictY(x,w,b)
    plt.plot(x,y,'r-')
    plt.scatter(x,y)
    plt.show()

# Cost function
# J(w,b) = (1/m) * sum((predictY(xi,w,b) - yi)^2)

def costFunction(x,y,w,b):
    m = len(x)
    cost = 0
    for i in range(m):
        cost += (predictY(x[i],w,b) - y[i])**2
    return cost/m

# Derivative of cost function with respect to w
#dw = (1/m) * sum((predictY(xi,w,b) - yi) * xi)
#dw = d((1/m) * sum((predictY(xi,w,b) - yi)^2))/dw
#dw = (1/m) * sum(2*(predictY(xi,w,b) - yi) * d(predictY(xi,w,b))/dw)
#dw = (1/m) * sum(2*(predictY(xi,w,b) - yi) * xi)
#dw = (1/m) * sum((predictY(xi,w,b) - yi) * xi)
def dw(x,y,w,b):
    m = len(x)
    dw = 0
    for i in range(m):
        dw += (predictY(x[i],w,b) - y[i]) * x[i]
    return dw/m

# Derivative of cost function with respect to b
#db = d((1/m) * sum((predictY(xi,w,b) - yi)^2))/db
#db = (1/m) * sum(2*(predictY(xi,w,b) - yi) * d(predictY(xi,w,b))/db)
#db = (1/m) * sum(2*(predictY(xi,w,b) - yi) * 1)
#db = (1/m) * sum((predictY(xi,w,b) - yi))
def db(x,y,w,b):
    m = len(x)
    db = 0
    for i in range(m):
        db += (predictY(x[i],w,b) - y[i])
    return db/m



# Gradient descent
# w = w - alpha * dw
# b = b - alpha * db
# alpha = learning rate
# num_iters = number of iterations
# Learn this slope for more clarity
# https://www.youtube.com/watch?v=EQoNfxToez0


def gradientDescent(x,y,w,b,alpha,num_iters,J_history,p_history,w_history,b_history):
    m = len(x)
    for i in range(num_iters):
        dw = 0
        db = 0
        for j in range(m):
            dw += (predictY(x[j],w,b) - y[j]) * x[j]
            db += (predictY(x[j],w,b) - y[j])
            print("dw: ", dw)
            print("db: ", db)
        # As slope decreases, step size decreases  with below formulaand we get closer to the minimum cost function    
        w = w - alpha * dw/m
        b = b - alpha * db/m
        print("w: ", w)
        print("b: ", b)
        cost = costFunction(x,y,w,b)
        J_history.append(cost)
        p_history.append(predictY(x,w,b))
        w_history.append(w)
        b_history.append(b)
    return w,b


# Draw the gradient descent with respect to the cost function
import matplotlib.pyplot as plt 





# main function
def main(): 
    w = 0
    b = 0
    alpha = 0.01
    num_iters = 100
    x=np.array([1,2,3,4,5,6,7,8,9,10])
    y=np.array([2,4,6,8,10,12,14,16,18,20])
    J_history = []
    p_history = []
    w_history = []
    b_history = []

    gradientDescent(x,y,w,b,alpha,num_iters,J_history,p_history,w_history,b_history)
    plt.plot(b_history,J_history,'ro-')
    plt.xlabel("b")
    plt.ylabel("cost")
    plt.show()

    print("w: ", w)
    print("b: ", b)
if __name__ == "__main__":
    main()
        