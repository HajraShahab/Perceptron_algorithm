# no imports beyond numpy should be used in answering this question
import numpy as np
import matplotlib.pyplot as plt

# train datapoints: 2 features and binary output
train_separable = np.array([[2.7810836,2.550537003,-1],
    [1.465489372,2.362125076,-1],
    [3.396561688,4.400293529,-1],
    [1.38807019,1.850220317,-1],
    [3.06407232,3.005305973,-1],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]])

# train datapoints nonseparable: 2 features and binary output
train_nonseparable = np.array([[2.7810836,2.550537003,-1],
    [1.465489372,2.362125076,-1],
    [3.396561688,4.400293529,1],
    [1.38807019,1.850220317,-1],
    [3.06407232,3.005305973,-1],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,-1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]])

# plt.scatter(train_separable[:,0], train_separable[:,1], c= train_separable[:,2], marker='s')
plt.scatter(train_nonseparable[:,0], train_nonseparable[:,1], c= train_nonseparable[:,2], marker='s'

# test datapoints: 2 features and binary output
test = np.array([[1.927628496,7.200103829,-1],
    [9.182983992,5.290012983,1]])

x1_train = train_separable[:,0]
x2_train = train_separable[:,1]

plt.scatter(x1_train,x2_train, color = 'red', label = 'train')
x1_test = test_x[:,0]
x2_test = test_x[:,1]

plt.scatter(x1_test,x2_test, color = 'blue', label = 'test')
plt.legend()
x1_line = [i for i in range(10)]
x2_line = []
for i in x1_line:
    x2_line.append((-1+(2.06536*i)/ (2.3418)))
plt.plot(x1_line, x2_line, label = 'perceptron', color = 'green')   
plt.show()

##
x3_train = train_nonseparable[:,0]
x4_train = train_nonseparable[:,1]

plt.scatter(x3_train,x4_train, color = 'red', label = 'train')
x3_test = test_x[:,0]
x4_test = test_x[:,1]

plt.scatter(x3_test,x4_test, color = 'blue', label = 'test')
plt.legend()
plt.show()


##
plt.plot(sep, color = 'red', label = 'separable')
plt.plot(nonsep, color = 'green', label = 'nonseparable')
plt.legend()
plt.show()

#number of epochs
n_epoch = 100

def predict(x, theta):
    """
    Predict y_hat in {-1,1} for a given input x and parameters theta

    Input:
    x: Numpy ndarray shape (2,)
    theta: NumPy ndarray shape (3,), where theta[0] is the bias term
        and theta[1:]

    Return:
    y_hat: -1 or 1
    """
    # Code to make sure the input has the correct size
    assert x.shape == (2,)
    assert theta.shape == (3,)
    # Do not edit any code in this function outside the edit region
    ### BEGIN YOUR CODE ###
    initialize = theta[0]
    for i in range(len(x)):
        initialize += theta[i+1] * x[i]
    y_hat = 1 if initialize >= 0 else -1 
    ### END YOUR CODE ###
    return y_hat
# bias + w1*x1 +w2*x2

#Run perceptron algorithm
def train(train_x, train_y, n_epoch):
    theta = np.zeros(train_x.shape[1]+1)

    train_error = []

    # Train for a fixed number of epochs regardless of convergence
    for epoch in range(n_epoch):
        num_error = 0.0
        for i in range(train_x.shape[0]):
            x = train_x[i,:]
            y_hat = predict(x, theta)

            if train_y[i] != y_hat:
                num_error += 1

            # Update the bias term "theta[0]" and the feature-specific weights "theta[1:]" in the code block below
            ### BEGIN YOUR CODE ###
                x_bias = np.insert(x, 0, np.ones(1))
                theta = theta + train_y[i]*x_bias
                #print(theta)
                #error = x[-1] - y_hat
                #for i in range(len(x)-1):
                #    theta[i+1] = theta[i+1] + train_y * error * x[i] 
            ### END YOUR CODE ###
        
        print('epoch={}, errors={}'.format(epoch, num_error))
        train_error.append(num_error)
    return theta, train_error



if __name__ == '__main__':
    # split train data into features and predictions
    train_separable_x = train_separable[:,:-1]
    train_separable_y = train_separable[:,-1]
    
    # split non-separable train data into features and predictions
    train_nonseparable_x = train_nonseparable[:,:-1]
    train_nonseparable_y = train_nonseparable[:,-1]
    
    # split test data into features and predictions
    test_x = test[:,:-1]
    test_y = test[:,-1]

    # train theta using training data
    print('Train: separable')
    theta, sep = train(train_separable_x, train_separable_y, n_epoch)
    print(theta)
    # make predictions on test data and compare with groundtruth
    print('Test')
    for i in range(test.shape[0]):
        x = test_x[i,:]
        prediction = predict(x, theta)
        print("Expected={}, Predicted={}".format(test_y[i], prediction))
    print()

    # train theta using non-separable training data
    print('Train: nonseparable')
    theta,nonsep = train(train_nonseparable_x, train_nonseparable_y, n_epoch)

    # make predictions on test data and compare with groundtruth
   # print('Test')
    for i in range(test.shape[0]):
        x = test_x[i,:]
        y_hat = predict(x, theta)
      #  print("y={}, y_hat={}".format(test_y[i], y_hat))

