import numpy as np 
from helper import *
'''
Homework2: logistic regression classifier
'''


def logistic_regression(data, label, max_iter, learning_rate):
	'''
	The logistic regression classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
	'''
	w = np.copy(data[0])
	likelihood = 0

	for index in range(max_iter):
		for i, val in enumerate(data):
			y_w = np.dot(label[i],w)
			likelihood += (np.dot(label[i],data[i,0:]))/(1 + np.exp(np.dot(y_w,data[i,0:])))
		gradient = (-1/len(data)) * likelihood
		w = w - (learning_rate * gradient)
		likelihood = 0
	return w


def thirdorder(data):
	'''
	This function is used for a 3rd order polynomial transform of the data.
	Args:
	data: input data with shape (:, 3) the first dimension represents 
		  total samples (training: 1561; testing: 424) and the 
		  second dimesion represents total features.

	Return:
		result: A numpy array format new data with shape (:,10), which using 
		a 3rd order polynomial transformation to extend the feature numbers 
		from 3 to 10. 
		The first dimension represents total samples (training: 1561; testing: 424) 
		and the second dimesion represents total features.
	'''
	w = np.zeros(shape = (len(data), 10))

	for index in range(len(data)):
		x1 = data[index,1]
		x2 = data[index,0]
		thirdorder = np.array([(1), (x1), (x2), (x1**2), (x1*x2), (x2**2), (x1**3), ((x1**2)*x2), (x1*(x2**2)), (x2**3)])
		for i in range(10):
			w[index, i] = thirdorder[i]

	return w

def accuracy(x, y, w):
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.

    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''

    n, _ = x.shape
    mistakes = 0
    threshold = .5

    for index in range(n):
        lr = 1/(1 + np.exp(np.dot(-w, x[index, :])))
        if lr >= threshold:
            if y[index] != 1:
                mistakes += 1
        else:
            if y[index] != -1:
                mistakes += 1

    accuracy = (n - mistakes)/n
    return accuracy
