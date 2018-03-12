import numpy as np
from scipy import io as sio
from matplotlib import pyplot as plt
from random import shuffle as shuffle
#logistic regression bath gradient descent

def func_s(v): #v is the result of np.dot(X,w)
	vec_s = np.apply_along_axis(lambda x: 1/(1 + np.exp(-x)), 0, v)
	return vec_s

def stoch_s(X_iw):
	return 1/(1+np.exp(-X_iw))

def batch_update(w, epsilon, X_t, y, s, lmbda):
	w_u = np.add(w, np.multiply(epsilon, np.dot(X_t, np.subtract(y, s)))) - epsilon*lmbda*w*2
	return w_u

def stoch_update(w, epsilon, X_i_t, y_i, s_i, lmbda):
	first_mult = epsilon*( y_i - s_i)
	second_mult = np.multiply(X_i_t, first_mult)
	w_u =  np.add(w, second_mult) - epsilon*lmbda*w*2
	return w_u
def loss_func(w, y, s, lmbda):
	y = np.transpose(y)
	loss = -1 * np.dot(y, np.log(s)) - np.dot(np.subtract(1, y), np.log(np.subtract(1, s))) + lmbda*np.linalg.norm(w)**2
	return loss[0][0]

## TODO: find data format, extract data
##	Do we need to add a feature to indicate game type
##  What other features could we use? 
def treat_data(filenames): #takes a list of fileanams
	## TODO: implement this method
	pass

## TODO: graph error rate for different learning rates training on validiting on a season at the time
def find_eps_lmd():
	## TODO: implement this method
	pass
## TODO: With given epsilon and lambda compte batch gradient descent on a season at the time
def batch_grad_1s():
	## TODO: implement this method
	##  Should return an error rate output on validation for each season
	##QUESTION: Is there a clever to generate graphs for this? 
	pass

## TODO: implement gradient descent for all the seasons at once 
def batch_grad_all():
	## TODO: implement this method
	pass
## TODO: implement gradient for one season at the time, average the weights, validate on 20% combined games of all seasons
def batch_grad_av():
	## TODO: impement this method
def run_batch():
	data = sio.loadmat("data.mat")
	X = data["X"]
	y = data["y"]
	#find dim of feature. 
	dim_feature = len(X[0])
	w = np.zeros((dim_feature, 1))
	X_t = np.transpose(X)
	lmbda = 0.05
	epsilon = 0.00000005
	#compute the first s to have the loss at the first step
	s = func_s(np.dot(X, w))
	loss = [loss_func(w, y, s, lmbda)]
	iter_count = [0]
	#loop
	while iter_count[-1] < 100000:

		#for each iteration we compute w, the new s and save the loss.
		w = batch_update(w, epsilon, X_t, y, s, lmbda)
		s = func_s(np.dot(X, w)) 
		iter_count.append(iter_count[-1] + 1)
		loss.append(loss_func(w, y, s, lmbda))

	plt.plot(iter_count, loss)
	plt.title("Batch Gradient Descent epsilon:")
	plt.savefig("batch.png")


def train_batch():
	data = sio.loadmat("data.mat")
	X = data["X"]
	y = data["y"]
	X = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, X)
	#find dim of feature. 
	dim_feature = len(X[0])
	w = np.zeros((dim_feature, 1))
	X_t = np.transpose(X)
	lmbda = 0.00005
	epsilon = .005
	#compute the first s to have the loss at the first step
	s = func_s(np.dot(X, w))
	loss = [loss_func(w, y, s, lmbda)]
	iter_count = [0]
	#loop
	while iter_count[-1] < 500000:
		#for each iteration we compute w, the new s and save the loss.
		w = batch_update(w, epsilon, X_t, y, s, lmbda)
		s = func_s(np.dot(X, w)) 
		iter_count.append(iter_count[-1] + 1)
		loss.append(loss_func(w, y, s, lmbda))
	return w

def classify_using_batch():
	w = train_batch()
	test_data = sio.loadmat("data.mat")["X_test"]
	classified = []
	counter = 0
	for i in range(len(test_data)):
		cls_ = np.asarray(func_s(np.dot(np.asmatrix(test_data[i]), w)))[0][0]
		if cls_ >= 0.5:
			counter += 1
			classified.append(1)
		elif cls_ < 0.5:
			classified.append(0)
		else: 
			raise Exception()
	with open("kaggle_wine_submission.csv", "w+") as f:
		f.write("Id,Category\n")
		for i in range(len(classified)):
			f.write(str(i) + ','+str(classified[i]) + "\n")
	print counter

def run_stoch():
	data = sio.loadmat("data.mat")
	X = data["X"]
	y = data["y"]
	norm_X = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, X)
	combined = []
	for i in range(len(X)):
		combined.append((norm_X[i], y[i]))
	dim_feature = len(X[0])
	w = np.zeros((dim_feature, 1))
	lmbda = 0.0005
	epsilon = 0.001
	loss = []
	iter_count = [-1]
	for c in xrange(1):
		#shuffle the data
		shuffle(combined)
		for i in range(len(X)):
			s_i = stoch_s(np.dot(np.asmatrix(combined[i][0]), w)) #the first s
			loss.append(loss_func(w, y, func_s(np.dot(norm_X, w)), lmbda))
			iter_count.append(iter_count[-1] + 1) 
			w = stoch_update(w, epsilon, np.transpose(np.asmatrix(combined[i][0])), combined[i][1][0], np.asarray(s_i)[0][0], lmbda)
	plt.plot(iter_count[1:], loss)
	plt.title("Stochastic Gradient Descent")
	plt.savefig("stoch.png")

def run_stoch_dec_epsilon():
	data = sio.loadmat("data.mat")
	X = data["X"]
	y = data["y"]
	norm_X = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, X)
	combined = []
	for i in range(len(X)):
		combined.append((norm_X[i], y[i]))
	dim_feature = len(X[0])
	w = np.zeros((dim_feature, 1))
	lmbda = 0.0005
	epsilon = 0.001
	loss = []
	iter_count = [-1]
	for c in xrange(2):
		#shuffle the data
		shuffle(combined)
		for i in range(0, len(X)):
			epsilon = epsilon/(i+1)
			s_i = stoch_s(np.dot(np.asmatrix(combined[i][0]), w)) #the first s
			loss.append(loss_func(w, y, func_s(np.dot(norm_X, w)), lmbda))
			iter_count.append(iter_count[-1] + 1) 
			w = stoch_update(w, epsilon, np.transpose(np.asmatrix(combined[i][0])), combined[i][1][0], np.asarray(s_i)[0][0], lmbda)
	plt.figure(2)
	plt.plot(iter_count[1:], loss)
	plt.title("Stochastic Gradient Descent Decreasing Eps")
	plt.savefig("stoch_dec.png")

# run_stoch()
run_stoch_dec_epsilon()