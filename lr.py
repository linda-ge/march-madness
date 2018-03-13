import numpy as np
import pandas as pd
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
	s[s == 1] = .999999999 #preventing zeroes
	# if np.subtract(1,s).any() == 0: print "we have a subs prob"
	loss = -1 * np.dot(y, np.log(s)) - np.dot(np.subtract(1, y), np.log(np.subtract(1, s))) + lmbda*np.linalg.norm(w)**2
	return loss[0][0]

## TODO: find data format, extract data
##	Do we need to add a feature to indicate game type
##  What other features could we use? 
def treat_data(data_frame): #takes a list of filenames
	data_frame.drop(labels=['Season', 'WLoc'], inplace=True, axis=1)	
	inv_cols = ['DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore' , 'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF','WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
	norm_cols = ['DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
	win = []
	for i in range(len(data_frame)):
		if data_frame.loc[i]["WTeamID"] > data_frame.loc[i]["LTeamID"]:
			## we need to invert all the W columns and the L columns
			win.append(0)
			data_frame.loc[i, norm_cols] = data_frame.loc[i, inv_cols].values
		else:
			win.append(1)
	data_mat = data_frame.as_matrix()
	np.random.shuffle(data_mat)
	win = np.array(win)
	print win.shape
	win = win.reshape((len(win), 1))
	print win.shape
	return data_mat, win

def get_data_by_season(year): #input is an int
	data_set = pd.read_csv("data/RegularSeasonDetailedResults.csv")
	season = data_set.loc[data_set["Season"] == year]
	return season

def test_batch_grad_1s(year):
	season_data = get_data_by_season(year)
	X, y = treat_data(season_data)

	## Run the batch grad with X and Y
	## will be different then the code below since we are operating on a dataframe object 
	dim_feature = len(X[0])
	w = np.zeros((dim_feature, 1))
	X_t = np.transpose(X)
	lmbda = 0.0005
	epsilon = 0.00000000005
	#compute the first s to have the loss at the first step
	s = func_s(np.dot(X, w))
	loss = [loss_func(w, y, s, lmbda)]
	iter_count = [0]
	#loop
	while iter_count[-1] < 10000:

		#for each iteration we compute w, the new s and save the loss.
		w = batch_update(w, epsilon, X_t, y, s, lmbda)
		s = func_s(np.dot(X, w)) 
		iter_count.append(iter_count[-1] + 1)
		loss.append(loss_func(w, y, s, lmbda))
	plt.plot(iter_count, loss)
	plt.title("Batch Gradient Descent epsilon:")
	plt.savefig("batch.png")

#test_batch_grad_1s(2003)

def validate_batch_grad_1s(year):
	#TODO:implement this method
	#train on 80% of reg season games
	# validate on 20% of the games
	season_data = get_data_by_season(year)
	X, y = treat_data(season_data)

	size_training = int(len(X)*.8)
	training_X = X[:size_training]
	training_y = y[:size_training]

	validating_X = X[size_training:]
	validating_y = y[size_training:]

	## Run the batch grad with X and Y
	## will be different then the code below since we are operating on a dataframe object 
	dim_feature = len(training_X[0])
	w = np.zeros((dim_feature, 1))
	X_t = np.transpose(training_X)
	lmbda = 0.0005
	epsilon = 0.00000000005
	#compute the first s to have the loss at the first step
	s = func_s(np.dot(training_X, w))
	loss = [loss_func(w, training_y, s, lmbda)]
	iter_count = [0]
	#loop
	while iter_count[-1] < 10000:
		#for each iteration we compute w, the new s and save the loss.
		w = batch_update(w, epsilon, X_t, training_y, s, lmbda)
		s = func_s(np.dot(training_X, w)) 
		iter_count.append(iter_count[-1] + 1)
		loss.append(loss_func(w, training_y, s, lmbda))
	plt.plot(iter_count, loss)
	plt.title("Batch Gradient Descent Training epsilon:")
	plt.savefig("batch_training.png")

	classification = []
	for i in xrange(len(validating_X)):
		cls_ = np.asarray(func_s(np.dot(np.asmatrix(validating_X[i]), w)))[0][0]
		classification.append(cls_)

	with open("validation_results.txt", "w+") as f:
		for i, elem in enumerate(classification):
			f.write(str(elem) + ", " + str(validating_y[i]) + "\n")
	classification[classification >= .5] = 1
	classification[classification < .5] = 0
	print "classification accuracy:" + str(np.sum(classification)/float(len(classification)))

#validate_batch_grad_1s(2003)

def test_batch_grad_1s_on_tourney(year):
	season_data = get_data_by_season(year)
	X, y= treat_data(season_data)

	training_X = X
	training_y = y

	## Run the batch grad with X and Y
	## will be different then the code below since we are operating on a dataframe object 
	dim_feature = len(training_X[0])
	w = np.zeros((dim_feature, 1))
	X_t = np.transpose(training_X)
	lmbda = 0.0005
	epsilon = 0.00000000005
	#compute the first s to have the loss at the first step
	s = func_s(np.dot(training_X, w))
	loss = [loss_func(w, training_y, s, lmbda)]
	iter_count = [0]
	#loop
	while iter_count[-1] < 10000:
		#for each iteration we compute w, the new s and save the loss.
		w = batch_update(w, epsilon, X_t, training_y, s, lmbda)
		s = func_s(np.dot(training_X, w)) 
		iter_count.append(iter_count[-1] + 1)
		loss.append(loss_func(w, training_y, s, lmbda))

	plt.plot(iter_count, loss)
	plt.title("Batch Gradient Descent Training epsilon:")
	plt.savefig("tourney_training.png")

	tourney_data = pd.read_csv("data/NCAATourneyDetailedResults.csv")
	season= tourney_data.loc[tourney_data["Season"] == year]
	to_classify, answers = treat_data(season)
	classification = []
	for i in xrange(len(to_classify)):
		cls_ = np.asarray(func_s(np.dot(np.asmatrix(to_classify[i]), w)))[0][0]
		classification.append(cls_)

	with open("tourney_test_results.txt", "w+") as f:
		for i, elem in enumerate(classification):
			f.write(str(elem) + ", " + str(answers[i]) + "\n")
	classification[classification >= .5] = 1
	classification[classification < .5] = 0
	print "classification accuracy:" + str(np.sum(classification)/float(len(classification)))

test_batch_grad_1s_on_tourney(2003)

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
# run_stoch_dec_epsilon()

