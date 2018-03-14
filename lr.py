import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import special
from matplotlib import pyplot as plt
from random import shuffle as shuffle
#logistic regression bath gradient descent

def func_s(v): #v is the result of np.dot(X,w)
	vec_s = np.apply_along_axis(lambda x: special.expit(x), 0, v)
	return vec_s

def stoch_s(X_iw):
	return 1/(1+np.exp(-X_iw))

def batch_update(w, epsilon, X_t, y, s, lmbda):
	w_u = np.add(w, np.multiply(epsilon, np.dot(X_t, np.subtract(y, s)))) - epsilon*lmbda*w*2
	# w_u = np.add(w, np.multiply(epsilon, np.dot(X_t, np.subtract(y, s)))) 
	return w_u

def stoch_update(w, epsilon, X_i_t, y_i, s_i, lmbda):
	first_mult = epsilon*( y_i - s_i)
	second_mult = np.multiply(X_i_t, first_mult)
	w_u =  np.add(w, second_mult) - epsilon*lmbda*w*2
	return w_u

def loss_func(w, y, s, lmbda):
	y = np.transpose(y)
	s[s == 1] = 0.9999999999999 #preventing zeroes
	s[s == 0] = 0.0000000000001
	# if np.subtract(1,s).any() == 0: print "we have a subs prob"
	loss = -1 * np.dot(y, np.log(s)) - np.dot(np.subtract(1, y), np.log(np.subtract(1, s))) + lmbda*np.linalg.norm(w)**2
	# loss = -1 * np.dot(y, np.log(s)) - np.dot(np.subtract(1, y), np.log(np.subtract(1, s))) 
	return loss[0][0]

## TODO: find data format, extract data
##	Do we need to add a feature to indicate game type
##  What other features could we use?
def get_average_stats(year): #TODO: make everything numpy arrays and add as if they were matrices
	## compiles the average stats
	data_set = pd.read_csv('data/RegularSeasonDetailedResults_Prelim2018.csv')
	season = data_set.loc[data_set["Season"] == year]
	season.reset_index(drop=True, inplace=True)
	avTable = np.zeros((1000, 14))
	counter = [0.0]*1000
	for i in range(len(season)):
		#get the stats of both ids
		id1 = season.loc[i]["WTeamID"]
		id_1 = id1 - 1000
		mat_1 = season.loc[i, ['WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].as_matrix()
		mat_1 = mat_1.astype('f')
		avTable[id_1] += mat_1
		counter[id_1] += 1
		
		id2 = season.loc[i]["LTeamID"]
		id_2 = id2 - 1000
		mat_2 = season.loc[i, ['LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].as_matrix()
		mat_2 = mat_2.astype('f')
		avTable[id_2] += mat_2
		counter[id_2] += 1

	for i in range(1000):
		if counter[i] > 0:
			avTable[i] = avTable[i] / counter[i]

	return season, avTable
def treat_data(data_frame, avTable):
	data_frame.drop(labels=['Season', 'WLoc'], inplace=True, axis=1)	
	inv_cols = ['DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore' , 'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF','WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
	norm_cols = ['DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
	data_mat = np.zeros((len(data_frame), 14))
	win = []
	for i in range(len(data_frame)):
		if data_frame.loc[i]["WTeamID"] > data_frame.loc[i]["LTeamID"]:
			## we need to invert all the W columns and the L columns
			win.append(0)
			data_frame.loc[i, norm_cols] = data_frame.loc[i, inv_cols].values
		else:
			win.append(1)
		# compute point difference
		id_1 = data_frame.loc[i]['WTeamID'] - 1000 
		mat_1 = avTable[id_1]
		id_2 = data_frame.loc[i]['LTeamID'] - 1000
		mat_2 = avTable[id_2]

		diff_mat = mat_1 - mat_2
		data_mat[i] += diff_mat
	win = np.array(win)
	win = win.reshape((len(win), 1))
	return data_mat, win

# def get_data_by_season(year): #input is an int
# 	data_set = pd.read_csv("data/RegularSeasonDetailedResults.csv")
# 	season = data_set.loc[data_set["Season"] == year]
# 	season.reset_index(drop=True, inplace=True)
# 	return season


# test_batch_grad_1s(2003)
def read_treated_data(year):
	data_filename= "data/treated" + str(year) + ".npy"
	cls_filename = "data/treated" + str(year) + "Cls.npy"
	data = np.load(data_filename)
	cls_ = np.load(cls_filename)
	return data, cls_

## TODO: With given epsilon and lambda compte batch gradient descent on a season at the time
def treat_all_seasons(years):
	for y in years:
		data_filename = "data/treated" + str(y) + ".npy"
		cls_filename = "data/treated" + str(y) + "Cls.npy"
		season, avTable = get_average_stats(y)
		data, cls_ = treat_data(season, avTable)
		np.save(data_filename, data)
		np.save(cls_filename, cls_)


# treat_all_seasons([x for x in range(2003, 2019)])

def get_weights_for_all_years(years):
	epsilon = 0.00000005
	lmbda = 0.0005
	weight_array = []
	for year in years:
		X, y = read_treated_data(year)
		w = batch_training(X, y, epsilon, lmbda, year)
#

def batch_training(X, y, epsilon, lmbda, year):
	dim_feature = len(X[0])
	w = np.ones((dim_feature, 1))
	X_t = np.transpose(X)
	s = func_s(np.dot(X, w))
	loss = [loss_func(w, y, s, lmbda)]
	iter_count = [0]
	#loop
	while iter_count[-1] < 20000:
		#for each iteration we compute w, the new s and save the loss.
		w = batch_update(w, epsilon, X_t, y, s, lmbda)
		s = func_s(np.dot(X, w)) 
		iter_count.append(iter_count[-1] + 1)
		loss.append(loss_func(w, y, s, lmbda))
	print w
	print loss[-1]
	plt.plot(iter_count, loss)
	plt.title("Batch Gradient Descent Training epsilon:")
	training_loss_filename = "tourney_training_" + str(year) + ".png"
	plt.savefig(training_loss_filename)
	weight_filename = "weights_" + str(year) + ".npy"
	np.save(weight_filename, w)
	# save the weights somewhere
	return w

# get_weights_for_all_years([x for x in range(2003, 2017)])

# X,y = read_treated_data(2003)
# print X.shape 
# print y.shape
# batch_training(X, y, 0.00000005, 0.0005, 2003)
def validate_on_year(w, y):
	to_classify, answers = get_treated_tournament(y)
	classification = []
	for i in xrange(len(to_classify)):
		cls_ = np.asarray(func_s(np.dot(np.asmatrix(to_classify[i]), w)))[0][0]
		classification.append(cls_)
	classification = np.array(classification)
	res_filename = "tourney_" + str(y) + "_test_results.txt"
	with open(res_filename, "w+") as f:
		for i, elem in enumerate(classification):
			f.write(str(elem) + ", " + str(answers[i]) + "\n")
	classification[classification >= .5] = 1
	classification[classification < .5] = 0
	classification = classification.reshape((classification.shape[0], 1))
	print "classification accuracy: " + str(np.sum(classification == answers)/float(len(answers)))

def validate_on_years(years):
	for year in years:
		data_filename = "weights_" + str(year) + ".npy"
		w = np.load(data_filename)
		validate_on_year(w, year)

def get_treated_tournament(year):
	data_filename = "data/treatedTournament" + str(year) + ".npy"
	cls_filename = "data/treatedTournament" + str(year) + "Cls.npy"
	data = np.load(data_filename)
	cls_ = np.load(cls_filename)
	return data, cls_

# validate_on_years([x for x in range(2003, 2017)])

def treat_tournament_data(years):
	for y in years:
		data_filename = "data/treatedTournament" + str(y) + ".npy"
		cls_filename = "data/treatedTournament" + str(y) + "Cls.npy"
		tourney_data = pd.read_csv("data/NCAATourneyDetailedResults.csv")
		tournament_matchup = tourney_data.loc[tourney_data["Season"] == y]
		tournament_matchup.reset_index(drop=True, inplace=True)
		season, avTable = get_average_stats(y)
		to_classify, answers = treat_data(tournament_matchup, avTable)
		np.save(data_filename, to_classify)
		np.save(cls_filename, answers)

# treat_tournament_data([x for x in range(2003, 2017)])
# treat_tournament_data([2017])
# get_weights_for_all_years([2017])
# validate_on_years([2017])

def compute_average(years):
	sum_ = 0
	total = len(years)
	for year in years:
		weight_filename = "weights_" + str(year) + ".npy"
		sum_ += np.load(weight_filename)
	av = sum_ / total
	return av

# av_weights = compute_average([x for x in range(2003, 2018)])
def graph_average(w):
	plt.scatter([x for x in range(len(w))], w)
	plt.title("averaged weights for years 2003 through 2014")
	plt.savefig("average_weights.png")
	plt.clf()

# graph_average(av_weights)


def graph_weights(years):
	for year in years:
		weight_filename = "weights_" + str(year) + ".npy"
		w = np.load(weight_filename)
		plt.scatter([x for x in range(len(w))], w)
		plt.title("weights for years 2003 through 2014")
	plt.savefig("plotted_weights.png")
	plt.clf()
## Do 2018 work
#treat season data
# df = pd.read_csv('data/RegularSeasonDetailedResults_Prelim2018.csv')
# df = df.loc[df["Season"] == 2018]
# df.reset_index(drop=True, inplace=True)
# data, cls_ = treat_data(df)
# data_filename = "data/treated2018.npy"
# cls_filename = "data/treated2018Cls.npy"
# np.save(data_filename, data)
# np.save(cls_filename, cls_)

#compute 2018 weights
# get_weights_for_all_years([2018])
# w = get_weights_for_all_years([x for x in range(2003, 2019)])
# print w
# validate_on_years([x for x in range(2003, 2018)])
# graph_weights([x for x in range(2003, 2019)])
# w_av = compute_average([x for x in range(2003, 2019)])
# graph_average(w_av)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

def make_sub():
	w = compute_average([x for x in range(2003, 2019)])
	season, avTable = get_average_stats(2018)
	df_sample_sub = pd.read_csv('data/SampleSubmissionStage2.csv')
	matchups = np.zeros((2278, 14))
	for ii, row in df_sample_sub.iterrows():
		year, t1, t2 = get_year_t1_t2(row.ID)
		id_1 = t1 - 1000 
		mat_1 = avTable[id_1]
		id_2 = t2 - 1000
		mat_2 = avTable[id_2]
		diff_mat = mat_1 - mat_2
		matchups[ii] += diff_mat

	classification = []
	for i in xrange(len(matchups)):
		cls_ = np.asarray(func_s(np.dot(np.asmatrix(matchups[i]), w)))[0][0]
		classification.append(cls_)
	classification = np.array(classification)
	df_sample_sub.Pred = classification

	print df_sample_sub.head()
	df_sample_sub.to_csv('logreg_detailedres_diff.csv', index=False)
	
make_sub()
