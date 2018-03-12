## Logistic Gradient Descent

# Idea 1:

Using a feature to indicate whether the game is a championship game, we want to use the dataset operating solely on games within a year.
We will hold 20% of the games of every year as a validation set and start with no training set.
We want to perform mini batch gradient descent using logistic regression with both l1 and l2 normalization for comparison.
We want to first graph the training set results and validation set accuracy for every year.
Then we want to train on all the years and validate on 20% of each year, first using the same weights, then averaging our weights.

# Idea 2:

We want to build a Neural Net and follow a procedure similar to above. 
Train and validate the neural net on every year independently. 
Then train the neural on all the games, validating on 20% of each season. 
Then train the neural on every year independently, average the weights and validate on 20% of each year. 

