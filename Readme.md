## March Madness Training

#Idea 1:

Using a feature to indicate weather or not the game is a championship game, we want to use the dataset operating solely on games within a year.
We will hold 20% of the games of every year as a validation set [Start with no training set]
We want to preform mini batch gradient descent using logistic regression with both l1 and l2 normalization [for comparison]
We want to first graph the training set resullts and validtion set accuracy for every year
Then we want to train on all the years and validate on 20% of each year [First using the same weights, then averaging our weights]

#Idea 2:

We want to build a Neural Net and follow a procedure similar to above. 
Train and validate the neural net on every year independenlty. 
Then train the neural on all the games, validating on 20% of each season. 
Then train the neural on everyyear independenlty, average the weights and validate on 20% of each year. 


#BrainStorm:

We are given two team ids in our data, a winning team Id and a losing Team id
We want to train our algorithm such that it outputs a number between zero and 1 that gives us the two teams 
This means that we need to restructure our data a little 
Since in our data the winning team is written as such, we can just write the output as 1 for every game
