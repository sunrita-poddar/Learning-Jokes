# Learning-Jokes

This project aims to learn a language model from a joke corpus using a deep LSTM network. The problem appears in Open AI's list of fun problems in Deep Learning: https://openai.com/requests-for-research/

Abhinav Moudgil's blog post had many useful pointers which were helpful in getting started:
https://amoudgl.github.io/2017-03-01/funnybot/

This is work in progress, and I will keep updating the code to improve performance. Find below a a list of things yet to be done.

# Overview

A 3-layer deep LSTM network is trained on a sub-set of the joke dataset. A dictionary is generated to map each character in the corpus to an integer. Overlapping strings of length 50 characters are extracted from each joke, to train the network to predict the next character. Using the dictionary, all training data is one-hot encoded. After training, the network is to be fed a starting string and the joke is completed character-by-character. When the network predicts a '\n' character or the joke length exceeds the maximum length, the joke is complete.

The work is still in progress. Though the code is yet to generate very funny coherent jokes, there are a number of interesting and encouraging trends:
1. In the initial iterations the network learns the concepts of words separated by spaces. However, most "words" dont make sense.
2. Next, more "real" words start to appear.
3. With further iterations, the network learns to end jokes with the right punctuations. Usually '.' or '!' is followed by the '\n' character which signifies the end of the joke.
4. Further down, the network starts to put the right words together: "I asked her...." will be followed by "she said...".

# Code
 
## trainNetwork. py

Prepares the training and validation data from the .csv file. After every training epoch, a starting seed string is fed as input and characters are predicted one after the other. If the validation error decreases, then the model is saved. The training and validations errors are saved as numpy arrays and updated every epoch.

# Data

## shortJokes.csv

Collection of around 200,000 short jokes from Kaggle:
https://www.kaggle.com/abhinavmoudgil95/short-jokes

# Things to do

1. Make code more memory efficient. Currently the large one-hot encoded training data is loaded in memory.
2. Enable network to take variable length sequences as input.
3. Using above, enable the trained network to only take a random character as starting input, instead of a seed string.
4. Train for capital letters. Currently all text is converted to lower case.
5. Further study effect of hyper-parameters.
 
