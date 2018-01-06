
"""
Reads jokes from the .csv file and trains a character-level deep LSTM network
"""

import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import sys

# Filename of saved data. Download from:
# https://www.kaggle.com/abhinavmoudgil95/short-jokes
jokesFileName = 'shortjokesToy.csv'

# Read through the joke corpus once
len_text = 0
text = ''

print('Reading joke corpus')

with open(jokesFileName, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\n')
    for row in spamreader:
        tmp = ','.join(row[0].split(",")[1:])
        text = text + ' ' + tmp[1:len(tmp)-1] + '\n'
        
text = text.lower()
        
len_text = len(text)
print('Finished Reading') 
print('Corpus length:', len_text)

# Find unique characters
chars = sorted(list(set(text)))

# Number of unique characters. Length for One-hot encoding
num_chars = len(chars)
  
print('Number of characters:', num_chars)

# Create 2 dictionaries for character->index and index->character mappings
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Parameters for creating training data
max_train = 1500 # Maximum number of input examples
maxlen = 50 # Length of input sequence to LSTM
step = 3 # Characters skipped to generate next training example

num_jokes = 0 # Number of jokes of appropriate length
num_examples = 0 # Number of input strings
joke_string = [] # List of input strings for training
joke_start = [] # List of strings to start jokes
joke_char_next = [] # List of output characters

print('Generating training data')
with open(jokesFileName, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\n')
    for row in spamreader:
        # Read only upto max_train number of examples
        if num_examples > max_train:
            break
        # Remove the joke index and add '\n' to end joke
        joke = ','.join(row[0].split(",")[1:]) 
        joke = joke[1:len(joke)-1] + '\n'
        # Convert joke to lower case
        joke = joke.lower()
        # Only retain jokes longer than maxlen
        if len(joke) >= maxlen:
            num_jokes += 1
            # Save starting string of every joke
            joke_start.append(joke[0:maxlen])
            # Create dataset of training strings
            for joke_char_ind in range(0,len(joke)-maxlen,step):
                joke_string.append(joke[joke_char_ind:joke_char_ind+maxlen])
                joke_char_next.append(joke[joke_char_ind+maxlen])
                num_examples += 1     

print('Number of jokes:', num_jokes)
print('Number of training examples:', num_examples)
      
# Generate one-hot coded training data

x = np.zeros((num_examples, maxlen, num_chars))
y = np.zeros((num_examples, num_chars))
            
for i in range(num_examples):
    y[i,char_indices[joke_char_next[i]]] = 1
    for j in range(maxlen):
        x[i,j,char_indices[joke_string[i][j]]] = 1  
print('Finished generating training data')
             
# Shuffle data
indices = np.random.permutation(num_examples)
x = x[indices,:,:]
y = y[indices,:]

# Split into Training and Validation data
frac_train = 0.95 

x_valid = x[int(frac_train*num_examples):,:,:]
y_valid = y[int(frac_train*num_examples):,:]
x = x[:int(frac_train*num_examples),:,:]
y = y[:int(frac_train*num_examples),:]
             
# Define deep LSTM model
print('Build model...')
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(maxlen, num_chars),
               dropout=0.5))
model.add(LSTM(256, return_sequences=True, dropout=0.5))
model.add(LSTM(256, dropout=0.5))
model.add(Dense(num_chars))
model.add(Activation('softmax'))

# Compile model
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Train the network

batch_size = 128
iterations = 20
max_out_len = 400 # maximum length of the joke outputted

train_loss = np.zeros((iterations,1))
val_loss = np.zeros((iterations,1))

eps = np.finfo(float).eps # Define a small epsilon value

# Save model if validation error decreases
modelsave = ModelCheckpoint(filepath='jokeModel.h5', save_best_only=True,
                            verbose=1)

for iteration in range(1, iterations):
    print('\n')
    print('-' * 50)
    print('Iteration', iteration)
    hist = model.fit(x, y, batch_size=batch_size, epochs=1, verbose=2,
                     validation_data=(x_valid, y_valid), callbacks=[modelsave])
    
    # Save training and validation losses every epoch
    val_loss[iteration-1] = hist.history['val_loss'][0]
    train_loss[iteration-1] = hist.history['loss'][0]
    np.save('jokesTrainErr',train_loss[:iteration])
    np.save('jokesValidErr',val_loss[:iteration])
    
    # Generate jokes for different diversity values after every epoch
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('\n')
        print('----- diversity:', diversity)
        
        # Start joke with a random string
        randStart = np.random.randint(0,num_jokes)
        generated = joke_start[randStart][0:maxlen]
        print('----- Generating with seed: "' + generated + '"')
        sys.stdout.write(generated)
        # Generate one-hot encoding for the starting string
        x_pred = np.zeros((1,maxlen,num_chars))
        for i in range(maxlen):
            x_pred[0,i,char_indices[generated[i]]] = 1
        
        # Generate output string one character at a time
        # Restrict output length to max_out_len characters
        for i in range(max_out_len):
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds + eps, diversity)
            next_char = indices_char[next_index]
            # If '\n' character is predicted, end joke
            if next_char == '\n':
                break
            
            generated += next_char
            
            sys.stdout.write(next_char)
            sys.stdout.flush()
            
            x_pred[0,0:maxlen-1,:] = x_pred[0,1:maxlen,:]
            x_pred[0,maxlen-1,:] = 0
            x_pred[0,maxlen-1,next_index] = 1   
            
print('\n')
print('-' * 50)
print('Finished Training.')
            
            