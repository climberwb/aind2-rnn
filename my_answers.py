import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import re


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(len(series[:-window_size])):
        X.append(series[i:i+window_size])
    y = series[window_size:]
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # as the first layer in a Sequential model
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model
    

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
  
    punctuation = ['!', ',', '.', ':', ';', '?']
    text = text.lower()
    
    text = re.sub(r"[^a-z0-9%s]"%(''.join(punctuation)), " ", text)
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    series = text.split()
    step=0
    for i in range(len(series[:-window_size])):
        step = step_size+i
        inputs.append(series[step:step+window_size])
        if step >= len(series[:-window_size]):
            break
    #print(inputs)
    #outputs = np.asarray(outputs)
    #inputs = np.asarray(inputs)
    print(outputs)
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
