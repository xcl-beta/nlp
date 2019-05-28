
#%%
# coonfigure gpu memory
import keras 
from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))



from keras.datasets import imdb


#%%



#%%
max_features = 10000
maxlen = 20


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)


#%%
len(train_data[0]), train_labels[0], len(train_data), len(train_labels)


#%%
len(train_data[0] )


#%%
max([max(sequence) for sequence in train_data])


#%%
# decode vector to english words

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)


#%%
# one hot encoding
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
   results = np.zeros((len(sequences), dimension))
   for i, sequence in enumerate(sequences):
       results[i, sequence] = 1.
   return results    


#%%
x_test.shape


#%%
x_train = vectorize_sequences(train_data)
x_test  = vectorize_sequences(test_data)


#%%
y_train = np.asarray(train_labels).astype('float32')
y_test  = np.asarray(test_labels).astype('float32')


#%%
from keras import models
from keras import layers

#reset_graph() # cannot used
K.clear_session()
model = models.Sequential()
model.add(layers.Dense(128, activation="tanh", input_shape=(10000,)))
model.add(layers.Dense(128, activation="tanh"))
#model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1,  activation="sigmoid"))
model.summary()


#%%
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])


#%%
# split data to tune best epochs

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


#%%
callbacks_list = [
    keras.callbacks.EarlyStopping(
    monitor="acc",
    patience=1, ),
    keras.callbacks.ModelCheckpoint(
        filepath="my_model.h5",
        monitor="val_loss",
        save_best_only=True,)     
]


#%%
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    callbacks=callbacks_list,
                    validation_data=(x_val, y_val))


#%%
history_dict = history.history
history_dict.keys()


#%%
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


#%%
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label = "Training acc")
plt.plot(epochs, val_acc_values, 'b', label = 'Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

plt.show()


#%%
# fit less 
import keras

callbacks_list = [
    keras.callbacks.EarlyStopping(
    monitor="acc",
    patience=1, ),
    
    keras.callbacks.ModelCheckpoint(
        filepath="my_model.h5",
        monitor="val_loss",
        save_best_only=True,)     
]


K.clear_session()
model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1,  activation="sigmoid"))

model.compile(optimizer = 'rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])


model.fit(x_train,
                    y_train,
                    epochs=4,
                    batch_size=512,
                    callbacks=callbacks_list,
                    validation_data=(x_val, y_val))


#%%
results = model.evaluate(x_test,y_test)
results


#%%
model.predict(x_test)


#%%
max_features = 10000


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)


#%%
x_train.shape,len(y_test), len(x_train[0]) #, x_train[0], y_test 


#%%



#%%
max_features = 10000
maxlen = 100

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

## Using embedding


#%%



#%%

# processing with Embedding layer 

from keras import preprocessing 
# convert list to array, only keep the last maxlen numbers, truncate the begining sequence
x_train = preprocessing.sequence.pad_sequences(train_data, maxlen = maxlen)

x_test = preprocessing.sequence.pad_sequences(test_data, maxlen = maxlen)


#%%
x_train = np.asarray(x_train).astype('float32')
x_test  = np.asarray(x_test).astype('float32')


#%%
x_train.shape,len(y_test), len(x_train[0]), x_train[2], y_test 


#%%
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Embedding

K.clear_session()

model = Sequential()
model.add(Embedding(10000,8,input_length=maxlen, name = "features")) # train data cannot be one hot

model.add(Flatten())

model.add(Dense(1, activation = "sigmoid"))
model.compile(optimizer="rmsprop", loss = "binary_crossentropy", metrics=["acc"])
model.summary()


#%%
# !mkdir my_log_dir

callbacks = [ 
    keras.callbacks.TensorBoard(
        log_dir = "my_log_dir", 
        histogram_freq = 1,
        embeddings_freq = 1,
        embeddings_layer_names=['features'],
        embeddings_data = x_train,
        )]


#%%
x_test.shape, x_test.dtype, x_train.shape


#%%
history = model.fit(x_train, 
                   y_train,
                   epochs=10,
                   batch_size=32,
                   callbacks = callbacks,
                   #validation_split=0.2)
                    validation_data=(x_test, y_test))

#%% [markdown]
# # download the raw IMDB data

#%%


import os

imdb_dir = '/home/lixiaochuan/Downloads/kaggle_data/aclImdb/aclImdb/'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
        
        if label_type == 'neg':
            labels.append(0)
        else:
            labels.append(1)


#%%
len(texts), len(labels)


#%%
#list(word_index.keys())[:10], list(word_index.values())[:10]


#%%
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


maxlen = 200
training_samples = 10000 #10000
validation_samples = 5000 #10000
max_words = 10000
embedding_dim = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index)) # more than specified max_words, how the words not in the top max words represented?
data = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


#%%
#len(set([value for key, value in word_index.items()])) # total number of words 
# check the max_len effect 

len(sequences[0]),len(list(set(sequences[0])))
output = set([])

for sequence in sequences:
    output = output | set(sequence) 
    
    
    
len(list(output))

#%% [markdown]
#  
# 
# # use precomputed GLOVE word embedding

#%%
# use precomputed GLOVE word embedding

glove_dir = '/home/lixiaochuan/Downloads/kaggle_data/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

print(f)

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()
    
print('Found %s word vectors.' % len(embeddings_index))


#%%
len(coefs), word, coefs


#%%
# create embedding matrix
embedding_dim = 100 # should match with downloaded glove embedding size
embedding_matrix = np.zeros((max_words, embedding_dim)) # be careful of the dimension

# build the top 10000 (max_words) embedding matrix
for word, i in word_index.items(): # word_index is ordered by frequency, 
    if i < max_words:
        embedding_vector = embeddings_index.get(word) # embeddings_index: saved 400k glove embedding
        
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


#%%
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

K.clear_session()

model5 = Sequential()
model5.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model5.add(Flatten())
model5.add(Dense(32, activation='relu'))
model5.add(Dense(1, activation='sigmoid'))
model5.summary() 


#%%

# what is the use of this? set the embedding
model5.layers[0].set_weights([embedding_matrix])
model5.layers[0].trainable = False


#%%
#!mkdir log_glove_embedding


#%%


callbacks_g = [ 
    keras.callbacks.TensorBoard(
        log_dir = "log_glove_embedding",
        histogram_freq = 1,
        embeddintenogs_freq = 1,
        embeddings_layer_names=['embedding_1'],
        embeddings_data = x_train,
        )]


#%%
model5.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['acc'])
history = model5.fit(x_train, y_train, epochs=10, batch_size=32, 
                    #callbacks =callbacks_g,
                    validation_data=(x_val, y_val))


#%%
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%%
# use functional API
from keras import Input, layers
from keras.models import Model

K.clear_session()

input_tensor = Input(shape=(maxlen,))
x = layers.Embedding(output_dim=embedding_dim, input_dim=max_words, input_length=maxlen)(input_tensor) # embedding dimension is reversed
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(1, activation = 'sigmoid')(x)

model6 = Model(input_tensor, output_tensor)
model6.summary()



#%%
model6.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['acc'])
history = model6.fit(x_train, y_train, epochs=10, batch_size=32, 
                    #callbacks =callbacks_g,
                    validation_data=(x_val, y_val))


#%%
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%%
# Training the same model without pretrained word embeddings

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

K.clear_session()
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=10,  batch_size=32, validation_data=(x_val, y_val))


#%%
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%%
#Listing 6.17 Tokenizing the data of the test set
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)

    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())            
            f.close()

        if label_type == 'neg':
            labels.append(0)
        else:
            labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
        
x_test = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
y_test = np.asarray(labels)


#%%
x_test.shape, y_test.shape


#%%
model5.evaluate(x_test, y_test)


#%%
# simple RNN

from keras.layers import SimpleRNN
batch_size = 32


#%%
max_words, embedding_dim,  maxlen, batch_size


#%%
K.clear_session()

input_tensor = Input(shape=(maxlen,))
x = layers.Embedding(output_dim=embedding_dim, input_dim=max_words, input_length=maxlen)(input_tensor) # embedding dimension is reversed
#x = layers.Flatten()(x)
x = layers.SimpleRNN(units=batch_size, recurrent_dropout=0.0, stateful = False)(x)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(1, activation = 'sigmoid')(x)

model8a = Model(input_tensor, output_tensor)
model8a.summary()


#%%
K.clear_session()
model8 = Sequential()
model8.add(Embedding(max_words, embedding_dim))
model8.add(SimpleRNN(units=batch_size, recurrent_dropout=0.0, stateful = False)) 
# drop out reduce train data performance , but no improvement on validation dataset
model8.add(Dense(32, activation='relu')) # add another dense layer, no help
model8.add(Dense(1, activation='sigmoid'))          
model8.summary()

# basic Simple RNN is not working


#%%
#  set the pretrained embedding
model8.layers[0].set_weights([embedding_matrix])
model8.layers[0].trainable = False


#%%
model8a.compile(optimizer="rmsprop", loss = "binary_crossentropy", metrics = ["acc"])

# does batch size here should be consistant with the batch size of simpleRNN? here it is for gradient update
history = model8a.fit(x_train, y_train, epochs = 10, batch_size =128, validation_data=(x_val, y_val))


#%%
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%%
# use pretrained embedding

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False


#%%
model.compile(optimizer="rmsprop", loss = "binary_crossentropy", metrics = ["acc"])

history = model.fit(x_train, y_train, epochs = 10, batch_size =128, validation_data=(x_val, y_val))


#%%
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%%
# LSTM model 

from keras.layers import LSTM


#%%



#%%
model9 = Sequential()
model9.add(Embedding(max_words, embedding_dim))
model9.add(LSTM(batch_size))
model9.add(Dense(1, activation='sigmoid'))          
model9.summary()


#%%
model9.compile(optimizer="rmsprop", loss = "binary_crossentropy", metrics = ["acc"])

history = model9.fit(x_train, y_train, epochs = 10, batch_size =128, validation_data=(x_val, y_val))


#%%
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%%
# use pretrained embedding

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.summary()


#%%
model.compile(optimizer="rmsprop", loss = "binary_crossentropy", metrics = ["acc"])

history = model.fit(x_train, y_train, epochs = 10, batch_size =128, validation_data=(x_val, y_val))


#%%
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%%



#%%
# bidirectional
from keras import layers

model = Sequential()
model.add(Embedding(max_words, embedding_dim))
model.add(layers.Bidirectional(layers.LSTM(batch_size)))
model.add(Dense(1, activation='sigmoid'))   
 


#%%
model.compile(optimizer="rmsprop", loss = "binary_crossentropy", metrics = ["acc"])

history = model.fit(x_train, y_train, epochs = 10, batch_size =128, validation_data=(x_val, y_val))


#%%
# much better 

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%%
#Training and evaluating a simple 1D convnet on the IMDB data
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
  
model = Sequential()
model.add(layers.Embedding(max_words, embedding_dim, input_length= maxlen ))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()


#%%
# this cannot be run multiple times continues
# need to run the above cell to initiate the model first

# sometimes the above cell needs to run several times to be initialized 
model.compile(optimizer=RMSprop(lr=1e-4), 
                loss='binary_crossentropy',
                metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))


#%%
# fast and better 
# embedding dim is important

# peak at 10 epochs and acc drops after that  

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%%
model.evaluate(x_test, y_test)


#%%
# use keras callbackfunctions 
   
   # early stop
batch_size = 32    
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense




   
model = Sequential()
model.add(Embedding(max_words, embedding_dim))
model.add(LSTM(batch_size))
model.add(Dense(1, activation='sigmoid'))          
model.summary()
   


#%%
import keras


callbacks_list = [ 
    keras.callbacks.EarlyStopping(
        monitor="acc",
        patience = 1,),
    keras.callbacks.ModelCheckpoint(
        filepath="my_model.h5",
        monitor="val_loss",
        save_best_only=True,)]

model.compile(optimizer="rmsprop", loss = "binary_crossentropy", metrics = ["acc"])

history = model.fit(x_train, y_train, epochs = 10, batch_size =128, validation_data=(x_val, y_val))


#%%
callbacks = [keras.callbacks.TensorBoard(log_dir = "/home/lixiaochuan/Documents/mdl/conda/my_log_dir/",
                                         histogram_freq=1,
                                         embeddings_freq=1,)]

history = model.fit(x_train, y_train, epochs = 10, batch_size =128, validation_data=(x_val, y_val))


