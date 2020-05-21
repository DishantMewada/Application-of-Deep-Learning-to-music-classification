import os

import time

import matplotlib.pyplot as plt

#############################
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

#################################
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import SGD, Adagrad, Adadelta, Adamax, Nadam
from FeatureExtraction import FeatureExtraction  # local python class with Audio feature extraction (librosa)

# Turn off TF verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

genre_features = FeatureExtraction()
genre_features.load_preprocess_data()
# genre_features.load_deserialize_data()

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.


batch_size = 35
nb_epochs = 800

print("Training")
print("X shape: " + str(genre_features.train_X.shape))
print("Y shape: " + str(genre_features.train_Y.shape))

print("Validation")
print("X shape: " + str(genre_features.dev_X.shape))
print("Y shape: " + str(genre_features.dev_Y.shape))

print("Testing")
print("X shape: " + str(genre_features.test_X.shape))
print("Y shape: " + str(genre_features.test_X.shape))

input_shape = (genre_features.train_X.shape[1], genre_features.train_X.shape[2])
print('LSTM Model building')


model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=genre_features.train_Y.shape[1], activation='softmax'))

print("Using Optimizer SGD()")
print("Compiling the model...")
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])
model.summary()
model.save('model_SGD.h5')
model.get_weights()
model.optimizer

start_train = time.clock()

print("Training the model...")

history = model.fit(genre_features.train_X, genre_features.train_Y, batch_size=batch_size, epochs=nb_epochs, verbose = 2, validation_split = 0.2, shuffle=True)

print("Time taken to build a model: ",time.clock() - start_train,"Seconds")


print(history.history.keys()) # Displays keys from history, in my case loss,acc
plt.plot(history.history['acc']) #here I am trying to plot only accuracy, the same can be used for loss as well
plt.title('model accuracy')
plt.plot(history.history['val_loss'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("\nValidating ...")
score, accuracy = model.evaluate(genre_features.dev_X, genre_features.dev_Y, batch_size=batch_size, verbose=1)
print("Validation loss:  ", score)
print("Validation accuracy:  ", accuracy)


print("\nTesting the model...")
score, accuracy = model.evaluate(genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1)
print("Testing loss:  ", score)
print("Testting accuracy:  ", accuracy)

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(genre_features.test_X)
print(Y_pred)

import numpy as np
# y_pred = np.argmax(Y_pred, axis=1)
# print(y_pred)

#another way which directly gets the class value
y_pred = model.predict_classes(genre_features.test_X)
print(y_pred)
#y_pred = (y_pred > 0.5)

p = model.predict_proba(genre_features.test_X) #to predict probability

target_names = ['classical','folk','jazz']
print(classification_report(np.argmax(genre_features.test_Y,axis=1),y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(genre_features.test_Y,axis=1),y_pred)) #f1_score(y_true, y_pred, average='micro')
#############################

print("Using Optimizer Adagrad()")
print("Compiling the model...")
model.compile(loss='categorical_crossentropy', optimizer=Adagrad(lr=0.001), metrics=['accuracy'])
model.summary()
model.save('model_Adagrad.h5')
model.get_weights()
model.optimizer

start_train = time.clock()

print("Training the model...")

history = model.fit(genre_features.train_X, genre_features.train_Y, batch_size=batch_size, epochs=nb_epochs, verbose = 2, validation_split = 0.2, shuffle=True)

print("Time taken to build a model: ",time.clock() - start_train,"Seconds")


print(history.history.keys()) # Displays keys from history, in my case loss,acc
plt.plot(history.history['acc']) #here I am trying to plot only accuracy, the same can be used for loss as well
plt.title('model accuracy')
plt.plot(history.history['val_loss'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("\nValidating ...")
score, accuracy = model.evaluate(genre_features.dev_X, genre_features.dev_Y, batch_size=batch_size, verbose=1)
print("Validation loss:  ", score)
print("Validation accuracy:  ", accuracy)


print("\nTesting the model...")
score, accuracy = model.evaluate(genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1)
print("Testing loss:  ", score)
print("Testting accuracy:  ", accuracy)

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(genre_features.test_X)
print(Y_pred)

import numpy as np
# y_pred = np.argmax(Y_pred, axis=1)
# print(y_pred)

#another way which directly gets the class value
y_pred = model.predict_classes(genre_features.test_X)
print(y_pred)
#y_pred = (y_pred > 0.5)

p = model.predict_proba(genre_features.test_X) #to predict probability

target_names = ['classical','folk','jazz']
print(classification_report(np.argmax(genre_features.test_Y,axis=1),y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(genre_features.test_Y,axis=1),y_pred)) #f1_score(y_true, y_pred, average='micro')

####################

print("Using Optimizer Adadelta()")
print("Compiling the model...")
model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=0.001), metrics=['accuracy'])
model.summary()
model.save('model_Adadelta.h5')
model.get_weights()
model.optimizer

start_train = time.clock()

print("Training the model...")

history = model.fit(genre_features.train_X, genre_features.train_Y, batch_size=batch_size, epochs=nb_epochs, verbose = 2, validation_split = 0.2, shuffle=True)

print("Time taken to build a model: ",time.clock() - start_train,"Seconds")


print(history.history.keys()) # Displays keys from history, in my case loss,acc
plt.plot(history.history['acc']) #here I am trying to plot only accuracy, the same can be used for loss as well
plt.title('model accuracy')
plt.plot(history.history['val_loss'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("\nValidating ...")
score, accuracy = model.evaluate(genre_features.dev_X, genre_features.dev_Y, batch_size=batch_size, verbose=1)
print("Validation loss:  ", score)
print("Validation accuracy:  ", accuracy)


print("\nTesting the model...")
score, accuracy = model.evaluate(genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1)
print("Testing loss:  ", score)
print("Testting accuracy:  ", accuracy)

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(genre_features.test_X)
print(Y_pred)

import numpy as np
# y_pred = np.argmax(Y_pred, axis=1)
# print(y_pred)

#another way which directly gets the class value
y_pred = model.predict_classes(genre_features.test_X)
print(y_pred)
#y_pred = (y_pred > 0.5)

p = model.predict_proba(genre_features.test_X) #to predict probability

target_names = ['classical','folk','jazz']
print(classification_report(np.argmax(genre_features.test_Y,axis=1),y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(genre_features.test_Y,axis=1),y_pred)) #f1_score(y_true, y_pred, average='micro')
##########

print("Using Optimizer Adamax()")
print("Compiling the model...")
model.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=0.001), metrics=['accuracy'])
model.summary()
model.save('model_Adamax.h5')
model.get_weights()
model.optimizer

start_train = time.clock()

print("Training the model...")

history = model.fit(genre_features.train_X, genre_features.train_Y, batch_size=batch_size, epochs=nb_epochs, verbose = 2, validation_split = 0.2, shuffle=True)

print("Time taken to build a model: ",time.clock() - start_train,"Seconds")


print(history.history.keys()) # Displays keys from history, in my case loss,acc
plt.plot(history.history['acc']) #here I am trying to plot only accuracy, the same can be used for loss as well
plt.title('model accuracy')
plt.plot(history.history['val_loss'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("\nValidating ...")
score, accuracy = model.evaluate(genre_features.dev_X, genre_features.dev_Y, batch_size=batch_size, verbose=1)
print("Validation loss:  ", score)
print("Validation accuracy:  ", accuracy)


print("\nTesting the model...")
score, accuracy = model.evaluate(genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1)
print("Testing loss:  ", score)
print("Testting accuracy:  ", accuracy)

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(genre_features.test_X)
print(Y_pred)

import numpy as np
# y_pred = np.argmax(Y_pred, axis=1)
# print(y_pred)

#another way which directly gets the class value
y_pred = model.predict_classes(genre_features.test_X)
print(y_pred)
#y_pred = (y_pred > 0.5)

p = model.predict_proba(genre_features.test_X) #to predict probability

target_names = ['classical','folk','jazz']
print(classification_report(np.argmax(genre_features.test_Y,axis=1),y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(genre_features.test_Y,axis=1),y_pred)) #f1_score(y_true, y_pred, average='micro')
##########

print("Using Optimizer Nadam()")
print("Compiling the model...")
model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=0.001), metrics=['accuracy'])
model.summary()
model.save('model_Nadam.h5')
model.get_weights()
model.optimizer

start_train = time.clock()

print("Training the model...")

history = model.fit(genre_features.train_X, genre_features.train_Y, batch_size=batch_size, epochs=nb_epochs, verbose = 2, validation_split = 0.2, shuffle=True)

print("Time taken to build a model: ",time.clock() - start_train,"Seconds")


print(history.history.keys()) # Displays keys from history, in my case loss,acc
plt.plot(history.history['acc']) #here I am trying to plot only accuracy, the same can be used for loss as well
plt.title('model accuracy')
plt.plot(history.history['val_loss'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("\nValidating ...")
score, accuracy = model.evaluate(genre_features.dev_X, genre_features.dev_Y, batch_size=batch_size, verbose=1)
print("Validation loss:  ", score)
print("Validation accuracy:  ", accuracy)


print("\nTesting the model...")
score, accuracy = model.evaluate(genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1)
print("Testing loss:  ", score)
print("Testting accuracy:  ", accuracy)

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(genre_features.test_X)
print(Y_pred)

import numpy as np
# y_pred = np.argmax(Y_pred, axis=1)
# print(y_pred)

#another way which directly gets the class value
y_pred = model.predict_classes(genre_features.test_X)
print(y_pred)
#y_pred = (y_pred > 0.5)

p = model.predict_proba(genre_features.test_X) #to predict probability

target_names = ['classical','folk','jazz']
print(classification_report(np.argmax(genre_features.test_Y,axis=1),y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(genre_features.test_Y,axis=1),y_pred)) #f1_score(y_true, y_pred, average='micro')
##########
