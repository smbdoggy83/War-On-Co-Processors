# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:10:23 2023

@author: Seth
"""

from keras.models import Sequential
from keras.layers import Dense


input_dim = 2
num_classes = 2

model = Sequential()
model.add(Dense(8, input_dim=input_dim, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
