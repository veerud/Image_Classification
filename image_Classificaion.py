#importing libraries
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Dense,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt


#getting data
img_size = 150
train_data_dir = 'cats_and_dogs_filtered/train'
validation_data_dir = 'cats_and_dogs_filtered/validation'
nb_train_samples = 1000
nb_validation_samples = 100
epochs = 50
batch_size = 32

if k.image_data_format()=='channels_first':
    input_shape = (3,img_size,img_size)
else:
    input_shape = (img_size,img_size,3)

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,target_size = (img_size,img_size),batch_size=batch_size,class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,target_size = (img_size,img_size),batch_size=batch_size,class_mode='binary')

#models

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer = 'rmsprop',metrics = ['accuracy'])

history=model.fit(train_generator,steps_per_epoch=nb_train_samples//batch_size,epochs = epochs,
                    validation_data=validation_generator,validation_steps = nb_validation_samples//batch_size)
model.save_weights('first_try.h5')
print(history)

#predicting the image

img_pred = image.load_img('cats_and_dogs_filtered/validation/dogs/dog.2001.jpg',target_size=(150,150))
plt.imshow(img_pred)
plt.show()
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred,axis = 0)

#exexute

result = model.predict(img_pred)
print(result)
if result[0][0] == 1:
    prediction = "The input image is a DOG "
else:
    prediction= "The input image is a CAT "
print(prediction)


#accuracy graph plotting


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation '
                                       'Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()