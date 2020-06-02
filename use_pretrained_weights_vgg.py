# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:58:40 2020

@author: msadi
"""


from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model, model_from_json
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

# import keras.backend as K
# import tensorflow as tf

# config = tf.compat.v1.ConfigProto(device_count={"CPU": 6})
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



# #ignore warnings in the output
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# from tensorflow.python.client import device_lib

# # Check all available devices if GPU is available
# print(device_lib.list_local_devices())
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


IMAGE_SIZE = [100,100]

epochs = 5
batch_size = 32
train_path = 'large_files/fruits-360-small/Training'
valid_path = 'large_files/fruits-360-small/validation'

image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

folders = glob(train_path + '/*')

plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()

vgg = VGG16(input_shape = IMAGE_SIZE + [3],weights = 'imagenet',include_top=False)

for layer in vgg.layers:
    layer.trainable = False
    
x=Flatten()(vgg.output)
#x=Dense(1000,activation='relu')(x)
prediction = Dense(len(folders),activation = 'softmax')(x)

model = Model(inputs=vgg.input,outputs = prediction)

model.summary()

model.compile(loss='categorical_crossentropy',optimizer = 'rmsprop',metrics=['accuracy'])

gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input)

test_gen = gen.flow_from_directory(valid_path,target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k,v in test_gen.class_indices.items():
    labels[v]=k

for x,y in test_gen:
    print("min:",x[0].min(),"max:",x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break

train_generator = gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size)

valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size)

r=model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(image_files)//batch_size,
    validation_steps = len(valid_image_files)//batch_size,
    use_multiprocessing=False)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")


def get_confusion_matrix(data_path,N):
    print("Generation confusion matrix",N)
    predictions=[]
    targets=[]
    i=0
    for x,y in gen.flow_from_directory(data_path,target_size=IMAGE_SIZE,shuffle=False,batch_size=batch_size*2):
        i+=1
        if i%50==0:
            print(i)
        p=loaded_model.predict(x)
        p=np.argmax(p,axis=1)
        y=np.argmax(y,axis=1)
        predictions=np.concatenate((predictions,p))
        targets=np.concatenate((targets,y))
        if len(targets)>=N:
            break
    cm = confusion_matrix(targets,predictions)
    return cm

cm= get_confusion_matrix(train_path,len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path,len(valid_image_files))
print(valid_cm)

from util import plot_confusion_matrix

plot_confusion_matrix(cm,labels,title='Train confusion matrix')
plot_confusion_matrix(valid_cm,labels,title='validation confusion matrix')