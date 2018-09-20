#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:14:08 2018
@author: Michael Samon
Meme Identifier in Keras
"""


import os
import shutil
import keras
from keras.preprocessing.image import ImageDataGenerator
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
import datetime




basedir = "/Users/user/Desktop/Keras/MemeFinder/"
#meme_src = os.path.join(basedir, "memes_orig/")
#lame_src = os.path.join(basedir, "lames_orig/")

train_dir = os.path.join(basedir, "train/")
test_dir = os.path.join(basedir, "test/")
val_dir = os.path.join(basedir, "validation/")


# seperate meme images into proper folders
#meme_fnames = os.listdir(basedir+"memes_orig/")
#lame_fnames = os.listdir(basedir+"lames_orig/")[0:3327]




# one time initial setup 
#for fname in lame_fnames:
#    src = lame_src + fname
#    dst = os.path.join(basedir, "lames_orig/", fname)
#    shutil.copyfile(src, dst)

def copy_files():

    for fname in lame_fnames[0:2326]:
        src = lame_src + fname
        dst = os.path.join(train_dir, "lames/", fname)
        shutil.copyfile(src, dst)
        
    for fname in lame_fnames[2326:2826]:
        src = lame_src + fname
        dst = os.path.join(test_dir, "lames/", fname)
        shutil.copyfile(src, dst)
        
    for fname in lame_fnames[2826:]:
        src = lame_src + fname
        dst = os.path.join(val_dir, "lames/", fname)
        shutil.copyfile(src, dst)
    
    
    for fname in meme_fnames[0:2326]:
        src = meme_src + fname
        dst = os.path.join(train_dir, "memes/", fname)
        shutil.copyfile(src, dst)
    
    for fname in meme_fnames[2326:2826]:
        src = meme_src + fname
        dst = os.path.join(test_dir, "memes/", fname)
        shutil.copyfile(src, dst)
        
    for fname in meme_fnames[2826:]:
        src = meme_src + fname
        dst = os.path.join(val_dir, "memes/", fname)
        shutil.copyfile(src, dst)


# read in meme images 
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=50,
        class_mode='binary'
        )

val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(150,150),
        batch_size=50,
        class_mode='binary'
        )

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150,150),
        batch_size=50,
        class_mode='binary'
        )





from keras import layers, models
epochs = 1

def img_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid')) 
    
    model.summary()
    
    model.compile(loss="binary_crossentropy", 
                  optimizer=keras.optimizers.Adam(lr=0.003),
                  metrics=['acc']
                  )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=3),
        keras.callbacks.ModelCheckpoint('./' + str(datetime.datetime.now()),
                                        monitor='val_acc'),
        keras.callbacks.TensorBoard(
                histogram_freq=0)
                ]
    
    model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=epochs,
      validation_data=val_generator,
      validation_steps=50,
      callbacks=callbacks)
       
#    loss = history.history['loss']
#    val_loss = history.history['val_loss']

    
def inception():
    from keras.applications import inception_resnet_v2
    base = inception_resnet_v2.InceptionResNetV2(include_top=False, input_shape=(150,150,3), pooling='max')
    model = models.Sequential()
    model.add(base)
    model.add(layers.Dense(700, activation='tanh'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.summary()
    
        
    model.compile(loss="binary_crossentropy", 
                  optimizer=keras.optimizers.Adam(lr=0.003),
                  metrics=['acc']
                  )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3),
        keras.callbacks.ModelCheckpoint('./chollet_crap.h5',
                                        monitor='val_acc'),
        keras.callbacks.TensorBoard(
                histogram_freq=0)
                ]
    
    model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=epochs,
      validation_data=val_generator,
      validation_steps=50,
      callbacks=callbacks)
    
    
  
    
def get_image():
    x,y = val_generator.next()
    i = random.randint(0,50)
    image = x[i]
    label = y[i]
    return image, label



# model = keras.models.load_model("chollet.h5")
def predict_meme(model, path=None):
    pred = list()
    
    if path:
        img = mpimg.imread(path)
        resized = resize(img, (150, 150), anti_aliasing=True)
        pred = model.predict(resized.reshape(1,150,150,3))[0]
        
    else:    
        img, _ = get_image()    
        pred = model.predict(img.reshape(1,150,150,3))[0]
    
    plt.imshow(img)
    plt.show()
    
    print "\n\nNeural Net predicted:", ('MEME' if pred >= 0.5 else 'LAME PIC') 
    print "Confidence level from 0 to 1 with 1 being a meme:", pred[0]
     
    

    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


















