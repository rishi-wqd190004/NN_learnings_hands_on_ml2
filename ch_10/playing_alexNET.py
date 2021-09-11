import tensorflow as tf
from tensorflow import keras
import os, time
import matplotlib.pyplot as plt
import numpy as np


(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'bird', 'cat', 'dog', 'frog', 'automobile', 'deer', 'horse', 'ship', 'truck']

valid_images, valid_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

# putting into tf.data.dataset API to create input pipeline for AlexNET
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))

# preprocessing and visualising dataset
plt.figure(figsize=(16,8))
for i, (image, label) in enumerate(train_ds.take(5)):
    ax = plt.subplot(5,5,i+1)
    plt.imshow(image)
    plt.title(class_names[label.numpy()[0]])
    plt.axis('off')
#plt.show()
## normalize and standardize the images
## then resize the image from 32x32 to 227x227 as AlexNet wants in 227x227 size
def preprocess_image(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227,227))
    return image, label

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
valid_ds_size = tf.data.experimental.cardinality(valid_ds).numpy()
print("size of training dataset", train_ds_size)
print("size of testinging dataset", test_ds_size)
print("size of validating dataset", valid_ds_size)

# conducting 3 primary steps: preprocess the data, shuffle the data and last batch data with in dataset
train_ds = (train_ds
                    .map(preprocess_image)
                    .shuffle(buffer_size=train_ds_size)
                    .batch(batch_size=32, drop_remainder=True))

test_ds = (train_ds
                    .map(preprocess_image)
                    .shuffle(buffer_size=test_ds_size)
                    .batch(batch_size=32, drop_remainder=True))
                    
valid_ds = (train_ds
                    .map(preprocess_image)
                    .shuffle(buffer_size=valid_ds_size)
                    .batch(batch_size=32, drop_remainder=True))

# model implementation
model = keras.
                    