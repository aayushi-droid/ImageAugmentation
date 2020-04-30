# -*- coding: utf-8 -*-
__project__ = 'ImageDataGenerator'

from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from numpy import expand_dims
from matplotlib import pyplot

img = load_img('bird.jpg')

data = img_to_array(img)

print(data.shape)

# expand dimension to one sample
samples= expand_dims(data,0)

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

it = datagen.flow(samples,batch_size=1)

for i in range(9):
  pyplot.subplot(330 + 1 + i)
  batch = it.next()
  image = batch[0].astype('uint8')
  pyplot.imshow(image)
pyplot.show()

