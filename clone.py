import csv
import cv2
import numpy as np

lines = []
with open ('../data/driving_log.csv') as cvsfile:
	reader = csv.reader(cvsfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines[1:]:
	source_path =  line[0]
	filename = source_path.split('/')[-1]
	current_path = '../data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)

	steering_center = float(line[3])
	measurement = steering_center 
	measurements.append(measurement)



lines2 = []
with open ('../data2/driving_log.csv') as cvsfile:
	reader = csv.reader(cvsfile)
	for line in reader:
		lines2.append(line)


for line in lines2[1:]:
	source_path =  line[0]
	filename = source_path.split('/')[-1]
	current_path = '../data2/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	
	steering_center = float(line[3])
	measurement = steering_center 
	measurements.append(measurement)


lines3 = []
with open ('../data3/driving_log.csv') as cvsfile:
	reader = csv.reader(cvsfile)
	for line in reader:
		lines3.append(line)


for line in lines3[1:]:
	source_path =  line[0]
	filename = source_path.split('/')[-1]
	current_path = '../data3/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	
	steering_center = float(line[3])
	measurement = steering_center 
	measurements.append(measurement)


lines4 = []
with open ('../data4/driving_log.csv') as cvsfile:
	reader = csv.reader(cvsfile)
	for line in reader:
		lines4.append(line)


for line in lines4[1:]:
	source_path =  line[0]
	filename = source_path.split('/')[-1]
	current_path = '../data4/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	
	steering_center = float(line[3])
	measurement = steering_center 
	measurements.append(measurement)




augmented_images,augmented_measurements = [],[]

for image, measurement in zip (images,measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)



X_train = np.array(augmented_images )
y_train =  np.array(augmented_measurements )

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)) ))

model.add(Convolution2D(24,5,5, subsample =  (2,2), activation = "relu"))
model.add(Convolution2D(36,5,5, subsample =  (2,2), activation = "relu"))
model.add(Convolution2D(48,5,5, subsample =  (2,2), activation = "relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse',optimizer = 'adam')
model.fit(X_train,y_train, validation_split = 0.2, shuffle =  True, nb_epoch=5)

model.save('model.h5')
