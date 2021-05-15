# This first block of text simply contains all the libraries that need importing as well as initialsing some variables
# of this example. You may change the batch_size and epochs variables, but should not change num_classes, img_rows and
# img_cols as these depend on the dataset being used

from __future__ import print_function
import tensorflow.keras as keras
import tensorflow.keras.datasets as datasets
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
if __name__ == "__main__":
	batch_size = 256
	num_classes = 10
	epochs = 10

	# input image dimensions
	img_rows, img_cols = 32, 32

	print("--------------------------------------------------------------------------------------------------------------")

	# The next block of text handles all the data loading (in this case of the CIFAR-10 dataset) and reshaping so that it
	# can be used to train and evaluate the CNN models. Two sets of data are loaded, the training data, used to generate the
	# model, and the test data, used to evaluate if this model is good at making predictions
	#
	# The final set of code of this cell defines the data augmentation pipeline, and it is commented by default

	print("--------------------------------------------------------------------------------------------------------------")

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
		input_shape = (3, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
		input_shape = (img_rows, img_cols, 3)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test_orig = y_test
	y_test = keras.utils.to_categorical(y_test, num_classes)

	# Uncomment these lines to enable the data augmentation
	datagen = ImageDataGenerator(
		#featurewise_center=True,
		#featurewise_std_normalization=True,
		rotation_range=15,
		width_shift_range=0.15,
		height_shift_range=0.15,
		horizontal_flip=True,
		#channel_shift_range=5,
		#zca_whitening=True
		)
	# compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied)
	datagen.fit(x_train)

	print("--------------------------------------------------------------------------------------------------------------")

	# This cell defines the CNN architecture. There are two versions of this code. The first one, identical to the practical
	# defines layer by layer the architecture.
	#
	# The second version loads a pre-defined architecture, that had been trained for a different problem. It removes the
	# last layer (because that would be specific to the original problem, and adds some new layers. This process of adapting
	# a network trained on a different problem is called Transfer Learning.
	#
	# The only essential new layer to add is the softmax one (last) that specifies the number of classes of the new dataset.
	# You can play with this code by commenting or uncommenting some of the previous lines that add layers.
	#
	# Please note that the syntax used in here to define the network is different from the practical. We don't add layers to
	# a stack with model.add, but chain each new layer to the previous one:
	# "newLayer = NameOfNewLayer(arguments)(oldLayer)". In the code below variable x is used in place of both newLayer and
	# oldLayer.

	print("--------------------------------------------------------------------------------------------------------------")



	# Comment the model above and uncomment the following lines to switch to a pre-defined CNN architecture
	base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
	#base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
	#base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
	#base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(1024, activation='relu')(x)
	# and a logistic layer -- let's say we have 200 classes
	predictions = Dense(num_classes, activation='softmax')(x)
	model = models.Model(inputs=base_model.input, outputs=predictions)




	# The rest of this cell is common to both defining the full architecture or using a pre-trained one
	model.compile(loss=keras.losses.categorical_crossentropy,
				  #optimizer=keras.optimizers.SGD(),
				  #optimizer=keras.optimizers.RMSprop(),
				  optimizer=keras.optimizers.Adamax(learning_rate=0.0004),
				  metrics=['accuracy'])

	model.summary()

	print("--------------------------------------------------------------------------------------------------------------")

	# This is the block of code that trains the model and evaluates its predictive capacity on the test data. There are two
	# versions of the model.fit function, without and with image augmentation (defined in one of the cells above)
	#
	# The evaluation code is more detailed that in the practical and has two parts. In the first part three performance
	# metrics (precision, recall and F1) are computed for the examples of each of the 10 classes of the dataset, and then
	# averaged. For the definition of these metrics, please see https://en.wikipedia.org/wiki/F-score.
	#
	# The second part of the evaluation is what is called a confusion matrix. It is a matrix of size
	# num_classes x num_classes, in which rows are associated to the actual classes of the test set, and columns to the
	# predicted classes. The number in each cell indicates the number of examples from the test set that belong to the class
	# of the row, and have been predicted as the class of the column. Examples in the diagonal are correct classifications.
	# Examples in the rest of the matrix are the misclassified ones.
	#
	# You will notice in the matrix that some columns have much bigger numbers than others. These means that the CNN model
	# tends to predict the classes for these columns much more often than for the other columns, i.e. some classes are
	# ignored.
	#
	# Finally, a plot of how the loss function (measure of the training error of the network) changes through the epochs of
	# training is shown.

	print("--------------------------------------------------------------------------------------------------------------")

	# history = model.fit(x_train, y_train,
	#                     batch_size=batch_size,
	#                     epochs=epochs,
	#                     verbose=1)

	# fits the model on batches with real-time data augmentation:
	history = model.fit(
						datagen.flow(x_train, y_train, batch_size=batch_size),
						steps_per_epoch=len(x_train) / batch_size,
						epochs=epochs,
						verbose=1)
	#datagen2 = ImageDataGenerator(zca_whitening=True)
	#datagen2.fit(x_test)
	predictions = model.predict(x_test)
	predicted_classes = np.argmax(predictions, axis=1)

	print(metrics.classification_report(y_test_orig, predicted_classes))
	print()
	print(metrics.confusion_matrix(y_test_orig, predicted_classes))

	plt.plot(history.history['loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.show()

	print("--------------------------------------------------------------------------------------------------------------")

	# This block of code picks an image from the test set (you can change the index to choose a different image), and shows
	# the probabilities that the CNN model estimates for each class, then shows the actual image.

	print("--------------------------------------------------------------------------------------------------------------")


	img_index = 42
	image = x_test[img_index]

	pred = model.predict(np.expand_dims(image, axis=0))[0]
	for cl in range(num_classes):
		print("Probability for class {}: {}".format(cl, pred[cl]))
	print("\nThe winner is {}".format(np.argmax(pred)))
	print("The correct class is {}\n".format(np.argmax(y_test[img_index])))

	plt.imshow(image.squeeze(), cmap='viridis')
	plt.show()
