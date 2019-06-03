import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import KMeans
from sklearn import metrics


def load_data():
	"""
	Loads the data and performs some preprocessing to the features that are not numerical.

	Returns:
		X_data: A numpy array with the 6 features.
		y_data: A numpy array with the labels.
	"""

	df = pd.read_csv('train.csv')

	y_data = df['Survived'].as_matrix()
	df = df[['Pclass','Sex','Age','SibSp','Parch','Fare']]

	class_vector = df['Pclass'].as_matrix()
	class_vector_cat = np_utils.to_categorical(class_vector,
		num_classes=4)[:,1:] # 3 classes

	int_sex = {'male':0, 'female':1}
	sex_vector = [int_sex[s] for s in df['Sex'].tolist()]
	sex_vector_cat = np_utils.to_categorical(sex_vector,
		num_classes=2) # 2 sexes

	mean_age = int(df['Age'].mean())
	
	age_vector = df['Age'].as_matrix()
	age_vector[np.isnan(age_vector)] = mean_age
		
	numeric_data = df[['SibSp','Parch','Fare']].as_matrix()

	numeric_data = preprocessing.scale(numeric_data)

	X_data = np.concatenate((numeric_data,
		age_vector.reshape((len(age_vector),1)),
		sex_vector_cat,class_vector_cat),
		axis=1)

	return X_data, y_data


def knn_classifier(X_train,X_val,y_train,y_val):
	"""
	Trains a KNN classifier trying out different values for 'K' and the weights.

	Arguments:
		X_train: numpy array with the training data.
		y_train: numpy array with the training labels.
		X_val: numpy array with the validation data.
		y_val: numpy array with the validation labels.

	Prints:
		The validation set accuracy and the best set of parameters.
	"""

	print 'KNN Classifier'
	best_acc = 0.0
	best_params = []
	for k in [1,3,5,7]:
		for w in ['uniform','distance']:
			knn = KNeighborsRegressor(n_neighbors=k,
				weights=w,
				metric='manhattan').fit(X_train,y_train)

			acc = knn.score(X_val,y_val)

			if acc > best_acc:
				best_acc = acc
				best_params = [k,w]

	print 'best K:',best_params[0],'best weights:',best_params[1]
	print 'acc:',best_acc

	return


def random_forest(X_train,X_val,y_train,y_val):
	"""
	Trains a random forest classifier trying out different values for the criterion and number of trees.

	Arguments:
		X_train: numpy array with the training data.
		y_train: numpy array with the training labels.
		X_val: numpy array with the validation data.
		y_val: numpy array with the validation labels.

	Prints:
		The validation set accuracy and the best set of parameters.
	"""

	print 'Random Forest Classifier'
	criteria = ['gini','entropy']
	num_trees = [5,10,15,20,25,30]

	best_metrics = []
	best_acc = 0.0
	best_params = []
	for crit in criteria:
		for tr in num_trees:
			rf = RandomForestClassifier(criterion=crit,
				n_estimators=tr,
				random_state=10)
			rf.fit(X_train,y_train)

			acc = rf.score(X_val,y_val)

			preds = rf.predict(X_val)

			metrics = precision_recall_fscore_support(y_val, preds,
				average='binary')
			
			if acc > best_acc:
				best_acc = acc
				best_metrics = [acc,metrics[0],metrics[1],metrics[2]]
				best_params = [crit,tr]

	print 'best criteria:',best_params[0],'best num of trees:',best_params[1]
	print 'acc:',best_metrics[0]
	print 'precision:',best_metrics[1]
	print 'recall:',best_metrics[2]
	print 'f1-score:',best_metrics[3]

	return


def svm(X_train,X_val,y_train,y_val):
	"""
	Trains a SVM classifier trying out different values for the kernel and C parameter.

	Arguments:
		X_train: numpy array with the training data.
		y_train: numpy array with the training labels.
		X_val: numpy array with the validation data.
		y_val: numpy array with the validation labels.

	Prints:
		The validation set accuracy and the best set of parameters.
	"""

	print 'SVM Classifier'
	best_acc = 0.0
	best_params = []
	kernels = ['linear', 'rbf']
	for kerner_func in kernels:
		c_params = [1e-2, 1, 10]
		for c_param in c_params:
			print kerner_func,c_param
			svm_model = SVC(kernel=kerner_func,
				C=c_param)
			svm_model.fit(X_train,y_train)

			acc = svm_model.score(X_val,y_val)
			if acc > best_acc:
				best_acc = acc
				best_params = [kerner_func,c_param]

	print 'best kernel function:',best_params[0],'best C parameter:',best_params[1]
	print 'acc:',best_acc

	return


def build_model(input_dim,
	lr=0.0002):
	"""
	Builds an MLP classifier.

	Arguments:
		input_dim: the dimensionality of the input features.
		lr: the learning rate.

	Return:
		A compiled keras model
	"""

	input_layer = Input(shape=(input_dim,))

	# Try linear, sigmoid and relu activation
	dense1 = Dense(100,
		activation='elu',
		kernel_initializer='glorot_uniform')(input_layer)

	drop1 = Dropout(0.0)(dense1)

	dense2 = Dense(70,
		activation='elu',
		kernel_initializer='glorot_uniform')(drop1)

	drop2 = Dropout(0.0)(dense2)

	dense3 = Dense(20,
		activation='elu',
		kernel_initializer='glorot_uniform')(drop2)

	drop3 = Dropout(0.0)(dense3)

	output = Dense(1,
		activation='sigmoid')(drop3)

	model = Model(inputs=input_layer,
		outputs=output)

	adam = Adam(lr=lr,
		decay=1e-6)

	model.compile(loss='binary_crossentropy',
		optimizer=adam,
		metrics=['accuracy'])

	model.summary()
	return model


def mlp_model(X_train,X_val,y_train,y_val):
	"""
	Trains an MLP classifier.

	Arguments:
		X_train: numpy array with the training data.
		y_train: numpy array with the training labels.
		X_val: numpy array with the validation data.
		y_val: numpy array with the validation labels.

	Prints:
		The training and validation set loss and accuracy for each epoch.
	"""

	print 'MLP Classifier'
	STAMP = 'titanic_mlp'
	input_dim = X_train.shape[1]
	model = build_model(input_dim)

	model_json = model.to_json()

	with open("model/" + STAMP + ".json", "w") as json_file:
		json_file.write(model_json)

	early_stopping =EarlyStopping(monitor='val_loss',
		patience=5)
	bst_model_path = "model/" + STAMP + '.h5'
	model_checkpoint = ModelCheckpoint(bst_model_path,
		monitor='val_acc',
		verbose=1,
		save_best_only=True,
		save_weights_only=True)

	hist = model.fit(X_train,y_train,
		validation_data=(X_val, y_val),
		epochs=1000,
		batch_size=64,
		shuffle=True,
		callbacks=[early_stopping,model_checkpoint])

	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	return


def k_means(X_data,y_data):
	"""
	Performs K-means clustering trying out different K values.

	Arguments:
		X_train: numpy array with the training data.
		y_train: numpy array with the training labels.
		X_val: numpy array with the validation data.
		y_val: numpy array with the validation labels.

	Prints:
		Some metrics for each K value.
	"""

	for num_c in [2,3,4]:
		kmeans = KMeans(n_clusters=num_c, random_state=10)

		kmeans.fit(X_data)

		inertia = kmeans.inertia_
		homogeneity = metrics.homogeneity_score(y_data, kmeans.labels_)
		completeness = metrics.completeness_score(y_data, kmeans.labels_),
		silhouette_sc = metrics.silhouette_score(X_data, kmeans.labels_,
			metric='euclidean',
			sample_size=200)
		
		print 'num clusters:',num_c
		print 'inertia:',inertia
		print 'homogeneity:',homogeneity
		print 'completeness:',completeness[0]
		print 'silhouette score:',silhouette_sc

	return
		


if __name__ == '__main__':
	X_data,y_data = load_data()
	X_train,X_val,y_train,y_val = train_test_split(X_data,y_data,
		test_size=0.2,
		random_state=10)

	print X_train.shape,X_val.shape
	#knn_classifier(X_train,X_val,y_train,y_val)
	#random_forest(X_train,X_val,y_train,y_val)
	#svm(X_train,X_val,y_train,y_val)
	#mlp_model(X_train,X_val,y_train,y_val)

	k_means(X_data,y_data)

