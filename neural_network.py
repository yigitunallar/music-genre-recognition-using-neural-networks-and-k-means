import glob
import os
import matplotlib
# to define plot backends, pick one of those: Agg, Qt4Agg, TkAgg
matplotlib.use('TkAgg')
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

######################### Helper Methods ################################

def loadMusics(filePath):
	musics = []
	for path in filePath:
		X, sr = librosa.load(path)
		musics.append(X)
	return musics

def featureExtraction(fileName):
	raw, rate = librosa.load(fileName)
	stft = np.abs(librosa.stft(raw))
	mfcc = np.mean(librosa.feature.mfcc(y=raw,sr=rate,n_mfcc=40).T, axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=rate).T, axis=0)
	mel = np.mean(librosa.feature.melspectrogram(raw, sr=rate).T, axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=rate).T, axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(raw), sr=rate).T, axis=0)
	return mfcc, chroma, mel, contrast, tonnetz

# Takes parent directory name, subdirectories within parent directory, and file extension as input. 
def parseAudio(parentDirectory, subDirectories, fileExtension="*.wav"):
	features, labels = np.empty((0,193)), np.empty(0)
	for subDir in subDirectories:
		for fn in glob.glob(os.path.join(parentDirectory, subDir, fileExtension)):
			mfcc, chroma, mel, contrast, tonnetz = featureExtraction(fn)
			tempFeatures = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
			features = np.vstack([features, tempFeatures])
			# pop = 1, classical = 2, metal = 3, rock = 0
			if subDir == "pop":
				labels = np.append(labels,1)
			elif subDir == "classical":
				labels = np.append(labels,2)
			elif subDir == "metal":
				labels = np.append(labels,3)
			else : #Corresponds to "rock"
				labels = np.append(labels,0)
	return np.array(features), np.array(labels, dtype=np.int)

#########################################################################
	
def oneHotEncoder(labels):
	n = len(labels)
	nUnique = len(np.unique(labels))
	encoder = np.zeros((n, nUnique))
	encoder[np.arange(n), labels] = 1
	return encoder
############################ NN Params ##################################

training = "training"
test = "test"
subDirectories = ["pop", "classical", "metal", "rock"]
trainingFeatures, trainingLabels = parseAudio(training, subDirectories)
testFeatures, testLabels = parseAudio(test, subDirectories)

trainingLabels = oneHotEncoder(trainingLabels)
testLabels = oneHotEncoder(testLabels)

epochs = 5000
# trainingFeatures is a 32 x 193 matrix
nDim = trainingFeatures.shape[1]
nClasses = 4
nHiddenUnitsOne = 280	
nHiddenUnitsTwo = 300
sd = 1 / np.sqrt(nDim)
learningRate = 0.01

######################## NN Structure in Tensorflow ########################

X = tf.placeholder(tf.float32,[None,nDim])
Y = tf.placeholder(tf.float32,[None,nClasses])

W1 = tf.Variable(tf.random_normal([nDim,nHiddenUnitsOne], mean = 0, stddev=sd))
b1 = tf.Variable(tf.random_normal([nHiddenUnitsOne], mean = 0, stddev=sd))
h1 = tf.nn.tanh(tf.matmul(X,W1) + b1)


W2 = tf.Variable(tf.random_normal([nHiddenUnitsOne,nHiddenUnitsTwo], mean = 0, stddev=sd))
b2 = tf.Variable(tf.random_normal([nHiddenUnitsTwo], mean = 0, stddev=sd))
h2 = tf.nn.sigmoid(tf.matmul(h1,W2) + b2)


W = tf.Variable(tf.random_normal([nHiddenUnitsTwo,nClasses], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([nClasses], mean = 0, stddev=sd))
y = tf.nn.softmax(tf.matmul(h2,W) + b)

init = tf.global_variables_initializer()

costFunction = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y), reduction_indices=[1])) 
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(costFunction)

correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

###################### Training Loop ######################################

costHistory = np.empty(shape=[1],dtype=float)
yTrue, yPred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):            
        _,cost = sess.run([optimizer,costFunction],feed_dict={X:trainingFeatures,Y:trainingLabels})
        costHistory = np.append(costHistory,cost)
    
    yPred = sess.run(tf.argmax(y,1),feed_dict={X: testFeatures})
    yTrue = sess.run(tf.argmax(testLabels,1))

#################### Performance Results ###################################

fig = plt.figure(figsize=(10,8))
plt.plot(costHistory)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.axis([0,epochs,0,np.max(costHistory)])
plt.show()

p,r,f,s = precision_recall_fscore_support(yTrue, yPred, average='micro')
print ("F-Score:", round(f,3))

print("True Labels:", yTrue)
print("Predicted Labels:", yPred)




