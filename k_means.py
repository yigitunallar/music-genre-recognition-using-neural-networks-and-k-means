
import glob
import os
import matplotlib
# to define plot backends, pick one of those: Agg, Qt4Agg, TkAgg
matplotlib.use('TkAgg')
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd   

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
			else : # Corresponds to "rock"
				labels = np.append(labels,0)
	return np.array(features), np.array(labels, dtype=np.int)

training = "training"
test = "test"
subDirectories = ["pop", "classical", "metal", "rock"]
# Traning Labels [1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 0 0 0 0 0 0 0 0]
trainingFeatures, trainingLabels = parseAudio(training, subDirectories)
# Test Labels [1 1 2 2 3 3 0 0]
testFeatures, testLabels = parseAudio(test, subDirectories)

###################### Training Loop ######################################

model = KMeans(n_clusters=4)
model.fit(trainingFeatures)

#################### Test Results ###################################

print(model.labels_)

