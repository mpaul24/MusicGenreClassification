
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np
import os
import pandas as pd

CLASSICAL_DIR = "C:\\Users\\Manojit Paul\\Music\\Classical\\"
METAL_DIR = "C:\\Users\\Manojit Paul\\Music\\Metal\\"
JAZZ_DIR = "C:\\Users\\Manojit Paul\\Music\\Jazz\\"
POP_DIR = "C:\\Users\\Manojit Paul\\Music\\Pop\\"
PATH = "E:\\git\\python_speech_features\\covariance\\"

x = [CLASSICAL_DIR, METAL_DIR, JAZZ_DIR, POP_DIR]
t = 100
columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9',
           'Feature10', 'Feature11', 'Feature12', 'Feature13']
dataset = []
genre = []
for i in x:
    if i == CLASSICAL_DIR:
        for index in range(0, t):
            genre.append(0)
            file_name = "classical.000"+str(index).zfill(2)
            file = file_name+".wav"
            (rate, signal) = wav.read(CLASSICAL_DIR+file)
            mfcc_feat = mfcc(signal, rate)
            cov = np.cov(mfcc_feat, rowvar=0)
            mean = np.mean(mfcc_feat, axis=0)
            # if not os.path.exists(PATH+file_name):
            #     os.makedirs(PATH+file_name)
            pd.DataFrame(cov).to_csv(PATH+"classical"+str(index)+'.csv', index=False, header=False)
            dataset.append(mean)
    elif i == METAL_DIR:
        for index in range(0, t):
            genre.append(1)
            file_name = "metal.000" + str(index).zfill(2)
            file = file_name + ".wav"
            (rate, signal) = wav.read(METAL_DIR + file)
            mfcc_feat = mfcc(signal, rate)
            cov = np.cov(mfcc_feat, rowvar=0)
            mean = np.mean(mfcc_feat, axis=0)
            # if not os.path.exists(PATH+file_name):
            #     os.makedirs(PATH+file_name)
            pd.DataFrame(cov).to_csv(PATH + "metal"+str(index) + '.csv', index=False, header=False)
            dataset.append(mean)
    elif i == JAZZ_DIR:
        for index in range(0, t):
            genre.append(2)
            file_name = "jazz.000" + str(index).zfill(2)
            file = file_name + ".wav"
            (rate, signal) = wav.read(JAZZ_DIR + file)
            mfcc_feat = mfcc(signal, rate)
            cov = np.cov(mfcc_feat, rowvar=0)
            mean = np.mean(mfcc_feat, axis=0)
            # if not os.path.exists(PATH + file_name):
            #     os.makedirs(PATH + file_name)
            pd.DataFrame(cov).to_csv(PATH + "jazz"+str(index) + '.csv', index=False, header=False)
            dataset.append(mean)
    elif i == POP_DIR:
        for index in range(0, t):
            genre.append(3)
            file_name = "pop.000" + str(index).zfill(2)
            file = file_name + ".wav"
            (rate, signal) = wav.read(POP_DIR + file)
            mfcc_feat = mfcc(signal, rate)
            cov = np.cov(mfcc_feat, rowvar=0)
            mean = np.mean(mfcc_feat, axis=0)
            # if not os.path.exists(PATH + file_name):
            #     os.makedirs(PATH + file_name)
            pd.DataFrame(cov).to_csv(PATH + "pop"+str(index) + '.csv', index=False, header=False)
            dataset.append(mean)

dataset = pd.DataFrame(data=dataset, columns=columns)
dataset['genre'] = genre
dataset = dataset[['genre', 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7',
                   'Feature8', 'Feature9', 'Feature10', 'Feature11', 'Feature12', 'Feature13']]
dataset.to_csv("Dataset.csv", index=False)
#x = numpy.loadtxt(open("cov.csv", "r"), delimiter=",", skiprows=1)
#print(type(x))