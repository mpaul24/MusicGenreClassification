#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import pandas as pd

(rate, signal) = wav.read("classical.00000.wav")
print(rate)
print(len(signal))
#plt.plot(signal)
#plt.show()
mfcc_feat = mfcc(signal, rate)
cov = np.cov(mfcc_feat, rowvar=0)
mean_1 = np.mean(mfcc_feat, axis=0)
print(pd.DataFrame(mean_1))
const = np.power(2*np.pi, 6.5)*np.square(np.linalg.det(cov))
mgd = []
for feat in mfcc_feat:
    temp = (1/const)*np.exp(-1 * np.dot(np.dot((feat-mean_1), np.linalg.inv(cov)), (feat-mean_1).T))
    mgd.append(temp)

(rate, signal) = wav.read("rock.00000.wav")

mfcc_feat_1 = mfcc(signal, rate)
cov = np.cov(mfcc_feat_1, rowvar=0)
mean_2 = np.mean(mfcc_feat_1, axis=0)
const = np.power(2*np.pi, 6.5)*np.square(np.linalg.det(cov))
mgd_1 = []
for feat in mfcc_feat_1:
    temp = (1/const)*np.exp(-1 * np.dot(np.dot((feat-mean_2), np.linalg.inv(cov)), (feat-mean_2).T))
    mgd_1.append(temp)
#plt.plot(mgd_1)
#plt.show()
v = []
v.append(mean_1)
v.append(mean_2)
pd.DataFrame(cov).to_csv('cov.csv',index=False,header=False)
# with open("output.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(v)
#d_mfcc_feat = delta(mfcc_feat, 2)
#fbank_feat = logfbank(signal, rate)

#print(fbank_feat[1:3,:])
