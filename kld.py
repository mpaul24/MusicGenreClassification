import pandas as pd
import numpy as np
PATH = "E:\\git\\python_speech_features\\covariance\\"
dataset = pd.read_csv('Dataset.csv')
dataset = dataset.iloc[:, 1:].values
genre = ["classical", "metal", "jazz", "pop"]
Distance = []
t = 100
for i in range(0, 4):
    x = 0
    for j in range(0, t):
        mean_ = dataset[x+j]
        cov_ = np.loadtxt(open(PATH+genre[i]+str(j)+".csv", "r"), delimiter=",")
        for k in range(0, 4):
            y = 0
            for l in range(0, t):
                mean = dataset[y+l]
                cov = np.loadtxt(open(PATH+genre[k]+str(l)+".csv", "r"), delimiter=",")

                log_part = np.log(np.abs(np.linalg.det(cov_)/np.linalg.det(cov)))
                trace = np.trace(np.dot(np.linalg.inv(cov_), cov))
                single = np.dot(np.dot((mean-mean_).T, np.linalg.inv(cov_)), (mean-mean_))
                kl_ = (log_part+trace+single-13)/2

                log_part_ = np.log(np.abs(np.linalg.det(cov)) / np.linalg.det(cov_))
                trace_ = np.trace(np.dot(np.linalg.inv(cov), cov_))
                single_ = np.dot(np.dot((mean_ - mean).T, np.linalg.inv(cov)), (mean_ - mean))
                kl = (log_part_+trace_+single_-13)/2
                kldDistane = kl+kl_

                Distance.append([i, j, k, l, kldDistane])
            y += 100
    x += 100

columns = ["Genre", "SongIndex", "ComparedSongGenre", "ComparedSongIndex", "Distance"]
dataset = pd.DataFrame(data=Distance, columns=columns)
dataset.to_csv('Distance.csv', index=False)