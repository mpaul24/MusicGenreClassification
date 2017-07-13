import pandas as pd
# dataset = pd.read_csv('Distance.csv', header=None)
# dataset = dataset.iloc[1:, :].values
# distance = []
# for i in range(0, len(dataset)):
#     distance.append(dataset[i][4])
# x = []
# for i in range(0, 400):
#     for j in range(0, 400):
#         x.append([i, j])
# category = []
# for i in range(0, 4):
#     for j in range(0, 40000):
#         category.append(i)
# print(len(category))
# column = ["genre_1", "genre_2"]
# db = pd.DataFrame(data=x, columns=column)
# db["distance"] = distance
# db["category"] = category
# db = db[["genre_1", "genre_2", "distance", "category"]]
# db.to_csv("Distance_Updated.csv", index=False)
import numpy as np
dataset = pd.read_csv('Dataset.csv', header=None)
X = np.array(range(0,400))
y = dataset.iloc[1:, 0].values

dataset = pd.read_csv('Distance_Updated.csv', header=None)
dataset = dataset.iloc[1:, :].values

accuracy = []
for q in range(0, 500):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # print(X_train)
    # print(X_test)
    k = 5


    dic = dict()
    y_pred = []
    for i in range(0, len(dataset)):
        key = str(dataset[i][0]) + str(dataset[i][1])
        dic[key] = float(dataset[i][2])

    for i in X_test:
        temp = []
        index = 0
        c = dict()
        c[0] = 0
        c[1] = 0
        c[2] = 0
        c[3] = 0
        for j in X_train:
            key = str(i) + str(j)
            temp.append([dic[key], index])
            index += 1
        temp.sort(key=lambda x:x[0])
        for j in range(0, k):
            ind = temp[j][1]
            clas = y_train[ind]
            c[int(clas)] += 1
        pred_class = 0
        max = c[0]
        for j in range(1, 4):
            if c[j] > max :
                max = c[j]
                pred_class = j
        y_pred.append(pred_class)
    # print(y_pred)
    for i in range(0, len(y_pred)):
        y_pred[i] = str(y_pred[i])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    trace = np.trace(cm)
    accuracy.append((trace/80)*100)
    print(q)
print(accuracy)
d = pd.DataFrame(data=accuracy)
d.to_csv('Accuracy_k5.csv',header=False, index=False)
import matplotlib.pyplot as plt
y = range(1, 501)
plt.bar(y, accuracy, label='Accuracy', color='r')
plt.show()
plt.plot(accuracy)
plt.show()
