import numpy as np
import pandas as pd

dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
y = y.astype(np.int)
accuracy = []
for q in range(0, 1000):
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf', C=9, gamma=0.22, decision_function_shape='ovo')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    trace = np.trace(cm)
    accuracy.append((trace / 80) * 100)
    print(q)

print(np.mean(accuracy))
# d = pd.DataFrame(data=accuracy)
# d.to_csv('Accuracy_svm.csv', header=False, index=False)

# Applying k-Fold Cross Validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv=10)
# print(accuracies.mean())
# print(accuracies.std())

# Applying Grid Search to find the best model and the best parameters
# from sklearn.model_selection import GridSearchCV
# parameters = [{'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'kernel': ['rbf'],
#                'gamma': [0.17, 0.18, 0.19, 0.2, 0.21, 0.215, 0.22, 0.225],
#                'decision_function_shape':['ovo', 'ovr']}]
# grid_search = GridSearchCV(estimator = classifier,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10,)
# grid_search = grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print(best_accuracy)
# print(best_parameters)
