import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#print(x_train)
#print(y_train)
classes = ['malignant', 'benign']

model = svm.SVC(kernel="linear")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

for x in range(len(y_pred)):
    print("predicted: ", classes[y_pred[x]], "\n data: ", x_test[x], "\n actual: ", classes[y_test[x]])
    print("-------------------------------------------------------------------------------------------")
