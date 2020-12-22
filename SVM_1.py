import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()

X = cancer.data
y = cancer.target
# malignant = 0, benign = 1

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.15)
classes = ["malignant", "benign"]
clf = svm.SVC(kernel="linear", C=2)
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, prediction)
print(round(acc*100), "%")

for x in range(len(prediction)):
    print(classes[int(round(prediction[x]))], X_test[x], classes[y_test[x]])






