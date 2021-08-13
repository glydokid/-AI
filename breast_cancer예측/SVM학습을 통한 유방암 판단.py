#SVM
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn import svm 
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


digit = datasets.load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size = 0.8)

sc = StandardScaler()
sc.fit(x_train)

x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

s = svm.SVC(gamma=0.001, C = 30)
s.fit(x_train_std, y_train) #학습(모델링) 피처와 라벨

pre = s.predict(x_test_std)
print("정확도: ", accuracy_score(y_test,pre)*100, "%")