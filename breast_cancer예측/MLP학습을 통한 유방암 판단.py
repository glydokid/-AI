#MLP
from sklearn import datasets 
from sklearn.model_selection import train_test_split #데이터셋을 trainning과 test로 분리
from sklearn import svm #svm모델을 사용하여 학습
import numpy as np #매트릭스를 만들기위해
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

digit = datasets.load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size= 0.8)

#mlp 모델 ,옵션 
mlp = MLPClassifier(hidden_layer_sizes=(10000), #히든 레이어 은닉층 노드 갯수
                    learning_rate_init=0.0001, #첫 러닝넷 설정 따로 설정 안하면 이걸로 계속 유지
                    batch_size=10, #전체 훈련 데이터 미니배치 사이즈
                    solver='sgd', #문제를 풀때 미니배치 sgd를 사용한다
                    verbose=True) #훈련을 시킬때 학습과정을 보여준다.
mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

conf = np.zeros((2,2))
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print(conf)

correct = 0
for i in range(2):
    correct += conf[i][i]
accuracy = correct / len(res)
print("테스트 집합에 대한 정확률: ", accuracy*100, "%")