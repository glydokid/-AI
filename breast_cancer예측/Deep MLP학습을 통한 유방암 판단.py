#Deep MLP
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras import optimizers

cancer_data = load_breast_cancer()

X_data = cancer_data.data
y_data = cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.8) # 학습 데이터(0.7)와 검증 데이터(0.3)로  전체 데이터 셋을 나눈다

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = Sequential()

model.add(Dense(100, input_shape = (30,), activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

sgd = optimizers.SGD(lr = 0.001)    # stochastic gradient descent optimizer
model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['acc'])   

model.summary()

model.fit(X_train, y_train, batch_size = 30, epochs = 100, verbose = 1)

results = model.evaluate(X_test, y_test)


print(model.metrics_names)     # 모델의 평가 지표 이름
print(results)                 # 모델 평가 지표의 결과값

print('loss: ', results[0])
print('accuracy: ', results[1] *100)