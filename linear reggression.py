import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()
diabetes_X = diabetes.data
print(diabetes_X)
diabetes_X_train = diabetes_X[:-30]

diabetes_X_test = diabetes_X[-30:]



diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predicted=model.predict(diabetes_X_test)

print("Mean squared error is:",mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))

print("weight:",model.coef_)

print("intercept:",model.intercept_)

#plt.scatter(diabetes_X_test,diabetes_Y_test)
#plt.plot(diabetes_X_test,diabetes_Y_predicted)

#plt.show()
#Mean squared error is: 1826.5364191345425
#weight: [  -1.16924976 -237.18461486  518.30606657  309.04865826 -763.14121622
 # 458.90999325   80.62441437  174.32183366  721.49712065   79.19307944]
#intercept: 153.05827988224112
