import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

students = pd.read_csv('student-mat.csv', sep=";")

data = students[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

X = np.array(data.drop(["G3"], axis=1))
y = np.array(students["G3"])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# getting the best trained model
# best = 0
#
# for _ in range(60):
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#
#     if accuracy > best:
#         best = accuracy
#         with open('model.pickle', 'wb') as f:
#             pickle.dump(linear, f)
#         print(accuracy)

model = open('model.pickle', 'rb')
linear = pickle.load(model)

predictions = linear.predict(x_test)

print('Coefficient: ', linear.coef_)
print('Intercept: ', linear.intercept_)

for x in range(len(predictions)):
    print(round(predictions[x]), "/", x_test[x], "/", y_test[x])

p = 'G1'
style.use("ggplot")
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()
