import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

students = pd.read_csv('student-mat.csv', sep=";")

#print(students.head())
data = students[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

X = np.array(data.drop(["G3"], axis=1))
y = np.array(students["G3"])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(round(predictions[x]), "/", x_test[x], "/", y_test[x])
