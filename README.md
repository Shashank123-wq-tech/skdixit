# skdixit
Exploring the world's New Fuel:"Data".
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([2,3,4,5,6,7,8]).reshape(-1,1)
Y = np.array([100000,200000,300000,400000,500000,600000,700000])

model = LinearRegression()
model.fit(X,Y)

y_pred = model.predict(X)

print("Intercept:",model.intercept_)
print("Slope:",model.coef_)

plt.scatter(X , Y, color = "red", label = "Actual")
plt.plot(X , y_pred, color = "black", label = "Predicted line")
plt.legend()
plt.show()
