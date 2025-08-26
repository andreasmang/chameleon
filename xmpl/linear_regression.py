import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# generate synthetic data
np.random.seed(42)
u = 2.0 * np.random.rand(100, 1)  # features
y = 4.0 + 3.0 * u + np.random.randn(100, 1)

# fit linear regression model
lin_reg = LinearRegression()
lin_reg.fit(u, y)
y_pred = lin_reg.predict(u)

# print coefficients
print("intercept:", lin_reg.intercept_)
print("coefficients:", lin_reg.coef_)

# plot data and regression line
plt.scatter(u, y, color="blue", label="data points")
plt.plot(u, y_pred, color="red", linewidth=2, label="fitted model")
plt.xlabel("u (predictor)")
plt.ylabel("y (response)")
plt.title("simple linear regression example")
plt.legend()
plt.show()




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
