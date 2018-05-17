import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = pd.read_csv('data.txt', sep = "\t", header=0)
print(data)

regression = linear_model.LinearRegression()

X = (data.x).values.reshape(-1,1)
y = (data.y).values.reshape(-1,1)

#linear regresson
linear_model = regression.fit(X, y)
y_pred_linear = linear_model.predict(X)
r_sq = r2_score(y, y_pred_linear)
n = data.shape[0]
r2_adj_linear = 1 - (1 - r_sq) * (n - 1)/(n - 2)

#plot linear model
plt.figure(1)
plt.subplot(231)
title_l = "Linear model, Adj R-sq = {}".format(round(r2_adj_linear,2))
plt.plot(X, y, 'bo', X, y_pred_linear, 'r', markersize=3)
plt.title(title_l, fontsize = 10)

#polynomial regression
subplot_no = 232
r_sq_all = [r2_adj_linear]

#fit a model and create a subplot for degree 2 to 6 polynomial
for i in range(2,7):

    poly = PolynomialFeatures(degree = i)
    X_transform = poly.fit_transform(X)

    poly_model = regression.fit(X_transform,y)
    y_pred_poly = poly_model.predict(X_transform)

    #calculate r-squared
    r_sq = r2_score(y, y_pred_poly)

    #calculate adjusted R-squared
    #number of observations
    n = data.shape[0]
    #number of features (1 is a constant)
    k = X_transform.shape[1] - 1
    r2_adj = 1-(1-r_sq)*(n - 1)/(n-(k + 1))

    r_sq_all.append(r2_adj)

    title_p = "Polynomial degree {}, Adj R-sq = {}".format(i,round(r2_adj,2))
    plt.subplot(subplot_no)
    plt.plot(X, y, 'bo', X, y_pred_poly, 'r', markersize=3)
    plt.title(title_p, fontsize = 10)
    subplot_no += 1

#plot adjusted R2 against degree of polynomial
plt.figure(2)
plt.plot(range(1,7), r_sq_all)
plt.xlabel('degree')
plt.ylabel('Adjusted R-squared')

plt.show()