# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class SimpleLinearRegressionClassifier:
    def __init__(self):
        self.beta1 = 0.0
        self.beta0 = 0.0

    def train(self, x_train, y_train):
        # find centroid
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # compute the slope beta1
        # formula Σ((xi - xmean) * (yi - ymean)) / Σ ((xi - xmean)**2)
        self.beta1 = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean)**2)

        # compute y intercept beta0 from y = beta0 + beta1 * x
        self.beta0 = y_mean - self.beta1 * x_mean

    def predict(self, x_pred):
        y_pred = self.beta0 + self.beta1 * x_pred
        return y_pred

    def mse(self, x_mse, y_mse):
        y_mse_pred = self.predict(x_mse)
        mse = ((y_mse_pred - y_mse) ** 2).mean(axis=0)
        return mse

    # r2 or coefficient of determination - closer to 1.0 means more significance
    def rsquare(self, x_rsquare, y_actual):
        y_rsquare_pred = self.predict(x_rsquare)
        y_actual_mean = np.mean(y_actual)
        sse = np.sum((y_actual - y_rsquare_pred)**2)        # sum of squared errors, which is the sum of the square of the difference between actual y and the predicted y
        ssr = np.sum((y_rsquare_pred - y_actual_mean)**2)   # sum of squared residuals, which is the sum of the square of the difference between the predicted y and the actual y mean
        sst = np.sum((y_actual - y_actual_mean)**2)         # sum of squares total, which is the sum of the square of the difference between the actual y and the actual y mean
        rsquare = ssr / sst
        return (rsquare, sse, ssr, sst)

# load de sklearn iris dataset
data_iris = load_iris()
X_iris = data_iris.data[:, 0]   # sepal length
y_iris = data_iris.data[:, 1]   # sepal width

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

model = SimpleLinearRegressionClassifier()
model.train(X_train, y_train)

y_pred = model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(X_test, y_test)
ax.plot(X_test, y_pred, color="red")

mse = model.mse(X_test, y_test)
rsquare, sse, ssr, sst = model.rsquare(X_test, y_test)

ax.set_title("Linear regression on iris dataset w/ MSE %.2f" % mse + ", R2 of %.2f" % rsquare +
             " (sse: %.2f" % sse + ", ssr: %.2f" % ssr + ", sst: %.2f" % rsquare + ")")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()