__author__ = 'ccg'

from NeuronNet import NeuronNet;
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets

np.random.seed(0)

X, y = sklearn.datasets.make_moons(200, noise=0.20)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

Xs=X.tolist()
Ys=[];
for i in range(0, len(y)):
    if y[i] == 0:
        Ys.append([1, 0]);
    else:
        Ys.append([0, 1]);

#n = NeuronNet(2, 2, [2]);
#n = NeuronNet(2, 2, [3]);
#n = NeuronNet(2, 2, [4]);
#n = NeuronNet(2, 2, [5]);
#n = NeuronNet(2, 2, [10]);
#n = NeuronNet(2, 2, [3,3]);
#n = NeuronNet(2, 2, [20]);

n = NeuronNet(2, 2, [100]);

n.train(Xs, Ys);
print "train finish";


def plot_decision_boundary():
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    rx = xx.ravel()
    ry = yy.ravel()
    rz = []
    for i in range(rx.shape[0]):
        rz.append(n.pred([rx[i], ry[i]]))

    Z = np.asarray(rz);

    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

plot_decision_boundary()