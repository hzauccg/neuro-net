__author__ = 'ccg'
from abc import ABCMeta, abstractmethod
import math;
import random;

epsilon = 0.01;
random.seed(0);

class Variable(object):
    def __init__(self):
        self.value = 0
        self.grad = 0

class Function(object):
    @abstractmethod
    def forward(self): pass

    @abstractmethod
    def backprop(self): pass

class Mult(Function):
    def __init__(self, x, y, out):
        self.x = x
        self.y = y
        self.out = out

    def forward(self):
        self.out.value = self.x.value * self.y.value
        self.x.grad = 0
        self.y.grad = 0

    def backprop(self):
        self.x.grad += self.y.value * self.out.grad
        self.y.grad += self.x.value * self.out.grad

class Add(Function):
    def __init__(self, x, out):
        self.x = x
        self.out = out

    def forward(self):
        self.out.value = 0
        for x_i in self.x:
            self.out.value += x_i.value
            x_i.grad = 0

    def backprop(self):
        for x_i in self.x:
            x_i.grad += 1 * self.out.grad

class Sigmoid(Function):
    def __init__(self, x, out):
        self.x = x
        self.out = out

    def doSigmoid(self, z):
        if z > 200:
            return 1;
        if z < -200:
            return 0;
        return 1 / (1 + math.exp(-z))

    def forward(self):
        self.out.value = self.doSigmoid(self.x.value)
        self.x.grad = 0;

    def backprop(self):
        s = self.out.value;
        self.x.grad += s * (1 - s) * self.out.grad

class Neuron(object):
    def __init__(self, X):
        self.W = []
        self.F = []
        mults = []
        for x in X:
            w = Variable();
            w.value = random.random() - 0.5;

            m = Variable();
            self.W.append(w);
            self.F.append(Mult(x, w, m));
            mults.append(m);

        b = Variable();
        self.W.append(b);

        mults.append(b);
        z = Variable();

        self.out = Variable();
        self.F.extend([Add(mults, z), Sigmoid(z, self.out)]);

    def forward(self):
        for f in self.F:
            f.forward();
        return self.out.value

    def backprop(self):
        for f in reversed(self.F):
            f.backprop()

class Layer(object):
    def __init__(self, X, size):
        self.neurons = [];
        for i in range(0, size):
            self.neurons.append(Neuron(X))

    def forward(self):
        for neuron in self.neurons:
            neuron.forward();

    def backprop(self):
        for neuron in self.neurons:
            neuron.backprop();

    def outs(self):
        return map(lambda u: u.out, self.neurons);

class NeuronNet(object):
    def __init__(self, numInput, numOutput, layerSizes):
        self.numInput = numInput;
        self.numOutput = numOutput;
        self.grads = [];
        self.X = [];
        for i in range(0, numInput):
            self.X.append(Variable());

        input = self.X;
        self.layers = [];
        for size in layerSizes:
            layer = Layer(input, size);
            self.layers.append(layer);
            input = layer.outs();

        # append output layer
        self.layers.append(Layer(input, numOutput));

    def forward(self, sample):
        assert self.numInput == len(sample);
        for i in range(0, self.numInput):
            self.X[i].value = sample[i];

        for layer in self.layers:
            layer.forward();

    def outs(self):
        return self.layers[len(self.layers) - 1].outs();

    def updateOutGrads(self, Y):
        assert self.numOutput == len(Y);
        outs = self.outs();
        for i in range(0, len(Y)):
            outs[i].grad = Y[i] - outs[i].value;

    def backprop(self):
        for layer in reversed(self.layers):
            layer.backprop()

    def copyWeights(self):
        weights = [];
        for i in range(0, len(self.layers)):
            layer = self.layers[i];
            for j in range(0, len(layer.neurons)):
                weights.extend(layer.neurons[j].W);
        return weights;

    def copyGrads(self):
        return map(lambda w: w.grad, self.copyWeights())

    def train(self, Xs, Ys):
        for k in range(0, 500):
            for i in range(0, len(Ys)):
                while Ys[i][self.pred(Xs[i])] != 1:
                    self.forward(Xs[i]);
                    self.updateOutGrads(Ys[i]);
                    self.backprop();
                    for w in self.copyWeights():
                        w.value += (w.grad + 0.01 * w.value) * epsilon;
            self.errors(k, Xs, Ys);

    def maxarg(self):
        outs = self.outs();
        best = 0;
        for i in range(0, self.numOutput):
            if outs[i].value > outs[best].value:
                best = i;
        return best;

    def pred(self, ins):
        assert (len(ins) == self.numInput);
        self.forward(ins);
        return self.maxarg();

    def errors(self, k, Xs, Ys):
        error = 0.0;
        for i in range(0, len(Ys)):
            if Ys[i][self.pred(Xs[i])] != 1:
                error += 1
        print "iter %d error: %f rate: %f" % (k, error, error / len(Ys))

def checkGrad():
    net = NeuronNet(2, 1, [2,2]);
    net.forward([1, 1]);
    out = net.outs()[0];
    prevValue = out.value;
    for w in net.copyWeights():
        w.value += epsilon;
        net.forward([1, 1]);
        out.grad = -1;
        net.backprop();
        grad = (out.value - prevValue) / epsilon;
        w.value -= epsilon;
        print "expected:%f actual%f" % (grad, w.grad);

def checkNet():
    net = NeuronNet(2, 1, [2]);
    X = [1,1];
    out = net.outs()[0];
    for i in range(0, 100):
        net.forward(X);
        print "value: %f" % out.value;
        out.grad = -1;
        net.backprop();
        for w in net.copyWeights():
            w.value += w.grad * epsilon;

if __name__ == "__main__":
    checkGrad();