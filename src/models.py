import numpy as np

class MultiLayerPerceptron:
    def __init__(self, hidden_layers, activation='Sigmoid', learning_rate=0.01, epochs=1000, use_bias=True,seed=42):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_bias = use_bias
        self.accuracy_history = []
        self.weights = []
        self.biases = []
        self.seed = seed

    def activationFunction(self, x):
        if self.activation == 'Sigmoid':
            values = x.copy()
            values[values > 500] = 500
            values[values < -500] = -500
            return 1 / (1 + np.exp(-values))
        return np.tanh(x)

    def activationFunctionDerivative(self, a):
        if self.activation == 'Sigmoid':
            return a * (1 - a)
        return 1 - (a ** 2)

    def fit(self, X, y):
        np.random.seed(self.seed)
        target_y = 2 * y - 1 if self.activation == 'Tanh' else y.copy()
        layer_sizes = [X.shape[1]] + self.hidden_layers + [y.shape[1]]
        self.weights, self.biases = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(np.random.uniform(-0.5, 0.5, (n_out, n_in)))
            self.biases.append(np.random.uniform(-0.5, 0.5, n_out) if self.use_bias else np.zeros(n_out))
        self.accuracy_history = []
        for epoch in range(self.epochs):
            correct = 0
            indices = np.random.permutation(len(X))
            for i in indices:
                inputs = [X[i]]
                for W, b in zip(self.weights, self.biases):
                    inputs.append(self.activationFunction(np.dot(W, inputs[-1]) + b))
                if list(inputs[-1]).index(max(inputs[-1])) == list(y[i]).index(max(y[i])):
                    correct += 1
                delta = (target_y[i] - inputs[-1]) * self.activationFunctionDerivative(inputs[-1])
                for l in range(len(self.weights) - 1, -1, -1):
                    if l > 0:
                       prev_delta = np.dot(self.weights[l].T, delta) * self.activationFunctionDerivative(inputs[l])
                    self.weights[l] += self.learning_rate * np.outer(delta, inputs[l])
                    if self.use_bias:
                        self.biases[l] += self.learning_rate * delta
                    if l > 0:
                        delta = prev_delta
            self.accuracy_history.append(correct / len(X))
        return self

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            inputs = X[i]
            for W, b in zip(self.weights, self.biases):
                inputs = self.activationFunction(np.dot(W, inputs) + b)
            predictions.append(inputs)
        return np.array(predictions)

    def predict_classes(self, X):
        raw_preds = self.predict(X)
        class_preds = []
        for pred in raw_preds:
            one_hot = np.array([0] * len(pred))
            best_idx = list(pred).index(max(pred))
            one_hot[best_idx] = 1
            class_preds.append(one_hot)
        return np.array(class_preds)

def evaluate_predictions(y_true, y_pred_classes):
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    correct = 0
    for i in range(len(y_true)):
        actual = list(y_true[i]).index(max(y_true[i]))
        predicted = list(y_pred_classes[i]).index(max(y_pred_classes[i]))
        cm[actual][predicted] += 1
        if actual == predicted:
            correct += 1
    return cm, correct / len(y_true)