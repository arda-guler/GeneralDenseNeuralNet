import numpy as np

class NeuralNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.n_layers = len(n_neurons)
        self.weights = []
        self.biases = []

        for i in range(1, self.n_layers):
            self.weights.append(np.random.randn(n_neurons[i], n_neurons[i-1]))
            self.biases.append(np.random.randn(n_neurons[i], 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def fwd(self, X):
        A = X
        Zs = []
        As = [A]

        for i in range(self.n_layers - 1):
            Z = np.dot(self.weights[i], A) + self.biases[i]
            A = self.sigmoid(Z)
            Zs.append(Z)
            As.append(A)

        return Zs, As

    def bck(self, X, Y, Zs, As):
        dZs = [None] * (self.n_layers - 1)
        dWs = [None] * (self.n_layers - 1)
        dbs = [None] * (self.n_layers - 1)

        dZs[-1] = As[-1] - Y
        dWs[-1] = np.dot(dZs[-1], As[-2].T)
        dbs[-1] = np.sum(dZs[-1], axis=1, keepdims=True)

        for i in range(self.n_layers - 3, -1, -1):
            dZs[i] = np.dot(self.weights[i+1].T, dZs[i+1]) * self.sigmoid_prime(Zs[i])
            dWs[i] = np.dot(dZs[i], As[i].T)
            dbs[i] = np.sum(dZs[i], axis=1, keepdims=True)

        return dWs, dbs

    def update_weights(self, dWs, dbs, rate):
        for i in range(self.n_layers - 2, -1, -1):
            self.weights[i] -= rate * dWs[i]
            self.biases[i] -= rate * dbs[i]

    def train(self, X, Y, epochs=1000, rate=0.01):
        for epoch in range(epochs):
            Zs, As = self.fwd(X)
            dWs, dbs = self.bck(X, Y, Zs, As)
            self.update_weights(dWs, dbs, rate)

    def predict(self, X):
        Zs, As = self.fwd(X)
        return As[-1]

    def __repr__(self):
        output = "\n=== Neural Network ==="
        for idx_w, w in enumerate(self.weights):
            output += "\n\nLayer " + str(idx_w+1)
            for idx_n, n in enumerate(w):
                output += "\nNeuron " + str(idx_n+1) + "\n"
                output += str(w)

        output += "\n\n=== END Neural Network ==="

        return output

    def save(self, filename="nn.txt"):
        print("Saving neural network state to " + filename + ".")
        copy_weights = []
        for arr in self.weights:
            copy_weights.append(arr.tolist())

        copy_biases = []
        for arr in self.biases:
            copy_biases.append(arr.tolist())

        output = str(copy_weights) + "\n"
        output += str(copy_biases) + "\n"

        f = open(filename, "w")
        f.write(output)
        f.close()

        print("Network state saved.\n")

    def load(self, filename="nn.txt"):
        print("Loading neural network state from " + filename + ".")
        f = open(filename, "r")
        f_lines = f.readlines()
        f.close()
        f_weights = eval(f_lines[0])
        f_biases = eval(f_lines[1])

        for idx_l in range(len(f_weights)):
            f_weights[idx_l] = np.array(f_weights[idx_l])

        for idx_l in range(len(f_biases)):
            f_biases[idx_l] = np.array(f_biases[idx_l])

        self.weights = f_weights
        self.biases = f_biases
        print("Network state loaded.\n")

# Training on XOR function
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_train = np.array([[0], [1], [1], [0]])

n_neurons = [2, 5, 5, 1]
nn = NeuralNetwork(n_neurons)
nn.train(X_train.T, Y_train.T, epochs=10000, rate=0.1)

X_test = X_train
Y_pred = nn.predict(X_test.T)
print("Input Data:\n", X_test)
print("Predicted Output:\n", Y_pred.T)

# save trained state
nn.save("testsave.txt")

# load state into a fresh, untrained network
nn2 = NeuralNetwork(n_neurons)
nn2.load("testsave.txt")

# fresh network gets a headstart with instant training
# so the work isn't wasted
Y_pred = nn2.predict(X_test.T)
print("Predicted Output:\n", Y_pred.T)

