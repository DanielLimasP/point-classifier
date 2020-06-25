# TODO: Finish
# Train a Linear Classifier...

import numpy as np
import matplotlib.pyplot as plt

class PointClassifier:
    def __init__(self, N, D, K, X, y):
        self.N = N 
        self.D = D
        self.K = K 
        self.X = X
        self.y = y

    def conf_plot(self):
        # Some conf of pyplot
        plt.rcParams['figure.figsize'] = (7.0, 6.0) # set default size of plots
        plt.rcParams['image.interpolation'] = 'nearest'

    def plot_points(self):
        # Plot the points
        for j in range(self.K):
            ix = range(self.N*j, self.N*(j+1))
            r = np.linspace(0.0, 1, self.N) # Radius of balls
            t = np.linspace(j*4, (j+1)*4, self.N) + np.random.randn(N) * 0.2 # Theta
            self.X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            self.y[ix] = j
        plt.scatter(self.X[:,0], self.X[:,1], c = y, s = 40, cmap = 'viridis')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.show()

    def train_linear_classifier(self):
        # Train the linear classifier
        # Init parameteres W and b
        W = 0.01 * np.random.randn(D,K)
        b = np.zeros((1,K))
        # Hyperparameters
        learning_rate = 1e-0
        reg = 1e-3 # regularization strength

        # Gradient descent loop
        num_examples = X.shape[0]
        for i in range(200):
            # Evaluate each score of the class
            scores = np.dot(X, W) + b
            # Compute class probabilities
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)

            # Compute the loss: average cross-entropy loss and regularization
            correct_logprobs = -np.log(probs[range(num_examples), y])
            data_loss = np.sum(correct_logprobs) / num_examples
            reg_loss = 0.5 * reg * np.sum(W*W)
            loss = data_loss - reg_loss
            if i % 10 == 0:
                print("Iteration %d: loss: %f" % (i, loss))
            
            # Compute the gradient on scores
            dscores = probs
            dscores[range(num_examples), y ] -= 1
            dscores /= num_examples

            # Backpropagation of the gradient to the parameters W and b
            dW = np.dot(X.T, dscores)
            db = np.sum(dscores, axis = 0, keepdims = True)

            dW += reg*W # Regularization of the gradient

            # Update parameters W and b
            W += -learning_rate * dW
            b += -learning_rate * db
        
        # Evaluate training accuracy
        scores = np.dot(X, W) + b
        predicted_class = np.argmax(scores, axis = 1)
        print("Training accuracy: %.2f" %(np.mean(predicted_class == y)))

        # Plot the result of the classifier
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
        Z = np.argmax(Z, axis = 1)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap = 'viridis', alpha = 0.4)
        plt.scatter(X[:, 0], X[:, 1], c = y, s = 40, cmap = 'viridis')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.show()


N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # Matrix of feats of every example, I think
y = np.zeros(N*K, dtype='uint8') # Label for all classes

p = PointClassifier(N, D, K, X, y)
p.conf_plot()
p.plot_points()
p.train_linear_classifier()


    