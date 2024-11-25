import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define activation functions and respective derivatives
def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return (x > 0).astype(float)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        # Choose activation function
        if activation == 'tanh':
            self.activation, self.activation_derivative = tanh, tanh_derivative
        elif activation == 'relu':
            self.activation, self.activation_derivative = relu, relu_derivative
        elif activation == 'sigmoid':
            self.activation, self.activation_derivative = sigmoid, sigmoid_derivative
        else:
            raise ValueError("Unsupported activation function")
        
        # For visualization
        self.hidden_activations = None
        self.gradients = None

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
         # Compute activations for the hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1  # Input to hidden layer
        self.A1 = self.activation(self.Z1)     # Activation of hidden layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Input to output layer
        self.A2 = sigmoid(self.Z2)             # Activation of output layer

        out = self.A2

        # TODO: store activations for visualization
        self.hidden_activations = self.A1
       
        return out

    def backward(self, X, y):
        m = X.shape[0]

        # TODO: compute gradients using chain rule
        #OUtput layer
        dZ2 = self.forward(X) - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # TODO: update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # TODO: store gradients for visualization
        self.gradients = (np.abs(dW1).mean(), np.abs(dW2).mean())

        # pass

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)

    # Add step information
    step = frame * 10
    ax_hidden.set_title(f"Learned Features in Hidden Space (Step {step})")
    ax_input.set_title(f"Input Space (Step {step})")
    ax_gradient.set_title(f"Gradient Visualization (Step {step})")
        
    # TODO: Plot hidden features
    hidden_features = mlp.hidden_activations
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y.ravel(), cmap='bwr', alpha=0.7
    )

    # TODO: Hyperplane visualization in the hidden space
    x_hidden = np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 50)
    y_hidden = np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 50)
    x_mesh, y_mesh = np.meshgrid(x_hidden, y_hidden)
    w1, w2, w3 = mlp.W2[:, 0]
    b = mlp.b2[0, 0]
    z_mesh = -(w1 * x_mesh + w2 * y_mesh + b) / (w3 + 1e-8)
    ax_hidden.plot_surface(
        x_mesh, y_mesh, z_mesh, color='wheat', alpha=0.5, edgecolor='none'
    )

    # TODO: Distorted input space transformed by the hidden layer
   
    

    # TODO: Plot input layer decision boundary
    xx = np.linspace(-3, 3, 100)
    yy = np.linspace(-3, 3, 100)
    XX, YY = np.meshgrid(xx, yy)
    input_points = np.column_stack((XX.ravel(), YY.ravel()))
    hidden_transformed = np.tanh(np.dot(input_points, mlp.W1) + mlp.b1) 
    Z = mlp.forward(input_points).reshape(XX.shape)
    ax_input.contourf(XX, YY, Z, levels=1, colors=['blue', 'red'], alpha=0.2)
    ax_input.scatter(X[y.ravel() == -1, 0], X[y.ravel() == -1, 1], 
                    c='blue', alpha=0.7, s=20)
    ax_input.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], 
                    c='red', alpha=0.7, s=20)
    ax_input.set_xlim(-3, 3)
    ax_input.set_ylim(-2, 2)


    # TODO: Visualize features and gradients as nodes and edges 
    # The edge thickness visually represents the magnitude of the gradient
    # Gradient visualization
    pos = {
        'x1': (0.0, 0.0),
        'x2': (0.0, 1.0),
        'h1': (0.4, 0.0),
        'h2': (0.4, 0.5),
        'h3': (0.4, 1.0),
        'y': (1.0, 0.0)
    }

    # Plot nodes
    for name, (x, y) in pos.items():
        circle = Circle((x, y), 0.05, color='blue', alpha=1.0)
        ax_gradient.add_patch(circle)
        ax_gradient.text(x-0.02, y+0.05, name, fontsize=10)

    # Plot edges with gradient-based thickness
    grad_w1, grad_w2 = mlp.gradients
    # Input to hidden connections
    for i_pos in [pos['x1'], pos['x2']]:
        for h_pos in [pos['h1'], pos['h2'], pos['h3']]:
            width = 5.0 * grad_w1 / (grad_w1.max() + 1e-8)
            ax_gradient.plot([i_pos[0], h_pos[0]], [i_pos[1], h_pos[1]], 
                           'purple', alpha=0.3, linewidth=width)

    # Hidden to output connections
    for h_pos in [pos['h1'], pos['h2'], pos['h3']]:
        width = 5.0 * grad_w2 / (grad_w2.max() + 1e-8)
        ax_gradient.plot([h_pos[0], pos['y'][0]], [h_pos[1], pos['y'][1]], 
                        'purple', alpha=0.6, linewidth=width)

    ax_gradient.set_xlim(-0.2, 1.2)
    ax_gradient.set_ylim(-0.2, 1.2)
    ax_gradient.set_aspect('equal')
    ax_gradient.axis('off')


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)