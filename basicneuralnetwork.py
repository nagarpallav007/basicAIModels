import numpy as np

# Input features
income = np.array([[73], [93], [89], [96], [73]])
age = np.array([[40], [53], [38], [47], [33]])
loan_amount = np.array([[1000], [1200], [1100], [1400], [700]])

# Combine the input features into one array
X = np.column_stack((income, age, loan_amount))

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    # Define the feedforward function
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    # Define the backpropagation function
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    # Define the predict function
    def predict(self, x):
        self.input = x
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

# Define the training data
y = np.array([[1], [0], [0], [1], [1]])

# Initialize the neural network
nn = NeuralNetwork(X, y)

# Train the neural network
for i in range(1500):
    nn.feedforward()
    nn.backprop()

# Use the neural network to predict the loan eligibility

# Take user input and make a prediction
age = int(input("Enter age: "))
income = int(input("Enter income: "))
student = int(input("Enter student status (0 for not a student, 1 for student): "))

# Prepare the input data as a numpy array
input_data = np.array([[age, income, student]])




#input_data = np.array([[73, 40, 1000]])
result = nn.predict(input_data)
if result >= 0.5:
    print("Loan eligible")
else:
    print("Loan not eligible")
