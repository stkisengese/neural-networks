# neural-networks
A hands-on implementation of foundational neural network concepts—from single neurons to multi-layer networks—for both classification and regression tasks. Built from scratch in NumPy, this project covers forward propagation, activation functions, loss computation (log loss & MSE), and architectural adaptations for different prediction tasks.

## Setup

To get started with this project, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/stkisengese/neural-networks.git
    cd neural-networks
    ```

2.  **Install dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Jupyter Notebooks:**

    ```bash
    jupyter notebook
    ```

    This will open a browser window where you can navigate to and run the `.ipynb` files.

## Features

*   **Single Neuron Implementation**: Understand the basic building block of a neural network, including the weighted sum of inputs, bias, and activation function.
*   **Neural Network Implementation**: Build a simple neural network with a hidden layer and an output layer.
*   **Forward Propagation**: See how data flows through the network to generate predictions.
*   **Activation Functions**: Explore the use of sigmoid for classification and linear activation for regression.
*   **Loss Functions**: Learn how to evaluate model performance with log loss for classification and mean squared error (MSE) for regression.
*   **Classification and Regression**: Apply the neural network to solve both classification (predicting exam success) and regression (predicting physics grades) problems.

## Usage

### Classification

Here's how to use the neural network for a classification task, such as predicting student exam success based on their math and chemistry scores.

```python
import numpy as np
from helpers import Neuron, OurNeuralNetwork

# Student data
# name, math, chemistry, exam_success
students_data = [
    ("Bob", 12, 15, 1),
    ("Eli", 10, 9, 0),
    ("Tom", 18, 18, 1),
    ("Ryan", 13, 14, 1)
]

# Initialize network
neuron_h1 = Neuron(0.05, 0.001, 0)
neuron_h2 = Neuron(0.02, 0.003, 0)
neuron_o1 = Neuron(2, 0, 0)

network = OurNeuralNetwork(neuron_h1, neuron_h2, neuron_o1)

# Compute predictions for each student
predictions = []
y_true = []

print("Student Predictions:")
print("-" * 50)
for name, math, chemistry, success in students_data:
    pred = network.feedforward(math, chemistry)
    predictions.append(pred)
    y_true.append(success)
    print(f"{name}: Math={math}, Chem={chemistry} -> Pred={pred:.4f}, Actual={success}")
```

### Regression

Here's how to adapt the neural network for a regression task, such as predicting a student's physics grade based on their math and chemistry scores.

```python
import numpy as np
from helpers import Neuron, OurNeuralNetwork

# Student data for regression
# name, math, chemistry, physics
students_data = [
    ("Bob", 12, 15, 16),
    ("Eli", 10, 9, 10),
    ("Tom", 18, 18, 19),
    ("Ryan", 13, 14, 16)
]

# Initialize network for regression
# Hidden layers: use sigmoid (classification activation)
# Output layer: use identity (regression activation)
neuron_h1 = Neuron(0.05, 0.001, 0, regression=False)
neuron_h2 = Neuron(0.002, 0.003, 0, regression=False)
neuron_o1 = Neuron(2, 7, 10, regression=True)

network = OurNeuralNetwork(neuron_h1, neuron_h2, neuron_o1)

# Compute predictions
predictions = []
y_true = []

print("\nStudent Physics Grade Predictions:")
print("-" * 60)
for name, math, chemistry, physics in students_data:
    pred = network.feedforward(math, chemistry)
    predictions.append(pred)
    y_true.append(physics)
    print(f"{name}: Math={math}, Chem={chemistry} -> Predicted={pred:.2f}, Actual={physics}")
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.