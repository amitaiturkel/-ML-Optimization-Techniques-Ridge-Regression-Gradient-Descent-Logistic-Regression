
## Getting Started

To get started with the project, follow these steps:

### Download the Project

Clone this repository to your local machine by running the following command in your terminal:

`git clone https://github.com/amitaiturkel/ML-Optimization-Techniques-Ridge-Regression-Gradient-Descent-Logistic-Regression.git`
go to ML-Optimization-Techniques-Ridge-Regression-Gradient-Descent-Logistic-Regression folder 

### Set Up the Virtual Environment

Activate a virtual environment to isolate the project's dependencies. If you're using `virtualenv`, you can create and activate the environment with the following commands:

`python3 -m venv .venv` 
or run `python -m venv .venv`
and then
`source .venv/bin/activate`


### Install Dependencies

Use `poetry` to install the project's dependencies:

`poetry install`

### GUI BUGS

If you encounter issues with the graphical user interface (GUI) while running the project through a virtual machine (VM), particularly related to `matplotlib.pyplot`, please note that the VM environment may have limitations or configurations that affect GUI rendering.

To resolve this issue, it's recommended to run the script directly on your local machine instead of through a VM. This should help ensure proper functionality and display of the GUI components.

To install the required dependencies witout poetry, use the following command:

`pip install matplotlib torch scikit-learn pandas numpy` 




### Exploring Ridge Regression - Analytical Solution

For this exploration, I'll delve into ridge regression, a regularization technique used in linear regression to prevent overfitting. Our dataset consists of 2D coordinates of cities in Europe, with labels indicating the corresponding countries. 

#### Approach:
1. **Implement Ridge Regression**: I'll implement the analytical solution for ridge regression, optimizing a linear regression model by minimizing the regularized loss function.
2. **Training the Classifier**: Using the provided skeleton file, I'll train the ridge regression classifier on the `train.csv` data with various choices of the regularization strength (λ) - 0., 2., 4., 6., 8., and 10.


#### Questions to Explore:
6.1 **Accuracies vs. Regularization Strength**: I'll plot the training, validation, and test accuracies of the models against their λ values to understand how regularization affects model performance. Additionally, I'll report the test accuracy of the best model based on the validation set.

6.2 **Visualizing Prediction Space**: Utilizing a visualization helper, I'll plot the prediction space of the best and worst λ values determined from the validation set, using the test points for visualization. This will help me understand how the λ parameter influences the algorithm.

### Gradient Descent in NumPy

In this section, I'll implement gradient descent for a simple function using NumPy to optimize a (x, y) vector and reach the minimum of the function.

#### Approach:
7.1 **Optimizing with Gradient Descent**: I'll implement gradient descent for the given function \( f(x, y) = (x - 3)^2 + (y - 5)^2 \), using a learning rate of 0.1 for 1000 iterations. I'll initialize the vector to be optimized as (0, 0).

#### Analysis:
I'll plot the optimized vector through the iterations to visualize the optimization process and observe which point the algorithm converges to. This will give us a simple demo of how Gradient Descent works, and I will show the result in section 7.1.

### PyTorch: A Powerful Library for Gradient-Based Methods

PyTorch is a versatile library for deep learning and gradient-based optimization. In this section, I'll explore its functionalities and how it can be used for training gradient-based models.


### Logistic Regression - Stochastic Gradient Descent

Moving forward, I'll focus on logistic regression, a classification algorithm, and implement it using PyTorch with stochastic gradient descent.

#### Approach:
9.1 **Implementing Logistic Regression**: Inside the `model.py` file, I'll implement a logistic regression classifier using PyTorch and train it on the `train.csv` data using stochastic gradient descent.
9.2 **Exploring Binary Case**: I'll train logistic regression classifiers with different learning rates (0.1, 0.01, 0.001) and analyze training accuracies, losses, and validation/test set performances.

#### Questions to Explore:
9.3 **Visualizing Model Performance**: I'll visualize test predictions, plot training/validation/test losses over training epochs, and compare the performance with the ridge regression model to determine which method works better for our dataset. I will report the results of the different learning rates in section 9 and will try to explain the results.

9.3.1 **Best Model Evaluation**: I'll visualize test predictions, plot training/validation/test losses over training epochs of the best model.

9.3.2 **Generalization Analysis**: In this section, I will try to explain if the best model generalizes well from the training data and why.

9.3.3 **Comparison between Models**: I will compare between our 2 models, the Ridge Regression model and the Logistic Regression model, and discuss their differences.


### Logistic Regression - Multi-Class Case

Finally, I'll extend logistic regression to the multi-class case and train classifiers on a multi-class dataset.

#### Approach:
9.4 **Training Multi-Class Classifier**: I'll train logistic regression classifiers on the multi-class data with various initial learning rates (0.01, 0.001, 0.0003) for 30 epochs with a batch size of 32, decaying the learning rate every 5 epochs.

#### Analysis:
9.4.1**Model Evaluation and Comparison**: I'll plot the test and validation accuracies of the model vs. their learning rate values, visualize training/validation/test losses over training epochs in section 9.4, i will find the best model and explain if it generalized well in section 9.4.1.

9.4.2 **Best Model Performance**: I'll plot the test and validation accuracies of the best model vs epochs and discuss if it generalizes well.

9.4.3 **Comparison with Decision Trees**: I will use the sklearn library to train a decision tree on the data with max_depth of 2, and compare logistic regression best model with decision trees to determine the most suitable model for our task.

9.4.4 **Deep Decision Trees Analysis**: After that, I will do the same thing with using max depth = 10. I will compare this decision tree with our best model and will try to answer the question, Which one is more suitable for this task, and what changed by changing the tree max depth?

Through these explorations, I aim to gain insights into different optimization techniques and their effectiveness in solving classification problems with our spatial dataset. Let's embark on this journey together!

