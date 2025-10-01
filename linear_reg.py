import numpy as np
import pandas as pd
import torch as t
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from data_process import load_dataset_pd, split_training_test, split_xy

# Dont change
rng = np.random.default_rng(42)
# Dont change
def generate_scatterplot(x: np.ndarray, y: np.ndarray, x_label: str, y_label: str) -> None:
    '''
    TODO:
    This function generates and displays a scatterplot of the provided x and y data points.
    The plot should be displayed using matplotlib.
    This is necessary for completing step #2 in the assignment handout.
    '''
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Scatterplot of Data Points")
    plt.show()

def train_model(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    '''
    TODO:
    Find the function that best fits the data using the LinearRegression model from sklearn.
    This is necessary for completing step #3 in the assignment handout.
    '''
    model = LinearRegression()
    # Reshape x to 2D if it's 1D (sklearn requires 2D input)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    model.fit(x, y)
    return model
    
def calc_rmse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    '''
    TODO:
    Calculate and return the RMSE (Root Mean Squared Error) between predicted values (y_hat) and true values (y_true).
    Refer to the notes in the assignment handout for the RMSE formula.
    This is necessary for completing step #5 in the assignment handout.
    '''
    rmse = np.sqrt(np.mean((y_hat - y_true) ** 2))
    return rmse

def pretty_print_poly(intercept: np.ndarray, coefs: np.ndarray) -> None:
    '''
    Pretty prints the polynomial equation in the form: y = w_0 + w_1 * x_1 + ... + w_n * x_n
    given the intercept and coefficients.
    This is a helper function to visualize the learned model.
    This is necessary for completing step #4 in the assignment handout.
    '''
    terms = [f"{intercept[0]:.4f}"]
    for i, coef in enumerate(coefs):
        terms.append(f"{coef:.4f} * x_{i+1}")
    equation = " + ".join(terms)
    print(f"y = {equation}")

def generate_weights(x: np.ndarray, y: np.ndarray):
    '''
    TODO:
    Generate weights using the Normal Equation: w = (X^T * X)^(-1) * X^T * y
    This is necessary for completing step #5 in the assignment handout.
    '''
    gram = x.T @ x
    gram_inv = np.linalg.inv(gram)
    xTy = x.T @ y
    return gram_inv @ xTy

if __name__ == "__main__":

    # TODO: Load the training data from step #1 in the assignment handout
    #       and store it in the variables x and y.
    # This is necessary for completing step #1 in the assignment handout.
    # Hint: You can use numpy arrays to store the data.
    x = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0])
    y = np.array([17.5, 12.9, 9.5, 7.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 24.4])
    
    # Section Generating Scatterplot
    generate_scatterplot(x, y, "x", "y")

    # TODO: Train the model using the training data (x, y) and print the polynomial equation.
    # This is necessary for completing step #3/#4 in the assignment handout.
    model = train_model(x, y)
    pretty_print_poly(model.intercept_, model.coef_)

    # TODO:
    # Generate and predict new y values for 5 random (x) samples in the range [-5.0, 5.0]
    # Calculate and print the RMSE of the model on the random sample data.
    # This is necessary for completing step #6 in the assignment handout.
    random_X_data = np.random.uniform(-5.0, 5.0, (5, 1)) # Generate 5 random samples in the range [-5.0, 5.0]
    random_Y_data = np.array([[x[0]**2] for x in random_X_data]) # Generate 5 random true y values (you can use the same function as in the prior line)
    y_hat = model.predict(random_X_data) # Use the model to predict y values for new_samples
    print("Predicted values:", y_hat)
    print("RMSE:", calc_rmse(y_hat, random_Y_data))

    # Load the dataset from GasProperties.csv
    dataset = load_dataset_pd("GasProperties.csv")
    # Preprocess the dataset to separate features and target variable
    X, Y = split_xy(dataset)
    X_train, Y_train, X_test, Y_test = split_training_test(X,Y)

    # Generate weights using the training data
    weights = generate_weights(X_train, Y_train)

    # Predict y values for the test set
    y_hat_train = weights @ X_train.T
    # Print Training RMSE for the gas dataset
    print(f"Training RMSE: {calc_rmse(y_hat_train, Y_train)}")

    # Predict y values for the test set
    y_hat_test = weights @ X_test.T
    # Print Testing RMSE for the gas dataset
    print(f"Testing RMSE: {calc_rmse(y_hat_test, Y_test)}")