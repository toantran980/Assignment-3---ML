import numpy as np
import pandas as pd
import torch as t
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def basic_mlp_setup() -> MLPClassifier:
    '''
    TODO:
    Create and return a simple MLPClassifier with ONE hidden layer of 3 neurons.
    Do not change the number of layers or neurons in this function.
    You may tweak hyperparameters like max_iter to ensure convergence.
    '''
    return MLPClassifier(hidden_layer_sizes=(3,), max_iter=3000, random_state=42)

def advanced_mlp_setup() -> MLPClassifier:
    '''
    TODO:
    Create and return an MLPClassifier with the same architecture as the basic version,
    BUT improve performance by tuning hyperparameters such as activation function, solver,
    alpha, learning rate, etc. MAKE SURE hidden_layer_sizes matches the one in basic_mlp_setup().
    '''
    return MLPClassifier(
        hidden_layer_sizes=(3,),
        activation= "relu",
        solver= "adam",
        alpha= 0.001,
        max_iter= 3000,
        learning_rate= "adaptive",
        random_state=42
    )

def compute_performance(clf: MLPClassifier, 
                        X_train: np.ndarray, 
                        X_test: np.ndarray, 
                        y_train: np.ndarray, 
                        y_test: np.ndarray) -> tuple[float, float]:
    '''
    TODO:
    Train the provided MLPClassifier on the training data, then compute and return
    both the training and testing accuracy as a tuple.

    Steps:
    1. Fit the classifier on the training data.
    2. Compute the accuracy on the training set.
    3. Compute the accuracy on the testing set.
    4. Return both accuracies as (train_score, test_score).
    '''
    
    # Step 1: Train the classifier
    clf.fit(X_train, y_train)

    # Step 2: Compute training accuracy
    train_score = clf.score(X_train, y_train)

    # Step 3: Compute testing accuracy
    test_score = clf.score(X_test, y_test)

    # Step 4: Return the two scores
    return train_score, test_score


if __name__ == "__main__":
    # TODO:
    # Load the Iris Flower dataset using sklearn.datasets.load_iris
    # This is a classification dataset with 150 samples and 3 classes.
    X, y = load_iris(return_X_y=True)

    # TODO:
    # Split the dataset into training (80%) and testing (20%) sets
    # Use sklearn's train_test_split with random_state=42 for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the basic MLP model
    bsc_clf = basic_mlp_setup()

    print("\nBasic MLP Performance:")
    bsc_train_score, bsc_test_score = compute_performance(bsc_clf, X_train, X_test, y_train, y_test)
    print(f"Training Accuracy: {bsc_train_score}")
    print(f"Testing Accuracy: {bsc_test_score}")

    # Train the advanced MLP model (with improved hyperparameters)
    adv_clf = advanced_mlp_setup()

    print("\nAdvanced MLP Performance:")
    adv_train_score, adv_test_score = compute_performance(adv_clf, X_train, X_test, y_train, y_test)
    print(f"Training Accuracy: {adv_train_score}")
    print(f"Testing Accuracy: {adv_test_score}")