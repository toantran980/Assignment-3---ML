import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from mlp_classifier import (
    basic_mlp_setup,
    advanced_mlp_setup,
    compute_performance
)

class TestMLPClassifierAssignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load and split the dataset once for all tests
        cls.X, cls.y = load_iris(return_X_y=True)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

    def test_basic_mlp_structure(self):
        """Test that basic MLP has exactly one hidden layer with 3 neurons."""
        model = basic_mlp_setup()
        self.assertIsInstance(model, MLPClassifier)
        self.assertEqual(model.hidden_layer_sizes, (3,), "Basic MLP should have one hidden layer with 3 neurons.")

    def test_advanced_mlp_structure(self):
        """Test that advanced MLP has the same hidden layer size as basic, but may differ in hyperparameters."""
        basic_model = basic_mlp_setup()
        adv_model = advanced_mlp_setup()
        self.assertEqual(
            adv_model.hidden_layer_sizes,
            basic_model.hidden_layer_sizes,
            "Advanced MLP must have the same architecture (hidden_layer_sizes) as the basic version initially."
        )

    def test_compute_performance_output(self):
        """Test compute_performance returns a tuple of floats within valid accuracy range."""
        clf = basic_mlp_setup()
        train_score, test_score = compute_performance(clf, self.X_train, self.X_test, self.y_train, self.y_test)
        self.assertIsInstance(train_score, float)
        self.assertIsInstance(test_score, float)
        self.assertGreaterEqual(train_score, 0.0)
        self.assertLessEqual(train_score, 1.0)
        self.assertGreaterEqual(test_score, 0.0)
        self.assertLessEqual(test_score, 1.0)

    def test_advanced_model_performance_is_not_worse(self):
        """
        Test that the advanced MLP model performs at least as well as the basic one.
        Some tolerance is allowed, but this ensures students actually tuned hyperparameters.
        """
        basic_model = basic_mlp_setup()
        adv_model = advanced_mlp_setup()

        basic_train, basic_test = compute_performance(basic_model, self.X_train, self.X_test, self.y_train, self.y_test)
        adv_train, adv_test = compute_performance(adv_model, self.X_train, self.X_test, self.y_train, self.y_test)

        # Advanced model should improve or match at least one of the scores
        self.assertTrue(
            adv_train >= basic_train or adv_test >= basic_test,
            "Advanced MLP should improve training or testing accuracy over the basic MLP."
        )

    def test_dataset_split_size(self):
        """Check that the dataset is split correctly into 80/20 for training and testing."""
        total_samples = len(self.X)
        self.assertEqual(len(self.X_train), int(0.8 * total_samples))
        self.assertEqual(len(self.X_test), total_samples - len(self.X_train))

if __name__ == '__main__':
    unittest.main()
