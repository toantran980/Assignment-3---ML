import io
import sys
import unittest
import numpy as np
from sklearn.linear_model import LinearRegression

from linear_reg import (
    generate_scatterplot,
    train_model,
    calc_rmse,
    pretty_print_poly,
    generate_weights
)

# Sample training data used across tests
X = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0]).reshape(-1, 1)
Y = np.array([17.5, 12.9, 9.5, 7.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 24.4]).reshape(-1, 1)


class TestLinearRegressionAssignment(unittest.TestCase):
    """
    Unit tests for the linear_reg.py assignment.
    These tests validate correctness of model training, RMSE calculation,
    polynomial printing, weight generation, and plotting functions.
    """

    def test_train_model_output_type_and_shape(self):
        """
        Test that train_model returns a LinearRegression model with correct coefficient and intercept shapes.
        """
        model = train_model(X, Y)
        self.assertIsInstance(model, LinearRegression)
        self.assertEqual(model.coef_.shape, (1, 1))
        self.assertEqual(model.intercept_.shape, (1,))

    def test_train_model_expected_values(self):
        """
        Test that the model learns expected slope and intercept from the dataset (with ~1 decimal accuracy).
        """
        model = train_model(X, Y)
        expected_slope = 1.06
        expected_intercept = 11.95
        self.assertAlmostEqual(model.coef_[0][0], expected_slope, places=1)
        self.assertAlmostEqual(model.intercept_[0], expected_intercept, places=1)

    def test_calc_rmse_exact_prediction(self):
        """
        Test that RMSE is zero when predictions match the true values exactly.
        """
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        rmse = calc_rmse(y_pred, y_true)
        self.assertAlmostEqual(rmse, 0.0)

    def test_calc_rmse_with_error(self):
        """
        Test that RMSE is correctly calculated for predictions with error.
        """
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 2, 4])
        expected_rmse = np.sqrt(((1**2 + 0**2 + 1**2) / 3))  # sqrt(2/3)
        rmse = calc_rmse(y_pred, y_true)
        self.assertAlmostEqual(rmse, expected_rmse, places=5)

    def test_pretty_print_poly_output(self):
        """
        Test that the pretty_print_poly function prints the correct polynomial equation format.
        """
        intercept = np.array([2.5])
        coefs = np.array([[1.0, -2.0]])
        captured_output = io.StringIO()
        sys.stdout = captured_output
        pretty_print_poly(intercept, coefs)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue().strip()
        self.assertTrue(output.startswith("y = 2.50 + 1.00 * x_1 - 2.00 * x_2"))

    def test_generate_weights_vs_sklearn(self):
        """
        Test that weights generated using the normal equation closely match those learned by sklearn's LinearRegression.
        """
        weights = generate_weights(X, Y)  # shape (1, 1)
        model = LinearRegression().fit(X, Y)
        sklearn_weights = model.coef_.flatten()
        np.testing.assert_allclose(weights.flatten(), sklearn_weights, atol=1e-2)

    def test_generate_scatterplot_runs(self):
        """
        Test that generate_scatterplot runs without throwing any exceptions.
        """
        try:
            generate_scatterplot(X, Y, "x", "y")
        except Exception as e:
            self.fail(f"generate_scatterplot raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()