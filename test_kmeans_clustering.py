import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from kmeans_clustering import setup_kmeans, calc_kmeans_rmse, rmse_per_cluster

class TestKMeansClustering(unittest.TestCase):
    
    def setUp(self):
        """Load the Iris dataset and set up basic variables for tests."""
        self.X, self.y_true = load_iris(return_X_y=True)
        self.kmeans = setup_kmeans(self.X)
    
    def test_setup_kmeans(self):
        """Test if KMeans is correctly set up and fitted."""
        
        # Check if the KMeans model has 3 clusters
        self.assertEqual(self.kmeans.n_clusters, 3)
        
        # Check that the model has been fitted (i.e., it has labels and centroids)
        self.assertTrue(hasattr(self.kmeans, 'labels_'))
        self.assertTrue(hasattr(self.kmeans, 'cluster_centers_'))
        
        # Check that the labels match the number of samples
        self.assertEqual(len(self.kmeans.labels_), self.X.shape[0])
    
    def test_calc_kmeans_rmse(self):
        """Test if RMSE calculation based on inertia is correct."""
        
        # Compute the global RMSE using inertia
        rmse_global = calc_kmeans_rmse(self.kmeans, len(self.X))
        
        # Assert RMSE is a positive value
        self.assertGreater(rmse_global, 0)
        
        # The inertia should be a positive number
        self.assertGreater(self.kmeans.inertia_, 0)
        
        # Check if the calculated RMSE matches the formula: sqrt(inertia / X_size)
        expected_rmse = np.sqrt(self.kmeans.inertia_ / len(self.X))
        self.assertAlmostEqual(rmse_global, expected_rmse, places=4)
    
    def test_rmse_per_cluster(self):
        """Test if RMSE per cluster is calculated correctly."""
        
        # Get the RMSE per cluster
        rmses = rmse_per_cluster(self.X, self.kmeans.labels_, self.kmeans.cluster_centers_)
        
        # Assert there are exactly 3 RMS values, one for each cluster
        self.assertEqual(len(rmses), 3)
        
        # Each RMSE should be a non-negative number
        for rmse in rmses:
            self.assertGreaterEqual(rmse, 0)

    def test_ari_score(self):
        """Test if the Adjusted Rand Index (ARI) score is valid and computed correctly."""
        
        # Compute the ARI score using adjusted_rand_score from sklearn
        ari = adjusted_rand_score(self.y_true, self.kmeans.labels_)
        
        # ARI should be a value between -1 and 1
        self.assertGreaterEqual(ari, -1)
        self.assertLessEqual(ari, 1)

    
    def test_confusion_matrix(self):
        """Test if the confusion matrix has the correct dimensions."""
        
        # Compute the confusion matrix
        cm = confusion_matrix(self.y_true, self.kmeans.labels_)
        
        # There should be 3 classes (clusters and true labels)
        self.assertEqual(cm.shape, (3, 3))
    
    def test_full_clustering_pipeline(self):
        """Test if the full KMeans clustering pipeline works."""
        
        # Get RMSE values
        rmses = rmse_per_cluster(self.X, self.kmeans.labels_, self.kmeans.cluster_centers_)
        rmse_global = calc_kmeans_rmse(self.kmeans, len(self.X))
        
        # Compute ARI and confusion matrix
        ari = adjusted_rand_score(self.y_true, self.kmeans.labels_)
        cm = confusion_matrix(self.y_true, self.kmeans.labels_)
        
        # Ensure ARI is a positive value (ideally, ARI should be above 0 in this case)
        self.assertGreater(ari, 0)
        
        # Check that the confusion matrix is 3x3
        self.assertEqual(cm.shape, (3, 3))
        
        # Check that the cluster's RMSE values are reasonable (i.e., not negative)
        for rmse in rmses:
            self.assertGreaterEqual(rmse, 0)
        
        # Global RMSE should be greater than 0
        self.assertGreater(rmse_global, 0)

    def test_rmse_single_cluster(self):
        """Test if RMSE handles trivial clusters correctly (e.g., clusters with only one point)."""
        rmses = rmse_per_cluster(self.X, self.kmeans.labels_, self.kmeans.cluster_centers_)
        
        # If a cluster contains only one point, the RMSE should be 0
        for rmse in rmses:
            self.assertGreaterEqual(rmse, 0)


if __name__ == '__main__':
    unittest.main()