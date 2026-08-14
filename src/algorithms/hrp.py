import numpy as np
import pandas as pd
from .base import BaseOptimizer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

class HRPOptimizer(BaseOptimizer):
    
    def _get_distance_matrix(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Builds the Surveyor's Grid (Distance Matrix).
        Step 1 of the HRP Robot.
        """
        # Brick 1: Get the correlation grid
        correlation = cov_matrix.corr()
        
        # Brick 2: Convert to pure numbers (NumPy array)
        correlation_array = correlation.to_numpy()
        
        # Brick 3: Apply the Pythagorean-like distance formula to every cell
        distance_array = np.sqrt(0.5 * (1 - correlation_array))
        
        # Brick 4: Reattach the labels (stock names) to the grid
        distance_matrix = pd.DataFrame(distance_array, 
                                       index=correlation.index, 
                                       columns=correlation.columns)
        return distance_matrix
    

    def _cluster(self, distance_matrix: pd.DataFrame) -> np.ndarray:
        """
        Builds the Family Tree (Dendrogram).
        Step 2 of the HRP Robot.
        """
        # Brick 1: Compress the Surveyor's Grid into a tidy list
        compressed_dist = squareform(distance_matrix.to_numpy())
        
        # Brick 2: Draw the circles and branches (build the tree)
        linkage_matrix = linkage(compressed_dist, method='ward')
        
        return linkage_matrix
    
    def _get_items(self, linkage_matrix: np.ndarray, node_id: int, n_assets: int) -> list:
        """
        Returns the list of original stock indices that belong to this node.
        """
        if node_id < n_assets:
            return [int(node_id)]
        
        row_idx = node_id - n_assets
        left = int(linkage_matrix[row_idx, 0])
        right = int(linkage_matrix[row_idx, 1])
        
        left_items = self._get_items(linkage_matrix, left, n_assets)
        right_items = self._get_items(linkage_matrix, right, n_assets)
        return left_items + right_items
    
    def _get_cluster_variance(self, cov_matrix: pd.DataFrame, cluster_indices: list) -> float:
        """
        Calculates the wobble (variance) of a specific branch.
        """
        # If it's just one stock, look up its variance directly
        if len(cluster_indices) == 1:
            return cov_matrix.iloc[cluster_indices[0], cluster_indices[0]]
        
        # Otherwise, grab the sub-section of the covariance grid for this branch
        sub_cov = cov_matrix.iloc[cluster_indices, cluster_indices]
        
        # Equal weight inside the branch (we assume every stock in this branch is equal for now)
        equal_weights = np.ones(len(cluster_indices)) / len(cluster_indices)
        
        # Calculate the variance: weight.T * covariance * weight
        variance = np.dot(equal_weights.T, np.dot(sub_cov, equal_weights))
        return variance
    
    def _bisect(self, cov_matrix: pd.DataFrame, linkage_matrix: np.ndarray, node_id: int, n_assets: int) -> dict:
        """
        Walks down the tree and splits money. 
        Returns a dictionary of {stock_index: weight}.
        """
        # Leaf node: This is an original stock
        if node_id < n_assets:
            return {int(node_id): 1.0}
        
        # Internal node: Find its two children in the linkage matrix
        row_idx = node_id - n_assets
        left_child = int(linkage_matrix[row_idx, 0])
        right_child = int(linkage_matrix[row_idx, 1])
        
        # Get the list of stocks in each child branch
        left_items = self._get_items(linkage_matrix, left_child, n_assets)
        right_items = self._get_items(linkage_matrix, right_child, n_assets)
        
        # Calculate the risk (variance) of each child branch
        left_var = self._get_cluster_variance(cov_matrix, left_items)
        right_var = self._get_cluster_variance(cov_matrix, right_items)
        
        # Risk Parity split: more money to the calmer branch
        total_var = left_var + right_var
        if total_var == 0:
            left_weight = 0.5
            right_weight = 0.5
        else:
            left_weight = (1 - left_var / total_var) / 1  # (right_var / total_var)
            right_weight = (1 - right_var / total_var) / 1
            
        # Recurse down both branches
        left_alloc = self._bisect(cov_matrix, linkage_matrix, left_child, n_assets)
        right_alloc = self._bisect(cov_matrix, linkage_matrix, right_child, n_assets)
        
        # Combine the results, scaling by the split weights
        combined = {}
        for k, v in left_alloc.items():
            combined[k] = v * left_weight
        for k, v in right_alloc.items():
            combined[k] = v * right_weight
        return combined
    

    def optimize(self, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        The main door to the HRP Robot.
        Runs all three steps and returns the final weights.
        """
        # Step 1: Surveyor builds the distance grid
        distance_matrix = self._get_distance_matrix(cov_matrix)
        
        # Step 2: Cartographer draws the family tree
        linkage_matrix = self._cluster(distance_matrix)
        
        # Step 3: Water Distributor walks the tree and splits money
        n_assets = len(cov_matrix)
        initial_node = 2 * n_assets - 2  # The root of the tree
        allocation_dict = self._bisect(cov_matrix, linkage_matrix, initial_node, n_assets)
        
        # Convert the dictionary to a numpy array in the correct stock order
        weights = np.array([allocation_dict[i] for i in sorted(allocation_dict.keys())])
        return weights