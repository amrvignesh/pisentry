# utils.py
# This script contains helper functions for the project,
# including simulated data generation and feature extraction.

import numpy as np
from sklearn.model_selection import train_test_split

def generate_simulated_data(num_samples=1000):
    """
    Generates a small, simulated dataset for training and testing.
    This replaces the need for real PCAP files for this proof-of-concept.
    """
    print("Generating simulated dataset...")
    # Simulated features (e.g., packet counts, byte ratios, etc.)
    # We create a clear separation between benign and malicious data points
    X_benign = np.random.randint(0, 50, size=(int(num_samples * 0.65), 25))
    X_malicious = np.random.randint(20, 100, size=(int(num_samples * 0.35), 25))
    X = np.concatenate((X_benign, X_malicious), axis=0)

    # Simulated labels
    y_benign = np.zeros(int(num_samples * 0.65))
    y_malicious = np.ones(int(num_samples * 0.35))
    y = np.concatenate((y_benign, y_malicious), axis=0)

    # Add some noise to make it more realistic
    X += np.random.normal(0, 5, X.shape)
    X[X < 0] = 0

    return X, y

def get_train_test_split(X, y):
    """
    Splits the data into training and testing sets.
    The 65:35 split is based on the project progress report.
    """
    print("Splitting data into training and test sets (65:35)...")
    return train_test_split(X, y, test_size=0.35, random_state=42)

def simulate_feature_extraction(flow_data):
    """
    Simulates the feature extraction process for a single network flow.
    In a real application, this would take raw packet data and produce
    a feature vector.
    """
    # For this simulation, we assume the input is already a feature vector
    # and we perform a simple type conversion.
    # In a real project, this function would use a library like Scapy
    # to process packets and calculate features.
    return flow_data.astype(int)

