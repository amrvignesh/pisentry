# utils.py
# Enhanced utility functions for the PiSentry project
# Includes comprehensive data generation, feature selection, and resource monitoring

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def generate_comprehensive_features(flow_data):
    """
    Generate comprehensive network flow features as described in the paper.
    Extracts statistical features from network flow data.
    """
    features = {}
    
    # Basic flow characteristics
    features['total_fwd_packets'] = np.sum(flow_data[:, 0]) if flow_data.size > 0 else 0
    features['total_bwd_packets'] = np.sum(flow_data[:, 1]) if flow_data.shape[1] > 1 else 0
    features['flow_duration'] = np.max(flow_data[:, 2]) - np.min(flow_data[:, 2]) if flow_data.shape[1] > 2 else 0
    
    # Packet size statistics
    if flow_data.size > 0:
        features['fwd_pkt_len_mean'] = np.mean(flow_data[:, 3]) if flow_data.shape[1] > 3 else 0
        features['fwd_pkt_len_std'] = np.std(flow_data[:, 3]) if flow_data.shape[1] > 3 else 0
        features['bwd_pkt_len_mean'] = np.mean(flow_data[:, 4]) if flow_data.shape[1] > 4 else 0
        features['bwd_pkt_len_std'] = np.std(flow_data[:, 4]) if flow_data.shape[1] > 4 else 0
    
    # Flow inter-arrival time statistics
    if flow_data.size > 0:
        features['flow_iat_mean'] = np.mean(np.diff(flow_data[:, 2])) if flow_data.shape[1] > 2 and len(flow_data) > 1 else 0
        features['flow_iat_std'] = np.std(np.diff(flow_data[:, 2])) if flow_data.shape[1] > 2 and len(flow_data) > 1 else 0
    
    # Protocol and port statistics
    features['protocol_type'] = np.random.choice([6, 17, 1])  # TCP, UDP, ICMP
    features['src_port'] = np.random.randint(1024, 65535)
    features['dst_port'] = np.random.choice([80, 443, 22, 23, 53, 25])
    
    # Additional behavioral features
    features['fwd_psh_flags'] = np.random.randint(0, 10)
    features['bwd_psh_flags'] = np.random.randint(0, 5)
    features['fwd_urg_flags'] = np.random.randint(0, 3)
    features['fin_flag_cnt'] = np.random.randint(0, 5)
    features['syn_flag_cnt'] = np.random.randint(0, 10)
    features['rst_flag_cnt'] = np.random.randint(0, 3)
    features['ack_flag_cnt'] = np.random.randint(0, 20)
    
    # Timing features
    features['active_mean'] = np.random.uniform(0, 1000)
    features['active_std'] = np.random.uniform(0, 500)
    features['idle_mean'] = np.random.uniform(0, 5000)
    features['idle_std'] = np.random.uniform(0, 2000)
    
    # Byte transfer ratios
    total_fwd_bytes = features['total_fwd_packets'] * features['fwd_pkt_len_mean']
    total_bwd_bytes = features['total_bwd_packets'] * features['bwd_pkt_len_mean']
    total_bytes = total_fwd_bytes + total_bwd_bytes
    
    features['fwd_bytes_ratio'] = total_fwd_bytes / total_bytes if total_bytes > 0 else 0
    features['bwd_bytes_ratio'] = total_bwd_bytes / total_bytes if total_bytes > 0 else 0
    
    return features

def generate_simulated_data(num_samples=1000, num_base_features=100):
    """
    Generates a comprehensive simulated dataset with realistic network flow patterns.
    Creates benign and malicious traffic with distinct behavioral signatures.
    """
    print(f"Generating simulated dataset with {num_samples} samples...")
    
    # Generate base features for benign traffic (65% of data)
    num_benign = int(num_samples * 0.65)
    num_malicious = num_samples - num_benign
    
    # Benign traffic characteristics
    benign_features = []
    for _ in range(num_benign):
        # Simulate normal IoT device behavior
        flow_data = np.random.exponential(scale=50, size=(10, 5))  # Smaller, regular patterns
        features = generate_comprehensive_features(flow_data)
        
        # Add IoT-specific normal behavior
        features.update({
            'packet_rate': np.random.uniform(1, 50),  # Lower packet rates
            'connection_attempts': np.random.randint(1, 5),  # Few connections
            'unique_ports': np.random.randint(1, 10),  # Limited port usage
            'dns_queries': np.random.randint(0, 20),  # Some DNS activity
            'http_requests': np.random.randint(0, 50),  # Normal web traffic
        })
        benign_features.append(features)
    
    # Malicious traffic characteristics (Mirai/Bashlite patterns)
    malicious_features = []
    for i in range(num_malicious):
        if i < num_malicious // 2:
            # Mirai-like behavior
            flow_data = np.random.uniform(low=80, high=200, size=(50, 5))  # High activity
            features = generate_comprehensive_features(flow_data)
            
            features.update({
                'packet_rate': np.random.uniform(100, 1000),  # High packet rates
                'connection_attempts': np.random.randint(50, 500),  # Many scan attempts
                'unique_ports': np.random.randint(20, 100),  # Port scanning
                'dns_queries': np.random.randint(100, 1000),  # DNS amplification
                'http_requests': np.random.randint(0, 10),  # Less normal web traffic
                'telnet_attempts': np.random.randint(10, 100),  # Telnet brute force
                'ssh_attempts': np.random.randint(5, 50),  # SSH attacks
            })
        else:
            # Bashlite-like behavior
            flow_data = np.random.uniform(low=60, high=150, size=(30, 5))  # Medium-high activity
            features = generate_comprehensive_features(flow_data)
            
            features.update({
                'packet_rate': np.random.uniform(50, 500),  # Moderate packet rates
                'connection_attempts': np.random.randint(20, 200),  # Scanning activity
                'unique_ports': np.random.randint(15, 80),  # Port scanning
                'dns_queries': np.random.randint(50, 500),  # DNS activity
                'http_requests': np.random.randint(0, 5),  # Minimal web traffic
                'shell_commands': np.random.randint(5, 30),  # Shell exploitation
                'payload_downloads': np.random.randint(1, 10),  # Payload activity
            })
        
        malicious_features.append(features)
    
    # Combine all features into DataFrame
    all_features = benign_features + malicious_features
    
    # Ensure all feature dictionaries have the same keys
    all_keys = set()
    for feat_dict in all_features:
        all_keys.update(feat_dict.keys())
    
    # Fill missing keys with 0
    for feat_dict in all_features:
        for key in all_keys:
            if key not in feat_dict:
                feat_dict[key] = 0
    
    # Convert to DataFrame
    X_df = pd.DataFrame(all_features)
    
    # Create labels
    y = np.concatenate([
        np.zeros(num_benign),      # Benign = 0
        np.ones(num_malicious)     # Malicious = 1
    ])
    
    # Add realistic noise
    numeric_columns = X_df.select_dtypes(include=[np.number]).columns
    noise = np.random.normal(0, X_df[numeric_columns].std() * 0.1, X_df[numeric_columns].shape)
    X_df[numeric_columns] += noise
    
    # Ensure no negative values (required for chi-square test)
    X_df[numeric_columns] = X_df[numeric_columns].clip(lower=0)
    
    # Shuffle the data
    indices = np.random.permutation(len(X_df))
    X_df = X_df.iloc[indices].reset_index(drop=True)
    y = y[indices]
    
    print(f"Generated {len(X_df)} samples with {len(X_df.columns)} features")
    print(f"Benign samples: {num_benign}, Malicious samples: {num_malicious}")
    
    return X_df, pd.Series(y)

def apply_chi_square_feature_selection(X, y, k=25):
    """
    Apply Chi-square feature selection to reduce dimensionality.
    Returns the selected features and the selector object.
    """
    print(f"Applying Chi-square feature selection to select top {k} features...")
    
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Scale features to ensure non-negative values for chi-square test
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Apply chi-square feature selection
    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X_scaled, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Get feature scores for analysis
    feature_scores = selector.scores_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'score': feature_scores,
        'selected': selector.get_support()
    }).sort_values('score', ascending=False)
    
    print(f"Selected {X_selected.shape[1]} features from {X.shape[1]} original features")
    print("Top 10 most important features:")
    print(feature_importance.head(10)[['feature', 'score']])
    
    return X_selected, selector, selected_features, feature_importance

def get_train_test_split(X, y, test_size=0.35, random_state=42):
    """
    Splits the data into training and testing sets.
    The 65:35 split is based on the project specifications.
    """
    print(f"Splitting data into training and test sets ({int((1-test_size)*100)}:{int(test_size*100)})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training set class distribution: {np.bincount(y_train)}")
    print(f"Test set class distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def simulate_feature_extraction(flow_data):
    """
    Simulates the feature extraction process for a single network flow.
    In a real application, this would process raw packet data.
    """
    if isinstance(flow_data, np.ndarray) and flow_data.size > 0:
        # Ensure the data is properly shaped
        if len(flow_data.shape) == 1:
            flow_data = flow_data.reshape(1, -1)
        
        # Generate comprehensive features for this flow
        features = generate_comprehensive_features(flow_data)
        
        # Convert to the format expected by trained models
        # This would typically be the same feature vector used during training
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        return feature_vector.astype(float)
    else:
        # Return zeros if no valid data
        return np.zeros((1, 25))  # Assuming 25 selected features

def load_real_dataset(dataset_path, dataset_type='botiot'):
    """
    Load real network security datasets (Bot-IoT, TON-IoT, etc.)
    This is a placeholder for future implementation.
    """
    print(f"Loading {dataset_type} dataset from {dataset_path}...")
    
    # Placeholder implementation
    # In practice, this would:
    # 1. Parse CSV files from Bot-IoT or TON-IoT datasets
    # 2. Extract relevant features
    # 3. Apply temporal windowing
    # 4. Return processed features and labels
    
    raise NotImplementedError("Real dataset loading not yet implemented. Use generate_simulated_data() instead.")

def validate_feature_consistency(X_train, X_test):
    """
    Validate that training and test sets have consistent features.
    """
    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        if not X_train.columns.equals(X_test.columns):
            print("Warning: Training and test sets have different feature columns!")
            return False
    
    if X_train.shape[1] != X_test.shape[1]:
        print(f"Warning: Feature dimension mismatch - Train: {X_train.shape[1]}, Test: {X_test.shape[1]}")
        return False
    
    return True

def get_feature_names(num_features=25):
    """
    Generate meaningful feature names for visualization and analysis.
    """
    base_features = [
        'total_fwd_packets', 'total_bwd_packets', 'flow_duration',
        'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_mean', 'bwd_pkt_len_std',
        'flow_iat_mean', 'flow_iat_std', 'protocol_type', 'src_port', 'dst_port',
        'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'fin_flag_cnt',
        'syn_flag_cnt', 'rst_flag_cnt', 'ack_flag_cnt', 'active_mean',
        'active_std', 'idle_mean', 'idle_std', 'fwd_bytes_ratio', 'bwd_bytes_ratio'
    ]
    
    # Extend with generic names if needed
    if num_features > len(base_features):
        for i in range(len(base_features), num_features):
            base_features.append(f'feature_{i+1}')
    
    return base_features[:num_features]
    
