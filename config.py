# config.py
# Configuration management for PiSentry project
# Centralizes all hyperparameters, thresholds, and system settings

import yaml
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    # Decision Tree parameters
    decision_tree_max_depth: int = 8
    decision_tree_min_samples_split: int = 5
    decision_tree_min_samples_leaf: int = 2
    decision_tree_criterion: str = 'gini'
    
    # Naive Bayes parameters
    naive_bayes_alpha: float = 0.1
    
    # General ML parameters
    random_state: int = 42
    cv_folds: int = 5
    
    # Feature selection
    num_selected_features: int = 25
    feature_selection_method: str = 'chi2'
    feature_selection_k: int = 25

@dataclass
class DataConfig:
    """Configuration for data processing."""
    # Dataset parameters
    num_samples: int = 1000
    num_base_features: int = 100
    test_size: float = 0.35
    
    # Data composition
    benign_ratio: float = 0.65
    malicious_ratio: float = 0.35
    
    # Temporal aggregation
    flow_window_seconds: int = 60
    
    # Data quality
    noise_level: float = 0.1
    min_feature_value: float = 0.0

@dataclass
class PerformanceConfig:
    """Configuration for performance targets and thresholds."""
    # Primary targets from paper
    target_f1_score: float = 0.92
    max_memory_mb: int = 10
    max_cpu_percent: int = 20
    max_inference_ms: int = 250
    
    # Extended performance metrics
    target_accuracy: float = 0.90
    max_false_positive_rate: float = 0.05
    target_precision: float = 0.90
    target_recall: float = 0.90
    
    # Resource monitoring
    monitoring_interval_seconds: float = 1.0
    max_monitoring_history: int = 1000
    
    # Performance warnings
    memory_warning_threshold_mb: int = 15
    cpu_warning_threshold_percent: int = 30
    inference_warning_threshold_ms: int = 100

@dataclass
class SystemConfig:
    """Configuration for system-level settings."""
    # Logging
    log_level: str = 'INFO'
    log_file: str = 'pisentry.log'
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    
    # File paths
    model_save_directory: str = './models'
    data_save_directory: str = './data'
    results_save_directory: str = './results'
    
    # Model file names
    decision_tree_model_file: str = 'pruned_decision_tree_model.joblib'
    naive_bayes_model_file: str = 'multinomial_naive_bayes_model.joblib'
    feature_selector_file: str = 'feature_selector.joblib'
    training_metadata_file: str = 'training_metadata.joblib'
    
    # Simulation parameters
    simulation_flow_interval: float = 0.5
    simulation_duration_minutes: int = 5
    simulation_malicious_probability: float = 0.15

@dataclass
class PiSentryConfig:
    """Main configuration class combining all settings."""
    model: ModelConfig
    data: DataConfig
    performance: PerformanceConfig
    system: SystemConfig
    
    def __init__(self, model=None, data=None, performance=None, system=None):
        self.model = model or ModelConfig()
        self.data = data or DataConfig()
        self.performance = performance or PerformanceConfig()
        self.system = system or SystemConfig()
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Create config objects from dictionaries
            model_config = ModelConfig(**config_dict.get('model', {}))
            data_config = DataConfig(**config_dict.get('data', {}))
            performance_config = PerformanceConfig(**config_dict.get('performance', {}))
            system_config = SystemConfig(**config_dict.get('system', {}))
            
            return cls(model_config, data_config, performance_config, system_config)
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            logger.info("Using default configuration")
            return cls()
    
    @classmethod
    def from_json(cls, config_path: str):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            model_config = ModelConfig(**config_dict.get('model', {}))
            data_config = DataConfig(**config_dict.get('data', {}))
            performance_config = PerformanceConfig(**config_dict.get('performance', {}))
            system_config = SystemConfig(**config_dict.get('system', {}))
            
            return cls(model_config, data_config, performance_config, system_config)
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            logger.info("Using default configuration")
            return cls()
    
    def save_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        try:
            config_dict = {
                'model': asdict(self.model),
                'data': asdict(self.data),
                'performance': asdict(self.performance),
                'system': asdict(self.system)
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {str(e)}")
    
    def save_json(self, config_path: str):
        """Save configuration to JSON file."""
        try:
            config_dict = {
                'model': asdict(self.model),
                'data': asdict(self.data),
                'performance': asdict(self.performance),
                'system': asdict(self.system)
            }
            
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {str(e)}")
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section, section_updates in updates.items():
            if hasattr(self, section) and isinstance(section_updates, dict):
                config_obj = getattr(self, section)
                for key, value in section_updates.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
                    else:
                        logger.warning(f"Unknown config parameter: {section}.{key}")
            else:
                logger.warning(f"Unknown config section: {section}")
    
    def validate(self):
        """Validate configuration parameters."""
        issues = []
        
        # Validate model parameters
        if self.model.decision_tree_max_depth < 1:
            issues.append("decision_tree_max_depth must be >= 1")
        
        if self.model.naive_bayes_alpha <= 0:
            issues.append("naive_bayes_alpha must be > 0")
        
        if self.model.cv_folds < 2:
            issues.append("cv_folds must be >= 2")
        
        # Validate data parameters
        if self.data.test_size <= 0 or self.data.test_size >= 1:
            issues.append("test_size must be between 0 and 1")
        
        if abs(self.data.benign_ratio + self.data.malicious_ratio - 1.0) > 0.01:
            issues.append("benign_ratio + malicious_ratio must equal 1.0")
        
        if self.data.num_samples < 100:
            issues.append("num_samples should be >= 100 for reliable results")
        
        # Validate performance parameters
        if self.performance.target_f1_score <= 0 or self.performance.target_f1_score > 1:
            issues.append("target_f1_score must be between 0 and 1")
        
        if self.performance.max_memory_mb <= 0:
            issues.append("max_memory_mb must be > 0")
        
        if self.performance.max_cpu_percent <= 0 or self.performance.max_cpu_percent > 100:
            issues.append("max_cpu_percent must be between 0 and 100")
        
        # Validate system parameters
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.system.log_level not in valid_log_levels:
            issues.append(f"log_level must be one of {valid_log_levels}")
        
        if issues:
            logger.error("Configuration validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("✓ Configuration validation passed")
        return True
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get parameters for specific model type."""
        if model_type.lower() == 'decision_tree':
            return {
                'max_depth': self.model.decision_tree_max_depth,
                'min_samples_split': self.model.decision_tree_min_samples_split,
                'min_samples_leaf': self.model.decision_tree_min_samples_leaf,
                'criterion': self.model.decision_tree_criterion,
                'random_state': self.model.random_state
            }
        elif model_type.lower() == 'naive_bayes':
            return {
                'alpha': self.model.naive_bayes_alpha
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_file_path(self, file_type: str) -> str:
        """Get full file path for specific file type."""
        base_paths = {
            'decision_tree_model': os.path.join(self.system.model_save_directory, self.system.decision_tree_model_file),
            'naive_bayes_model': os.path.join(self.system.model_save_directory, self.system.naive_bayes_model_file),
            'feature_selector': os.path.join(self.system.model_save_directory, self.system.feature_selector_file),
            'training_metadata': os.path.join(self.system.model_save_directory, self.system.training_metadata_file)
        }
        
        if file_type not in base_paths:
            raise ValueError(f"Unknown file type: {file_type}")
        
        return base_paths[file_type]
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.system.model_save_directory,
            self.system.data_save_directory,
            self.system.results_save_directory
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

# Global configuration instance
_global_config = None

def get_config() -> PiSentryConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_default_config()
    return _global_config

def set_config(config: PiSentryConfig):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config

def load_default_config() -> PiSentryConfig:
    """Load default configuration, checking for config files."""
    # Check for config files in order of preference
    config_files = [
        'pisentry_config.yaml',
        'pisentry_config.yml', 
        'config.yaml',
        'config.yml',
        'pisentry_config.json',
        'config.json'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            logger.info(f"Loading configuration from {config_file}")
            
            if config_file.endswith('.json'):
                return PiSentryConfig.from_json(config_file)
            else:
                return PiSentryConfig.from_yaml(config_file)
    
    # No config file found, use defaults
    logger.info("No configuration file found, using defaults")
    return PiSentryConfig()

def create_default_config_file(filename: str = 'pisentry_config.yaml'):
    """Create a default configuration file with comments."""
    config = PiSentryConfig()
    
    yaml_content = f"""# PiSentry Configuration File
# This file contains all configurable parameters for the PiSentry project

# Machine Learning Model Configuration
model:
  # Decision Tree parameters
  decision_tree_max_depth: {config.model.decision_tree_max_depth}  # Maximum tree depth to prevent overfitting
  decision_tree_min_samples_split: {config.model.decision_tree_min_samples_split}  # Minimum samples required to split node
  decision_tree_min_samples_leaf: {config.model.decision_tree_min_samples_leaf}  # Minimum samples required in leaf node
  decision_tree_criterion: "{config.model.decision_tree_criterion}"  # Split quality measure: 'gini' or 'entropy'
  
  # Naive Bayes parameters
  naive_bayes_alpha: {config.model.naive_bayes_alpha}  # Smoothing parameter
  
  # General parameters
  random_state: {config.model.random_state}  # Random seed for reproducibility
  cv_folds: {config.model.cv_folds}  # Number of cross-validation folds
  
  # Feature selection
  num_selected_features: {config.model.num_selected_features}  # Number of features to select
  feature_selection_method: "{config.model.feature_selection_method}"  # Feature selection method
  feature_selection_k: {config.model.feature_selection_k}  # Number of features for SelectKBest

# Data Processing Configuration
data:
  # Dataset size
  num_samples: {config.data.num_samples}  # Total number of samples to generate
  num_base_features: {config.data.num_base_features}  # Number of features before selection
  test_size: {config.data.test_size}  # Proportion of data for testing
  
  # Class distribution
  benign_ratio: {config.data.benign_ratio}  # Proportion of benign samples
  malicious_ratio: {config.data.malicious_ratio}  # Proportion of malicious samples
  
  # Temporal processing
  flow_window_seconds: {config.data.flow_window_seconds}  # Network flow aggregation window
  
  # Data quality
  noise_level: {config.data.noise_level}  # Amount of noise to add to features
  min_feature_value: {config.data.min_feature_value}  # Minimum feature value (for chi-square)

# Performance Targets and Thresholds
performance:
  # Primary targets from research paper
  target_f1_score: {config.performance.target_f1_score}  # Minimum F1 score target
  max_memory_mb: {config.performance.max_memory_mb}  # Maximum memory usage (MB)
  max_cpu_percent: {config.performance.max_cpu_percent}  # Maximum CPU usage (%)
  max_inference_ms: {config.performance.max_inference_ms}  # Maximum inference time (ms)
  
  # Extended metrics
  target_accuracy: {config.performance.target_accuracy}  # Minimum accuracy target
  max_false_positive_rate: {config.performance.max_false_positive_rate}  # Maximum false positive rate
  target_precision: {config.performance.target_precision}  # Minimum precision target
  target_recall: {config.performance.target_recall}  # Minimum recall target
  
  # Monitoring configuration
  monitoring_interval_seconds: {config.performance.monitoring_interval_seconds}  # Resource monitoring interval
  max_monitoring_history: {config.performance.max_monitoring_history}  # Maximum monitoring samples to keep
  
  # Warning thresholds
  memory_warning_threshold_mb: {config.performance.memory_warning_threshold_mb}  # Memory usage warning
  cpu_warning_threshold_percent: {config.performance.cpu_warning_threshold_percent}  # CPU usage warning
  inference_warning_threshold_ms: {config.performance.inference_warning_threshold_ms}  # Inference time warning

# System Configuration
system:
  # Logging configuration
  log_level: "{config.system.log_level}"  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: "{config.system.log_file}"  # Log file name
  enable_file_logging: {str(config.system.enable_file_logging).lower()}  # Enable file logging
  enable_console_logging: {str(config.system.enable_console_logging).lower()}  # Enable console logging
  
  # Directory structure
  model_save_directory: "{config.system.model_save_directory}"  # Directory for saved models
  data_save_directory: "{config.system.data_save_directory}"  # Directory for data files
  results_save_directory: "{config.system.results_save_directory}"  # Directory for results
  
  # Model file names
  decision_tree_model_file: "{config.system.decision_tree_model_file}"
  naive_bayes_model_file: "{config.system.naive_bayes_model_file}"
  feature_selector_file: "{config.system.feature_selector_file}"
  training_metadata_file: "{config.system.training_metadata_file}"
  
  # Simulation parameters
  simulation_flow_interval: {config.system.simulation_flow_interval}  # Seconds between simulated flows
  simulation_duration_minutes: {config.system.simulation_duration_minutes}  # Default simulation duration
  simulation_malicious_probability: {config.system.simulation_malicious_probability}  # Probability of malicious flow
"""
    
    try:
        with open(filename, 'w') as f:
            f.write(yaml_content)
        print(f"✓ Default configuration file created: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error creating config file {filename}: {str(e)}")
        return None

def setup_logging(config: PiSentryConfig):
    """Setup logging based on configuration."""
    # Configure logging level
    log_level = getattr(logging, config.system.log_level.upper(), logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler if enabled
    if config.system.enable_console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if config.system.enable_file_logging:
        try:
            file_handler = logging.FileHandler(config.system.log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {str(e)}")
    
    logger.info("Logging configured successfully")

def print_config_summary(config: PiSentryConfig):
    """Print a summary of the current configuration."""
    print("="*60)
    print("PISENTRY CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"\nModel Configuration:")
    print(f"  Decision Tree Max Depth: {config.model.decision_tree_max_depth}")
    print(f"  Naive Bayes Alpha: {config.model.naive_bayes_alpha}")
    print(f"  Feature Selection: {config.model.feature_selection_method} (k={config.model.num_selected_features})")
    print(f"  Cross-validation Folds: {config.model.cv_folds}")
    
    print(f"\nData Configuration:")
    print(f"  Dataset Size: {config.data.num_samples} samples")
    print(f"  Feature Count: {config.data.num_base_features} → {config.model.num_selected_features}")
    print(f"  Class Distribution: {config.data.benign_ratio:.1%} benign, {config.data.malicious_ratio:.1%} malicious")
    print(f"  Train/Test Split: {(1-config.data.test_size):.1%}/{config.data.test_size:.1%}")
    
    print(f"\nPerformance Targets:")
    print(f"  F1 Score: ≥{config.performance.target_f1_score}")
    print(f"  Memory Usage: ≤{config.performance.max_memory_mb} MB")
    print(f"  CPU Usage: ≤{config.performance.max_cpu_percent}%")
    print(f"  Inference Time: ≤{config.performance.max_inference_ms} ms")
    
    print(f"\nSystem Configuration:")
    print(f"  Log Level: {config.system.log_level}")
    print(f"  Model Directory: {config.system.model_save_directory}")
    print(f"  Results Directory: {config.system.results_save_directory}")
    
    print("="*60)

# Preset configurations for different scenarios

def get_raspberry_pi_config() -> PiSentryConfig:
    """Get configuration optimized for Raspberry Pi Zero deployment."""
    config = PiSentryConfig()
    
    # More aggressive resource constraints for Pi Zero
    config.performance.max_memory_mb = 8
    config.performance.max_cpu_percent = 15
    config.performance.max_inference_ms = 100
    
    # Smaller model for better performance
    config.model.decision_tree_max_depth = 6
    config.model.num_selected_features = 15
    
    # Smaller dataset for faster training
    config.data.num_samples = 800
    config.data.num_base_features = 50
    
    return config

def get_development_config() -> PiSentryConfig:
    """Get configuration optimized for development and testing."""
    config = PiSentryConfig()
    
    # More relaxed constraints for development
    config.performance.max_memory_mb = 50
    config.performance.max_cpu_percent = 50
    
    # Larger dataset for better model quality
    config.data.num_samples = 2000
    config.data.num_base_features = 150
    
    # More detailed logging
    config.system.log_level = 'DEBUG'
    
    # Faster simulation for testing
    config.system.simulation_duration_minutes = 2
    config.system.simulation_flow_interval = 0.1
    
    return config

def get_high_performance_config() -> PiSentryConfig:
    """Get configuration for high-performance deployment."""
    config = PiSentryConfig()
    
    # Higher performance targets
    config.performance.target_f1_score = 0.95
    config.performance.target_accuracy = 0.95
    config.performance.max_false_positive_rate = 0.02
    
    # More sophisticated model
    config.model.decision_tree_max_depth = 12
    config.model.num_selected_features = 40
    
    # Larger training dataset
    config.data.num_samples = 5000
    config.data.num_base_features = 200
    
    # More thorough validation
    config.model.cv_folds = 10
    
    return config

if __name__ == "__main__":
    print("🔧 PiSentry Configuration Manager")
    print("="*50)
    
    # Create and validate default config
    config = PiSentryConfig()
    
    print("Default Configuration:")
    print_config_summary(config)
    
    # Validate configuration
    if config.validate():
        print("\n✓ Configuration is valid")
    else:
        print("\n✗ Configuration validation failed")
    
    # Create default config file
    config_file = create_default_config_file()
    if config_file:
        print(f"\n✓ Created default configuration file: {config_file}")
        print("You can edit this file to customize PiSentry settings")
    
    # Test different preset configurations
    print("\n" + "="*50)
    print("PRESET CONFIGURATIONS")
    print("="*50)
    
    presets = {
        'Raspberry Pi Zero': get_raspberry_pi_config(),
        'Development': get_development_config(),
        'High Performance': get_high_performance_config()
    }
    
    for preset_name, preset_config in presets.items():
        print(f"\n{preset_name} Configuration:")
        print(f"  Max Memory: {preset_config.performance.max_memory_mb} MB")
        print(f"  Max CPU: {preset_config.performance.max_cpu_percent}%")
        print(f"  Tree Depth: {preset_config.model.decision_tree_max_depth}")
        print(f"  Features: {preset_config.model.num_selected_features}")
        print(f"  Dataset Size: {preset_config.data.num_samples}")
        print(f"  F1 Target: {preset_config.performance.target_f1_score}")
    
    print("\n" + "="*50)
    print("To use a preset configuration in your code:")
    print("  from config import get_raspberry_pi_config")
    print("  config = get_raspberry_pi_config()")
    print("  set_config(config)")
    print("="*50)
    
