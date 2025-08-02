# inference_pi.py
# Enhanced real-time detection agent for PiSentry
# Simulates the "on-device" program that would be deployed to Raspberry Pi Zero
# Includes realistic network flow simulation and comprehensive monitoring

import joblib
import numpy as np
import pandas as pd
import time
import psutil
import logging
import threading
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from utils import simulate_feature_extraction, get_feature_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pisentry_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PiSentryAgent:
    """
    Main detection agent class for PiSentry.
    Handles model loading, resource monitoring, and real-time inference.
    """
    
    def __init__(self, model_path='pruned_decision_tree_model.joblib', 
                 feature_selector_path='feature_selector.joblib'):
        self.model = None
        self.feature_selector = None
        self.model_path = model_path
        self.feature_selector_path = feature_selector_path
        self.is_running = False
        self.stats = {
            'total_flows_processed': 0,
            'malicious_flows_detected': 0,
            'benign_flows_detected': 0,
            'false_alarms': 0,
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': []
        }
        self.process = psutil.Process()
        
        # Load models during initialization
        self.load_models()
    
    def load_models(self):
        """Load the pre-trained model and feature selector."""
        try:
            # Load the main detection model
            self.model = joblib.load(self.model_path)
            logger.info(f"✓ Model loaded successfully from '{self.model_path}'")
            
            # Load feature selector if available
            try:
                self.feature_selector = joblib.load(self.feature_selector_path)
                logger.info(f"✓ Feature selector loaded from '{self.feature_selector_path}'")
            except FileNotFoundError:
                logger.warning(f"Feature selector not found at '{self.feature_selector_path}'")
                logger.warning("Will proceed without feature selection")
            
            return True
            
        except FileNotFoundError:
            logger.error(f"Model file '{self.model_path}' not found!")
            logger.error("Please run 'python train.py' first to train the models.")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def monitor_resources(self):
        """Monitor current resource usage."""
        try:
            memory = self.process.memory_info().rss / 1024 / 1024  # MB
            cpu = self.process.cpu_percent()
            
            self.stats['memory_usage'].append(memory)
            self.stats['cpu_usage'].append(cpu)
            
            # Keep only last 100 measurements
            if len(self.stats['memory_usage']) > 100:
                self.stats['memory_usage'] = self.stats['memory_usage'][-100:]
                self.stats['cpu_usage'] = self.stats['cpu_usage'][-100:]
            
            return {
                'memory_mb': memory,
                'cpu_percent': cpu,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error monitoring resources: {str(e)}")
            return None
    
    def simulate_network_flow(self, flow_type='random'):
        """
        Simulate realistic network flow data.
        In a real deployment, this would read from actual network interfaces.
        """
        if flow_type == 'random':
            # Randomly choose flow type based on realistic probabilities
            is_malicious = np.random.choice([True, False], p=[0.15, 0.85])  # 15% malicious
        else:
            is_malicious = (flow_type == 'malicious')
        
        if is_malicious:
            # Simulate malicious flow patterns (Mirai/Bashlite-like)
            attack_type = np.random.choice(['mirai', 'bashlite'])
            
            if attack_type == 'mirai':
                # Mirai characteristics: high connection attempts, telnet scanning
                flow_data = np.random.uniform(80, 200, size=(1, 25))
                flow_data[0, :5] *= 2  # Amplify connection-related features
                
            else:  # bashlite
                # Bashlite characteristics: moderate activity, shell exploitation
                flow_data = np.random.uniform(60, 150, size=(1, 25))
                flow_data[0, 5:10] *= 1.5  # Amplify payload-related features
            
            # Add some noise but maintain malicious signature
            flow_data += np.random.normal(0, 10, flow_data.shape)
            flow_data = np.clip(flow_data, 0, None)  # Ensure non-negative
            
        else:
            # Simulate benign IoT device behavior
            device_type = np.random.choice(['sensor', 'camera', 'smart_speaker', 'router'])
            
            if device_type == 'sensor':
                # Low activity, periodic reporting
                flow_data = np.random.exponential(scale=20, size=(1, 25))
                
            elif device_type == 'camera':
                # Medium activity, video streaming
                flow_data = np.random.uniform(30, 80, size=(1, 25))
                
            elif device_type == 'smart_speaker':
                # Variable activity, voice processing
                flow_data = np.random.gamma(2, 25, size=(1, 25))
                
            else:  # router
                # Higher activity, routing traffic
                flow_data = np.random.uniform(40, 120, size=(1, 25))
            
            # Add realistic noise
            flow_data += np.random.normal(0, 5, flow_data.shape)
            flow_data = np.clip(flow_data, 0, None)
        
        return flow_data, is_malicious
    
    def extract_and_select_features(self, flow_data):
        """
        Extract features from flow data and apply feature selection.
        """
        try:
            # Simulate feature extraction
            features = simulate_feature_extraction(flow_data)
            
            # Apply feature selection if available
            if self.feature_selector is not None:
                features = self.feature_selector.transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            return np.zeros((1, 25))  # Return default feature vector
    
    def classify_flow(self, features):
        """
        Classify a network flow as benign or malicious.
        Returns prediction, confidence, and timing information.
        """
        start_time = time.time()
        
        try:
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Get prediction confidence if available
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = np.max(proba)
            elif hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(features)[0]
                confidence = abs(decision)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # milliseconds
            
            return {
                'prediction': int(prediction),
                'confidence': confidence,
                'processing_time_ms': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            return {
                'prediction': 0,  # Default to benign
                'confidence': 0.0,
                'processing_time_ms': 0.0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def log_detection(self, result, actual_label=None, flow_id=None):
        """
        Log detection results and update statistics.
        """
        prediction = result['prediction']
        confidence = result.get('confidence', 'N/A')
        processing_time = result['processing_time_ms']
        
        # Update statistics
        self.stats['total_flows_processed'] += 1
        self.stats['processing_times'].append(processing_time)
        
        if prediction == 1:
            self.stats['malicious_flows_detected'] += 1
        else:
            self.stats['benign_flows_detected'] += 1
        
        # Keep only last 1000 processing times
        if len(self.stats['processing_times']) > 1000:
            self.stats['processing_times'] = self.stats['processing_times'][-1000:]
        
        # Create log message
        flow_id_str = f"Flow-{flow_id}" if flow_id else "Flow"
        confidence_str = f" (confidence: {confidence:.3f})" if confidence != 'N/A' else ""
        
        if prediction == 1:  # Malicious
            if actual_label is not None and actual_label == 0:
                # False positive
                self.stats['false_alarms'] += 1
                logger.warning(f"🚨 FALSE ALARM: {flow_id_str} flagged as MALICIOUS{confidence_str} "
                             f"[{processing_time:.2f}ms] - Actually benign")
            else:
                logger.critical(f"🚨 ALERT: {flow_id_str} - MALICIOUS TRAFFIC DETECTED{confidence_str} "
                               f"[{processing_time:.2f}ms]")
        else:  # Benign
            logger.info(f"✓ {flow_id_str} - Benign traffic{confidence_str} [{processing_time:.2f}ms]")
        
        # Check performance targets
        if processing_time > 250:  # 250ms target from paper
            logger.warning(f"Performance warning: Processing time {processing_time:.2f}ms exceeds 250ms target")
    
    def get_performance_summary(self):
        """
        Generate a summary of agent performance and statistics.
        """
        if not self.stats['processing_times']:
            return "No flows processed yet."
        
        avg_processing_time = np.mean(self.stats['processing_times'])
        max_processing_time = np.max(self.stats['processing_times'])
        min_processing_time = np.min(self.stats['processing_times'])
        
        avg_memory = np.mean(self.stats['memory_usage']) if self.stats['memory_usage'] else 0
        avg_cpu = np.mean(self.stats['cpu_usage']) if self.stats['cpu_usage'] else 0
        
        detection_rate = (self.stats['malicious_flows_detected'] / 
                         self.stats['total_flows_processed'] * 100) if self.stats['total_flows_processed'] > 0 else 0
        
        false_alarm_rate = (self.stats['false_alarms'] / 
                           self.stats['total_flows_processed'] * 100) if self.stats['total_flows_processed'] > 0 else 0
        
        summary = f"""
=== PiSentry Agent Performance Summary ===
Total Flows Processed: {self.stats['total_flows_processed']}
Malicious Flows Detected: {self.stats['malicious_flows_detected']} ({detection_rate:.1f}%)
Benign Flows Detected: {self.stats['benign_flows_detected']}
False Alarms: {self.stats['false_alarms']} ({false_alarm_rate:.1f}%)

Processing Time Statistics:
  Average: {avg_processing_time:.2f} ms
  Maximum: {max_processing_time:.2f} ms  
  Minimum: {min_processing_time:.2f} ms

Resource Usage:
  Average Memory: {avg_memory:.1f} MB
  Average CPU: {avg_cpu:.1f}%

Performance Targets:
  Processing Time Target (<250ms): {'✓ PASS' if avg_processing_time < 250 else '✗ FAIL'}
  Memory Target (<10MB): {'✓ PASS' if avg_memory < 10 else '✗ FAIL'}
  CPU Target (<20%): {'✓ PASS' if avg_cpu < 20 else '✗ FAIL'}
"""
        return summary
    
    def save_statistics(self, filename='detection_statistics.json'):
        """
        Save detection statistics to file for analysis.
        """
        try:
            stats_copy = self.stats.copy()
            stats_copy['summary'] = {
                'avg_processing_time_ms': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                'max_processing_time_ms': np.max(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                'avg_memory_mb': np.mean(self.stats['memory_usage']) if self.stats['memory_usage'] else 0,
                'avg_cpu_percent': np.mean(self.stats['cpu_usage']) if self.stats['cpu_usage'] else 0,
                'detection_rate_percent': (self.stats['malicious_flows_detected'] / 
                                         self.stats['total_flows_processed'] * 100) if self.stats['total_flows_processed'] > 0 else 0,
                'false_alarm_rate_percent': (self.stats['false_alarms'] / 
                                           self.stats['total_flows_processed'] * 100) if self.stats['total_flows_processed'] > 0 else 0
            }
            
            with open(filename, 'w') as f:
                json.dump(stats_copy, f, indent=2, default=str)
            
            logger.info(f"Statistics saved to '{filename}'")
            
        except Exception as e:
            logger.error(f"Error saving statistics: {str(e)}")
    
    def start_detection_loop(self, duration_minutes=5, flow_interval=0.5, save_stats=True):
        """
        Main detection loop that simulates continuous network monitoring.
        
        Args:
            duration_minutes: How long to run the simulation
            flow_interval: Seconds between flow processing
            save_stats: Whether to save statistics to file
        """
        if self.model is None:
            logger.error("No model loaded. Cannot start detection.")
            return
        
        logger.info("=" * 60)
        logger.info("🛡️  PiSentry Malicious Traffic Detection Agent Starting")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Feature Selector: {'Loaded' if self.feature_selector else 'Not Available'}")
        logger.info(f"Duration: {duration_minutes} minutes")
        logger.info(f"Flow Interval: {flow_interval} seconds")
        logger.info(f"Agent monitoring network traffic... (simulated)")
        logger.info("-" * 60)
        
        self.is_running = True
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        flow_counter = 0
        
        try:
            while self.is_running and time.time() < end_time:
                flow_counter += 1
                
                # Monitor system resources
                resource_info = self.monitor_resources()
                
                # Simulate a new network flow
                flow_data, actual_is_malicious = self.simulate_network_flow()
                
                # Extract and select features
                features = self.extract_and_select_features(flow_data)
                
                # Classify the flow
                result = self.classify_flow(features)
                
                # Log the detection
                actual_label = 1 if actual_is_malicious else 0
                self.log_detection(result, actual_label, flow_counter)
                
                # Check for performance warnings
                if resource_info:
                    if resource_info['memory_mb'] > 20:  # Warning threshold
                        logger.warning(f"High memory usage: {resource_info['memory_mb']:.1f} MB")
                    if resource_info['cpu_percent'] > 30:  # Warning threshold
                        logger.warning(f"High CPU usage: {resource_info['cpu_percent']:.1f}%")
                
                # Wait before processing next flow
                time.sleep(flow_interval)
                
                # Print periodic status updates
                if flow_counter % 20 == 0:
                    elapsed_minutes = (time.time() - start_time) / 60
                    logger.info(f"Status: {flow_counter} flows processed in {elapsed_minutes:.1f} minutes")
        
        except KeyboardInterrupt:
            logger.info("\n⚠️  Detection agent stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Error in detection loop: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
        
        # Generate final summary
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"🛡️  Detection session completed - Runtime: {total_time/60:.1f} minutes")
        logger.info(self.get_performance_summary())
        logger.info("=" * 60)
        
        # Save statistics
        if save_stats:
            self.save_statistics()
    
    def simulate_network_flow(self, flow_type='random'):
        """
        Enhanced network flow simulation with realistic patterns.
        """
        return self.simulate_network_flow(flow_type)
    
    def extract_and_select_features(self, flow_data):
        """
        Extract features and apply selection.
        """
        return self.extract_and_select_features(flow_data)

def run_batch_evaluation(agent, num_flows=100, include_malicious=True):
    """
    Run a batch evaluation to assess model performance.
    """
    logger.info(f"\n🧪 Running batch evaluation with {num_flows} flows...")
    
    results = []
    ground_truth = []
    
    for i in range(num_flows):
        # Generate flow with known label
        if include_malicious:
            flow_type = np.random.choice(['benign', 'malicious'], p=[0.8, 0.2])
        else:
            flow_type = 'benign'
        
        flow_data, actual_is_malicious = agent.simulate_network_flow(flow_type)
        features = agent.extract_and_select_features(flow_data)
        result = agent.classify_flow(features)
        
        results.append(result['prediction'])
        ground_truth.append(1 if actual_is_malicious else 0)
    
    # Calculate batch performance metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    accuracy = accuracy_score(ground_truth, results)
    f1 = f1_score(ground_truth, results)
    precision = precision_score(ground_truth, results, zero_division=0)
    recall = recall_score(ground_truth, results, zero_division=0)
    
    logger.info(f"Batch Evaluation Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'predictions': results,
        'ground_truth': ground_truth
    }

def main():
    """
    Main function to run the PiSentry detection agent.
    """
    print("🛡️  PiSentry Real-time Detection Agent")
    print("=====================================\n")
    
    # Initialize the agent
    agent = PiSentryAgent()
    
    if agent.model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Run options
    print("Available options:")
    print("1. Run continuous detection (5 minutes)")
    print("2. Run extended detection (30 minutes)")
    print("3. Run batch evaluation (100 flows)")
    print("4. Run quick test (10 flows)")
    print("5. Custom duration")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            agent.start_detection_loop(duration_minutes=5)
        elif choice == '2':
            agent.start_detection_loop(duration_minutes=30)
        elif choice == '3':
            run_batch_evaluation(agent, num_flows=100)
        elif choice == '4':
            agent.start_detection_loop(duration_minutes=0.5, flow_interval=0.2)
        elif choice == '5':
            duration = float(input("Enter duration in minutes: "))
            interval = float(input("Enter flow interval in seconds (0.5): ") or "0.5")
            agent.start_detection_loop(duration_minutes=duration, flow_interval=interval)
        else:
            print("Invalid option. Running default 5-minute detection...")
            agent.start_detection_loop(duration_minutes=5)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except ValueError:
        print("❌ Invalid input. Running default detection...")
        agent.start_detection_loop(duration_minutes=2)
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
    
