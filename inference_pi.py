# inference_pi.py
# This is the "on-device" program that would be deployed to the Raspberry Pi Zero.
# It simulates the real-time detection agent.

import joblib
import numpy as np
import time
from utils import simulate_feature_extraction

# --- 1. MODEL LOADING AND SETUP ---
# This is done once at the beginning to save resources.

def load_model(model_path):
    """Loads a pre-trained model from a file."""
    try:
        model = joblib.load(model_path)
        print(f"Model '{model_path}' loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please run train.py first.")
        return None

# --- 2. THE CORE INFERENCE LOOP ---
# This loop simulates the agent's continuous operation.

def start_detection_agent(model):
    """
    Simulates the main detection loop on the Raspberry Pi Zero.
    In a real project, this would interface with network packet capture.
    """
    if model is None:
        return

    print("--- Starting Malicious Traffic Detection Agent on Raspberry Pi Zero ---")
    print("Agent is now monitoring network traffic... (simulated)")

    # The agent runs continuously until stopped.
    try:
        while True:
            # In a real scenario, this is where you would use a library like
            # Scapy or read from Zeek logs to get a new traffic flow.
            # Here, we simulate a new flow of data.
            is_malicious = np.random.choice([True, False], p=[0.1, 0.9])
            if is_malicious:
                # Simulate a malicious flow (more high-value features)
                simulated_flow = np.random.randint(50, 150, size=(1, 25))
            else:
                # Simulate a benign flow
                simulated_flow = np.random.randint(0, 100, size=(1, 25))

            # Simulate the feature extraction process
            features = simulate_feature_extraction(simulated_flow)

            # Perform inference to get a prediction
            prediction = model.predict(features)
            
            # Raise an alert if malicious traffic is detected
            if prediction[0] == 1:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- ALERT: Malicious traffic detected! ---")
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Benign traffic detected.")
            
            # Simulate a brief wait time before the next check
            # Your project specifies an alert time of <250ms, so we can model this
            # with a quick loop and a small sleep interval.
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nDetection agent stopped by user.")

if __name__ == "__main__":
    # Load the pre-trained model
    # We will use the Decision Tree model as it had better performance in the paper draft.
    model_path = 'decision_tree_model.joblib'
    model = load_model(model_path)
    
    # Start the detection agent
    start_detection_agent(model)

