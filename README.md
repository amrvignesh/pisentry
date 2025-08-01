
# PiSentry: Lightweight AI-Driven Malware Detection for Resource-Constrained IoT Devices

### A Final Project for CSC 6222

This repository contains the source code and documentation for "PiSentry," a proof-of-concept for a lightweight, on-device machine learning agent. The agent is designed to detect malicious network traffic on resource-constrained IoT devices, specifically the Raspberry Pi Zero, by utilizing a pruned Decision Tree classifier.

The project was developed as a final submission for CSC 6222 and demonstrates a solution for real-time, offline malware detection that adheres to strict CPU and memory limitations.
    

## Project Overview

The proliferation of IoT devices presents a significant attack surface for botnets like Mirai. Conventional security solutions are often too resource-intensive for these devices. This project proposes a machine learning-based approach where a small, pre-trained model is deployed directly on the device.

The agent, running on a Raspberry Pi Zero, passively monitors network traffic, extracts key features from data flows, and uses a trained Decision Tree model to classify the traffic as either benign or malicious. The primary goal is to raise a local alert within 250 milliseconds of detection while consuming minimal system resources.

## Key Features

-   **Lightweight Model:** Utilizes a small, efficient Decision Tree classifier suitable for low-power hardware.
    
-   **On-Device Inference:** All detection logic is executed locally on the Raspberry Pi Zero, requiring no cloud connectivity.
    
-   **Real-Time Detection:** Designed to process network traffic and generate an alert with minimal latency.
    
-   **Simulated Environment:** Includes scripts for generating simulated data and a complete development workflow for training and evaluation.
    

## Project Structure

The repository is organized into the following key files:

-   `utils.py`: Contains helper functions for the project, including simulated data generation, data splitting, and feature extraction.
    
-   `train.py`: The model training script. This program trains the Decision Tree and Naïve Bayes classifiers on the simulated data and saves them as `.joblib` files.
    
-   `eval.py`: The model evaluation script. This program loads the trained models, evaluates their performance on a test set, and generates a comparison graph (`comparison_graph.png`).
    
-   `inference_pi.py`: The core "on-device" program. This script simulates the real-time detection agent that would run on the Raspberry Pi Zero. It loads a pre-trained model and continuously monitors simulated network flows.
    
-   `README.md`: This file.
    

## Getting Started

To get the project up and running, follow these steps:

1.  **Clone the Repository:**
    
    ```
    git clone https://github.com/amrvignesh/pisentry.git
    cd pisentry
    
    ```
    
2.  **Run the Training and Evaluation Scripts:** This will generate the necessary model files and the performance graph.
    
    ```
    python train.py
    python eval.py
    
    ```
    
3.  **Simulate the On-Device Agent:** This will start the simulated detection agent. Press `Ctrl+C` to stop it.
    
    ```
    python inference_pi.py
    
    ```
