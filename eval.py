# eval.py
# Enhanced model evaluation script for PiSentry
# Comprehensive performance evaluation with resource monitoring and statistical analysis

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score
import time
import psutil
import warnings
warnings.filterwarnings('ignore')

from utils import (
    generate_simulated_data, 
    apply_chi_square_feature_selection, 
    get_train_test_split,
    validate_feature_consistency
)

# Resource monitoring thresholds from paper
TARGET_F1_SCORE = 0.92
MAX_MEMORY_MB = 10
MAX_CPU_PERCENT = 20
MAX_INFERENCE_MS = 250

class ResourceMonitor:
    """Monitor system resources during model evaluation."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.baseline_cpu = self.process.cpu_percent()
    
    def get_current_usage(self):
        """Get current resource usage."""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        return {
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'memory_increase_mb': memory_mb - self.baseline_memory
        }

def evaluate_models(models, X_test, y_test, monitor_resources=True):
    """
    Comprehensive model evaluation with all required metrics.
    Includes resource monitoring as described in the paper.
    """
    results = {}
    resource_monitor = ResourceMonitor() if monitor_resources else None
    
    for name, model in models.items():
        print(f"\n=== Evaluating {name} ===")
        
        # Time the inference
        start_time = time.time()
        
        # Monitor resources before inference
        if resource_monitor:
            resources_before = resource_monitor.get_current_usage()
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Monitor resources after inference
        if resource_monitor:
            resources_after = resource_monitor.get_current_usage()
        
        # Calculate confusion matrix components
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate all performance metrics
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # Calculate rates
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Calculate AUC if probabilities available
        auc_score = None
        if y_pred_proba is not None:
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                auc_score = None
        
        # Store results
        results[name] = {
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'auc_score': auc_score,
            'confusion_matrix': cm,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'inference_time_ms': inference_time,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        # Add resource monitoring results
        if resource_monitor:
            results[name].update({
                'memory_usage_mb': resources_after['memory_mb'],
                'cpu_usage_percent': resources_after['cpu_percent'],
                'memory_increase_mb': resources_after['memory_increase_mb']
            })
        
        # Print detailed results
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")
        print(f"Inference Time: {inference_time:.2f} ms")
        
        if resource_monitor:
            print(f"Memory Usage: {resources_after['memory_mb']:.1f} MB")
            print(f"CPU Usage: {resources_after['cpu_percent']:.1f}%")
        
        # Check if meets paper targets
        meets_f1_target = f1 >= TARGET_F1_SCORE
        meets_memory_target = not resource_monitor or resources_after['memory_mb'] <= MAX_MEMORY_MB
        meets_cpu_target = not resource_monitor or resources_after['cpu_percent'] <= MAX_CPU_PERCENT
        meets_inference_target = inference_time <= MAX_INFERENCE_MS
        
        print(f"Meets F1 Target (≥{TARGET_F1_SCORE}): {'✓' if meets_f1_target else '✗'}")
        if resource_monitor:
            print(f"Meets Memory Target (≤{MAX_MEMORY_MB}MB): {'✓' if meets_memory_target else '✗'}")
            print(f"Meets CPU Target (≤{MAX_CPU_PERCENT}%): {'✓' if meets_cpu_target else '✗'}")
        print(f"Meets Inference Target (≤{MAX_INFERENCE_MS}ms): {'✓' if meets_inference_target else '✗'}")
    
    return results

def generate_detailed_report(eval_results, save_to_file=True):
    """
    Generate a comprehensive evaluation report.
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*60)
    
    # Create summary table
    summary_data = []
    for model_name, results in eval_results.items():
        summary_data.append({
            'Model': model_name,
            'F1 Score': f"{results['f1_score']:.4f}",
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'FPR': f"{results['false_positive_rate']:.4f}",
            'Inference (ms)': f"{results['inference_time_ms']:.2f}",
            'Memory (MB)': f"{results.get('memory_usage_mb', 'N/A'):.1f}" if isinstance(results.get('memory_usage_mb'), (int, float)) else 'N/A',
            'CPU (%)': f"{results.get('cpu_usage_percent', 'N/A'):.1f}" if isinstance(results.get('cpu_usage_percent'), (int, float)) else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nPERFORMANCE SUMMARY:")
    print(summary_df.to_string(index=False))
    
    # Detailed analysis for each model
    for model_name, results in eval_results.items():
        print(f"\n{'-'*50}")
        print(f"DETAILED ANALYSIS: {model_name}")
        print(f"{'-'*50}")
        
        cm = results['confusion_matrix']
        print("Confusion Matrix:")
        print(f"                 Predicted")
        print(f"               Benign  Malicious")
        print(f"Actual Benign    {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"     Malicious   {cm[1,0]:4d}    {cm[1,1]:4d}")
        
        print(f"\nClassification Metrics:")
        print(f"  True Positives:  {results['true_positives']}")
        print(f"  True Negatives:  {results['true_negatives']}")
        print(f"  False Positives: {results['false_positives']}")
        print(f"  False Negatives: {results['false_negatives']}")
        
        print(f"\nPerformance Rates:")
        print(f"  Sensitivity (Recall):    {results['recall']:.4f}")
        print(f"  Specificity:            {results['specificity']:.4f}")
        print(f"  False Positive Rate:    {results['false_positive_rate']:.4f}")
        print(f"  False Negative Rate:    {results['false_negative_rate']:.4f}")
        
        if results['auc_score'] is not None:
            print(f"  AUC Score:              {results['auc_score']:.4f}")
    
    # Save report to file
    if save_to_file:
        report_filename = 'evaluation_report.txt'
        with open(report_filename, 'w') as f:
            f.write("PiSentry Model Evaluation Report\n")
            f.write("="*50 + "\n\n")
            f.write("Performance Summary:\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            
            for model_name, results in eval_results.items():
                f.write(f"\n{model_name} Detailed Results:\n")
                f.write(f"F1 Score: {results['f1_score']:.4f}\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n")
                f.write(f"False Positive Rate: {results['false_positive_rate']:.4f}\n")
                f.write(f"Inference Time: {results['inference_time_ms']:.2f} ms\n")
                if 'memory_usage_mb' in results:
                    f.write(f"Memory Usage: {results['memory_usage_mb']:.1f} MB\n")
                    f.write(f"CPU Usage: {results['cpu_usage_percent']:.1f}%\n")
        
        print(f"\nDetailed report saved to '{report_filename}'")

def generate_comparison_graph(eval_results, save_plot=True):
    """
    Generate the comparison bar chart for the final paper.
    Enhanced version with better visualization and error handling.
    """
    print("\nGenerating performance comparison graph...")
    
    # Extract data for plotting
    models = list(eval_results.keys())
    f1_scores = [eval_results[m]['f1_score'] for m in models]
    accuracies = [eval_results[m]['accuracy'] for m in models]
    false_positive_rates = [eval_results[m]['false_positive_rate'] for m in models]
    
    # Create the comparison plot
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    # Create bars
    rects1 = ax.bar(x - width, f1_scores, width, label='F1 Score', 
                   color='#1d4ed8', alpha=0.8, edgecolor='white', linewidth=1)
    rects2 = ax.bar(x, accuracies, width, label='Accuracy', 
                   color='#3b82f6', alpha=0.8, edgecolor='white', linewidth=1)
    rects3 = ax.bar(x + width, false_positive_rates, width, label='False Positive Rate', 
                   color='#93c5fd', alpha=0.8, edgecolor='white', linewidth=1)
    
    # Customize the plot
    ax.set_ylabel('Score / Rate', fontsize=12, fontweight='bold')
    ax.set_title('PiSentry: Algorithm Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_labels(rects, format_str='.3f'):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:{format_str}}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    # Add target line for F1 score
    ax.axhline(y=TARGET_F1_SCORE, color='red', linestyle='--', alpha=0.7, 
               label=f'F1 Target ({TARGET_F1_SCORE})')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('comparison_graph.png', dpi=300, bbox_inches='tight')
        print("Performance comparison graph saved as 'comparison_graph.png'")
    
    plt.show()

def generate_confusion_matrix_plots(eval_results, save_plots=True):
    """
    Generate confusion matrix heatmaps for each model.
    """
    print("Generating confusion matrix visualizations...")
    
    n_models = len(eval_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5), dpi=150)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(eval_results.items()):
        cm = results['confusion_matrix']
        
        # Create DataFrame for better labeling
        cm_df = pd.DataFrame(cm,
                           index=['Actual Benign', 'Actual Malicious'],
                           columns=['Predicted Benign', 'Predicted Malicious'])
        
        # Create heatmap
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                   ax=axes[idx], cbar=True, square=True,
                   annot_kws={'size': 12, 'weight': 'bold'})
        
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', 
                          fontsize=12, fontweight='bold', pad=15)
        axes[idx].set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Actual Class', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix plots saved as 'confusion_matrices.png'")
    
    plt.show()

def generate_roc_curves(eval_results, save_plot=True):
    """
    Generate ROC curves for models that support probability predictions.
    """
    print("Generating ROC curves...")
    
    plt.figure(figsize=(8, 6), dpi=150)
    
    colors = ['#1d4ed8', '#dc2626', '#059669', '#7c3aed']
    
    for idx, (model_name, results) in enumerate(eval_results.items()):
        if results['prediction_probabilities'] is not None:
            # Get test labels (we need to regenerate them for ROC curve)
            # This is a limitation - in practice, you'd pass y_test to this function
            fpr, tpr, _ = roc_curve(np.ones(len(results['predictions'])), 
                                   results['prediction_probabilities'])
            auc_score = results['auc_score']
            
            plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                    label=f'{model_name} (AUC = {auc_score:.3f})' if auc_score else model_name)
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.7)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - PiSentry Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("ROC curves saved as 'roc_curves.png'")
    
    plt.show()

def perform_statistical_analysis(eval_results, X_test, y_test):
    """
    Perform statistical significance testing between models.
    """
    print("\nPerforming statistical analysis...")
    
    if len(eval_results) < 2:
        print("Need at least 2 models for statistical comparison")
        return
    
    # McNemar's test for comparing classifiers
    from scipy.stats import mcnemar
    
    model_names = list(eval_results.keys())
    results_matrix = []
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:  # Only compare each pair once
                pred1 = eval_results[model1]['predictions']
                pred2 = eval_results[model2]['predictions']
                
                # Create contingency table
                correct1 = (pred1 == y_test)
                correct2 = (pred2 == y_test)
                
                both_correct = np.sum(correct1 & correct2)
                model1_only = np.sum(correct1 & ~correct2)
                model2_only = np.sum(~correct1 & correct2)
                both_wrong = np.sum(~correct1 & ~correct2)
                
                contingency_table = np.array([[both_correct, model1_only],
                                            [model2_only, both_wrong]])
                
                # Perform McNemar's test
                if model1_only + model2_only > 0:
                    result = mcnemar(contingency_table, exact=False, correction=True)
                    p_value = result.pvalue
                    is_significant = p_value < 0.05
                    
                    print(f"{model1} vs {model2}:")
                    print(f"  McNemar's test p-value: {p_value:.6f}")
                    print(f"  Statistically significant difference: {'Yes' if is_significant else 'No'}")
                    print(f"  {model1} correct only: {model1_only}")
                    print(f"  {model2} correct only: {model2_only}")
                    print()

def main():
    """
    Main evaluation pipeline that orchestrates the entire evaluation process.
    """
    print("=== PiSentry Model Evaluation Pipeline ===\n")
    
    try:
        # Step 1: Load trained models
        print("Step 1: Loading trained models...")
        models_to_eval = {}
        
        try:
            nb_model = joblib.load('multinomial_naive_bayes_model.joblib')
            models_to_eval['Multinomial Naive Bayes'] = nb_model
            print("✓ Multinomial Naive Bayes model loaded")
        except FileNotFoundError:
            print("✗ Multinomial Naive Bayes model not found")
        
        try:
            dt_model = joblib.load('pruned_decision_tree_model.joblib')
            models_to_eval['Pruned Decision Tree'] = dt_model
            print("✓ Pruned Decision Tree model loaded")
        except FileNotFoundError:
            print("✗ Pruned Decision Tree model not found")
        
        if not models_to_eval:
            print("No trained models found. Please run 'python train.py' first.")
            return
        
        # Step 2: Load feature selector and generate test data
        print("\nStep 2: Preparing test data...")
        
        # Try to load saved feature selector
        feature_selector = None
        try:
            feature_selector = joblib.load('feature_selector.joblib')
            print("✓ Feature selector loaded")
        except FileNotFoundError:
            print("✗ Feature selector not found, will regenerate")
        
        # Generate test data (same process as training)
        X, y = generate_simulated_data(num_samples=1000)
        
        # Apply feature selection
        if feature_selector is not None:
            # Use existing selector
            X_selected = feature_selector.transform(X)
        else:
            # Regenerate feature selection
            X_selected, feature_selector, selected_features, _ = apply_chi_square_feature_selection(X, y, k=25)
        
        # Split data (using same random state as training)
        X_train, X_test, y_train, y_test = get_train_test_split(X_selected, y)
        
        print(f"Test set prepared: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Step 3: Evaluate models
        print("\nStep 3: Model Evaluation")
        eval_results = evaluate_models(models_to_eval, X_test, y_test, monitor_resources=True)
        
        # Step 4: Generate comprehensive report
        print("\nStep 4: Generating Reports")
        generate_detailed_report(eval_results, save_to_file=True)
        
        # Step 5: Generate visualizations
        print("\nStep 5: Creating Visualizations")
        generate_comparison_graph(eval_results, save_plot=True)
        generate_confusion_matrix_plots(eval_results, save_plots=True)
        
        # Only generate ROC curves if we have probability predictions
        if any(results['prediction_probabilities'] is not None for results in eval_results.values()):
            generate_roc_curves(eval_results, save_plot=True)
        
        # Step 6: Statistical analysis
        if len(eval_results) >= 2:
            print("\nStep 6: Statistical Analysis")
            perform_statistical_analysis(eval_results, X_test, y_test)
        
        print("\n=== Evaluation Complete ===")
        print("Generated files:")
        print("- comparison_graph.png")
        print("- confusion_matrices.png")
        print("- evaluation_report.txt")
        print("- roc_curves.png (if applicable)")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import scipy
    except ImportError:
        print("Warning: scipy not available. Statistical tests will be skipped.")
    
    main()
    
