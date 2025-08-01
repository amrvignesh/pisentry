# train.py
# Enhanced model training script for PiSentry
# Trains optimized models with proper hyperparameter settings and validation

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from utils import (
    generate_simulated_data, 
    apply_chi_square_feature_selection, 
    get_train_test_split,
    validate_feature_consistency
)

# Hyperparameters based on paper specifications
NB_ALPHA = 0.1  # Smoothing parameter for MultinomialNB
DT_MAX_DEPTH = 8  # Maximum depth for Decision Tree to prevent overfitting
RANDOM_STATE = 42  # For reproducibility
CV_FOLDS = 5  # Cross-validation folds

def train_naive_bayes(X_train, y_train, alpha=NB_ALPHA):
    """
    Train Multinomial Naïve Bayes classifier with optimized parameters.
    """
    print("Training Multinomial Naïve Bayes classifier...")
    
    # Initialize model with specified parameters
    nb_model = MultinomialNB(alpha=alpha)
    
    # Train the model
    nb_model.fit(X_train, y_train)
    
    # Perform cross-validation for model validation
    cv_scores = cross_val_score(nb_model, X_train, y_train, cv=CV_FOLDS, scoring='f1')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return nb_model

def train_decision_tree(X_train, y_train, max_depth=DT_MAX_DEPTH):
    """
    Train pruned Decision Tree classifier optimized for resource constraints.
    """
    print(f"Training Pruned Decision Tree (max_depth={max_depth})...")
    
    # Initialize model with resource-conscious parameters
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        min_samples_split=5,  # Prevent overfitting
        min_samples_leaf=2,   # Ensure leaf nodes have minimum samples
        criterion='gini'      # Use Gini impurity for faster computation
    )
    
    # Train the model
    dt_model.fit(X_train, y_train)
    
    # Perform cross-validation for model validation
    cv_scores = cross_val_score(dt_model, X_train, y_train, cv=CV_FOLDS, scoring='f1')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Display feature importance for the top features
    feature_names = [f'feature_{i+1}' for i in range(X_train.shape[1])]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance.head(10))
    
    return dt_model

def hyperparameter_tuning(X_train, y_train, model_type='decision_tree'):
    """
    Perform hyperparameter tuning for the specified model.
    """
    print(f"Performing hyperparameter tuning for {model_type}...")
    
    if model_type == 'decision_tree':
        # Define parameter grid for Decision Tree
        param_grid = {
            'max_depth': [6, 8, 10, 12],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize base model
        base_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
        
    elif model_type == 'naive_bayes':
        # Define parameter grid for Naive Bayes
        param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
        }
        
        # Initialize base model
        base_model = MultinomialNB()
    
    else:
        raise ValueError("Model type must be 'decision_tree' or 'naive_bayes'")
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=CV_FOLDS, 
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model_on_training(model, X_train, y_train, model_name):
    """
    Evaluate model performance on training data for detailed analysis.
    """
    print(f"\n=== {model_name} Training Performance ===")
    
    # Make predictions
    y_pred = model.predict(X_train)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_train, y_pred, target_names=['Benign', 'Malicious']))
    
    # Print confusion matrix
    cm = confusion_matrix(y_train, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    return y_pred

def save_models_and_metadata(models, feature_selector, selected_features, save_metadata=True):
    """
    Save trained models and associated metadata for deployment.
    """
    print("\n=== Saving Models and Metadata ===")
    
    # Save models
    for name, model in models.items():
        filename = f"{name.lower().replace(' ', '_')}_model.joblib"
        joblib.dump(model, filename)
        print(f"{name} model saved as '{filename}'")
    
    # Save feature selector
    if feature_selector is not None:
        joblib.dump(feature_selector, 'feature_selector.joblib')
        print("Feature selector saved as 'feature_selector.joblib'")
    
    # Save metadata
    if save_metadata:
        metadata = {
            'selected_features': selected_features,
            'num_features': len(selected_features),
            'models_trained': list(models.keys()),
            'hyperparameters': {
                'naive_bayes_alpha': NB_ALPHA,
                'decision_tree_max_depth': DT_MAX_DEPTH,
                'random_state': RANDOM_STATE
            },
            'training_config': {
                'cv_folds': CV_FOLDS,
                'test_size': 0.35,
                'feature_selection_method': 'chi_square'
            }
        }
        
        joblib.dump(metadata, 'training_metadata.joblib')
        print("Training metadata saved as 'training_metadata.joblib'")

def main():
    """
    Main training pipeline that orchestrates the entire process.
    """
    print("=== PiSentry Model Training Pipeline ===\n")
    
    # Step 1: Generate or load dataset
    print("Step 1: Data Generation")
    X, y = generate_simulated_data(num_samples=1000, num_base_features=100)
    
    # Step 2: Apply feature selection
    print("\nStep 2: Feature Selection")
    X_selected, feature_selector, selected_features, feature_importance = apply_chi_square_feature_selection(X, y, k=25)
    
    # Step 3: Split data
    print("\nStep 3: Data Splitting")
    X_train, X_test, y_train, y_test = get_train_test_split(X_selected, y)
    
    # Validate feature consistency
    if not validate_feature_consistency(X_train, X_test):
        print("Warning: Feature inconsistency detected!")
    
    # Step 4: Train models
    print("\nStep 4: Model Training")
    
    # Train Naive Bayes
    nb_model = train_naive_bayes(X_train, y_train)
    
    # Train Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    
    # Step 5: Optional hyperparameter tuning
    print("\nStep 5: Hyperparameter Tuning (Optional)")
    
    # Uncomment the following lines for hyperparameter tuning
    # This may take several minutes to complete
    """
    print("Tuning Decision Tree...")
    dt_tuned, dt_best_params = hyperparameter_tuning(X_train, y_train, 'decision_tree')
    
    print("Tuning Naive Bayes...")
    nb_tuned, nb_best_params = hyperparameter_tuning(X_train, y_train, 'naive_bayes')
    """
    
    # Step 6: Evaluate models on training data
    print("\nStep 6: Training Data Evaluation")
    
    models = {
        'Multinomial Naive Bayes': nb_model,
        'Pruned Decision Tree': dt_model
    }
    
    for name, model in models.items():
        evaluate_model_on_training(model, X_train, y_train, name)
    
    # Step 7: Save models and metadata
    print("\nStep 7: Saving Models")
    save_models_and_metadata(models, feature_selector, selected_features)
    
    print("\n=== Training Complete ===")
    print("Models have been saved and are ready for evaluation and deployment.")
    print("Next steps:")
    print("1. Run 'python eval.py' to evaluate model performance")
    print("2. Run 'python inference_pi.py' to test real-time inference")

if __name__ == "__main__":
    # Import pandas here to avoid issues if not available
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required for training. Please install with: pip install pandas")
        exit(1)
    
    main()
    
