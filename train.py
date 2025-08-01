# train.py
# This is the "model creator" program. It handles the training process
# and saves the final models as joblib files.
# It should be run on a development machine, not the Raspberry Pi.

import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from utils import generate_simulated_data, get_train_test_split

def train_models(X_train, y_train):
    """Trains and saves the Multinomial Naïve Bayes and Pruned Decision Tree classifiers."""
    print("--- Starting Model Training ---")
    
    # Train and save the Multinomial Naïve Bayes model
    print("Training Multinomial Naïve Bayes...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)
    joblib.dump(nb_model, 'naive_bayes_model.joblib')
    print("Naïve Bayes model saved as 'naive_bayes_model.joblib'.")

    # Train and save the Pruned Decision Tree model
    print("Training Pruned Decision Tree (max_depth=8)...")
    dt_model = DecisionTreeClassifier(max_depth=8)
    dt_model.fit(X_train, y_train)
    joblib.dump(dt_model, 'decision_tree_model.joblib')
    print("Decision Tree model saved as 'decision_tree_model.joblib'.")
    
    print("--- Training Complete ---")
    return nb_model, dt_model

if __name__ == "__main__":
    # Generate data and split it
    X, y = generate_simulated_data()
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    # Train the models and save them
    train_models(X_train, y_train)

