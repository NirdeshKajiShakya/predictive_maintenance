# IMPORTS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve
)
from imblearn.over_sampling import SMOTE

# CONFIGURATION
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_FILE = 'predictive_maintenance.csv'
FIGURES_DIR = 'figures'

# DATA LOADING AND PREPROCESSING
def load_and_preprocess_data(filepath):
    """
    Load and preprocess the predictive maintenance dataset.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        tuple: (X, y, df) - Features, target, and original dataframe
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Drop unique identifiers that don't contribute to predictions
    df = df.drop(['UDI', 'Product ID'], axis=1)
    
    # Encode categorical 'Type' column (L, M, H quality tiers)
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])
    
    # Separate features and target
    # Drop failure subtypes (TWF, HDF, PWF, OSF, RNF) to focus on main failure
    X = df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    y = df['Machine failure']
    
    print(f"Dataset loaded: {len(df)} samples, {X.shape[1]} features")
    print(f"Class distribution:\n{y.value_counts()}")
    
    return X, y, df

def handle_class_imbalance(X, y):
    """
    Apply SMOTE to handle class imbalance in the dataset.
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        
    Returns:
        tuple: (X_resampled, y_resampled) - Balanced dataset
    """
    print("\nApplying SMOTE to address class imbalance...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"After SMOTE: {len(y_resampled)} samples")
    print(f"Class distribution:\n{pd.Series(y_resampled).value_counts()}")
    
    return X_resampled, y_resampled


# MODEL TRAINING
def train_model(X_train, y_train):
    """
    Train Random Forest model with GridSearchCV for hyperparameter optimization.
    
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        
    Returns:
        GridSearchCV: Fitted grid search object containing best model
    """
    print("\nTraining Random Forest with GridSearchCV...")
    
    # Define the model
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    # Perform grid search with cross-validation
    # Using 'recall' scoring to prioritize catching failures
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, 
        scoring='recall',
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation recall score: {grid_search.best_score_:.4f}")
    
    return grid_search


# MODEL EVALUATION
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print metrics.
    
    Args:
        model: Trained model
        X_test (DataFrame): Test features
        y_test (Series): Test target
        
    Returns:
        tuple: (y_pred, y_pred_proba, cm) - Predictions, probabilities, and confusion matrix
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix Values:")
    print(f"  True Negatives (TN):  {tn:,}")
    print(f"  False Positives (FP): {fp:,}")
    print(f"  False Negatives (FN): {fn:,}")
    print(f"  True Positives (TP):  {tp:,}")
    
    # AUC Score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC Score: {auc_score:.4f}")
    
    return y_pred, y_pred_proba, cm


# VISUALIZATIONS
def setup_figures_directory():
    """Create directory for saving figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"\nFigures will be saved to '{FIGURES_DIR}/' directory")


def plot_class_distribution(y_original, y_resampled):
    """
    Plot class distribution before and after SMOTE.
    
    Args:
        y_original (Series): Original target variable
        y_resampled (Series): Resampled target variable
    """
    # Before SMOTE
    class_counts_before = y_original.value_counts()
    
    plt.figure(figsize=(8, 6))
    class_counts_before.plot(kind='bar', color=['steelblue', 'coral'])
    plt.title('Class Distribution Before SMOTE Oversampling', fontsize=14, fontweight='bold')
    plt.xlabel('Machine Failure', fontsize=11)
    plt.ylabel('Number of Samples', fontsize=11)
    plt.xticks([0, 1], ['No Failure (0)', 'Failure (1)'], rotation=0)
    
    for i, count in enumerate(class_counts_before):
        plt.text(i, count/2, f'{count:,}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/class_distribution_before_smote.png', dpi=300)
    plt.show()
    
    # After SMOTE
    class_counts_after = pd.Series(y_resampled).value_counts()
    
    plt.figure(figsize=(8, 6))
    class_counts_after.plot(kind='bar', color=['steelblue', 'coral'])
    plt.title('Class Distribution After SMOTE Oversampling', fontsize=14, fontweight='bold')
    plt.xlabel('Machine Failure', fontsize=11)
    plt.ylabel('Number of Samples', fontsize=11)
    plt.xticks([0, 1], ['No Failure (0)', 'Failure (1)'], rotation=0)
    
    for i, count in enumerate(class_counts_after):
        plt.text(i, count/2, f'{count:,}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/class_distribution_after_smote.png', dpi=300)
    plt.show()


def plot_confusion_matrix(cm, y_test, y_pred):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm (array): Confusion matrix
        y_test (Series): True labels
        y_pred (array): Predicted labels
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Predicted No Failure', 'Predicted Failure'],
        yticklabels=['Actual No Failure', 'Actual Failure'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Random Forest with SMOTE', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    
    # Add annotations for TP, TN, FP, FN
    plt.text(0.5, 0.1, 'TN', fontsize=12, ha='center', color='white', weight='bold')
    plt.text(1.5, 0.1, 'FP', fontsize=12, ha='center', color='black', weight='bold')
    plt.text(0.5, 1.1, 'FN', fontsize=12, ha='center', color='black', weight='bold')
    plt.text(1.5, 1.1, 'TP', fontsize=12, ha='center', color='white', weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/confusion_matrix.png', dpi=300)
    plt.show()


def plot_roc_curve(y_test, y_pred_proba):
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_test (Series): True labels
        y_pred_proba (array): Predicted probabilities
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=11)
    plt.title('ROC Curve - Random Forest Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/roc_curve.png', dpi=300)
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from the trained model.
    
    Args:
        model: Trained model
        feature_names (Index): Names of features
    """
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names[indices], palette='viridis')
    plt.title('Feature Importance in Predicting Machine Failure', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=11)
    plt.ylabel('Features', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/feature_importance.png', dpi=300)
    plt.show()


def save_hyperparameter_results(grid_search):
    """
    Save GridSearchCV results to CSV file.
    
    Args:
        grid_search (GridSearchCV): Fitted grid search object
    """
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Select and rename relevant columns
    results_table = results[[
        'param_n_estimators', 
        'param_max_depth',
        'param_min_samples_split', 
        'mean_test_score',
        'std_test_score'
    ]].copy()
    
    results_table.columns = [
        'N_Estimators', 
        'Max_Depth', 
        'Min_Samples_Split',
        'Mean_Recall_Score', 
        'Std_Recall_Score'
    ]
    
    # Sort by best score
    results_table = results_table.sort_values('Mean_Recall_Score', ascending=False)
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING RESULTS (Top 5)")
    print("="*70)
    print(results_table.head().to_string(index=False))
    
    # Save to CSV
    results_table.to_csv(f'{FIGURES_DIR}/gridsearch_results.csv', index=False)
    print(f"\nResults saved to '{FIGURES_DIR}/gridsearch_results.csv'")


# MAIN EXECUTION
def main():
    """Main execution function."""
    print("="*70)
    print("PREDICTIVE MAINTENANCE ML MODEL")
    print("="*70)
    
    # Setup
    setup_figures_directory()
    
    # Load and preprocess data
    X, y, df = load_and_preprocess_data(DATA_FILE)
    
    # Handle class imbalance with SMOTE
    X_resampled, y_resampled = handle_class_imbalance(X, y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, 
        y_resampled, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    print(f"\nTrain set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    
    # Train model with hyperparameter tuning
    grid_search = train_model(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Evaluate model
    y_pred, y_pred_proba, cm = evaluate_model(best_model, X_test, y_test)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_class_distribution(df['Machine failure'], y_resampled)
    plot_confusion_matrix(cm, y_test, y_pred)
    plot_roc_curve(y_test, y_pred_proba)
    plot_feature_importance(best_model, X.columns)
    
    # Save hyperparameter tuning results
    save_hyperparameter_results(grid_search)

    # Save trained model for Streamlit app inference
    joblib.dump(best_model, 'machine_failure_model.pkl')
    print("Saved trained model to 'machine_failure_model.pkl'")
    
    print("\n" + "="*70)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()