import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/MACHINE LEARNING PROJECT/EGG_dataset/egg_datasbase/extractecd_feature_labled.csv')

# Preprocess data
X = df.drop(columns=[df.columns[-1]])  # Features
y = df[df.columns[-1]]  # Target

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Ensure y is 1-dimensional
y = np.ravel(y)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Define models with learning rate parameters where applicable
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(probability=True, C=1, kernel='rbf'),  # C acts as regularization strength
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', learning_rate=0.1),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "ANN (MLP)": MLPClassifier(max_iter=500, learning_rate_init=0.001),
    "Gradient Boosting": GradientBoostingClassifier(learning_rate=0.1),
    "Extra Trees": ExtraTreesClassifier(),
    "AdaBoost": AdaBoostClassifier(learning_rate=0.1),
    "CatBoost": CatBoostClassifier(verbose=0, learning_rate=0.1)
}

# Hyperparameter tuning grids for models
param_grids = {
    "Logistic Regression": {'C': [0.1, 3, 14], 'solver': ['lbfgs', 'liblinear']},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']},
    "Decision Tree": {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]},
    "Random Forest": {'n_estimators': [56, 88, 157], 'max_depth': [4, 6, 8], 'min_samples_split': [2, 5, 10]},
    "XGBoost": {'n_estimators': [66, 155, 156], 'learning_rate': [0.02, 0.5, 0.6], 'max_depth': [3, 5, 7]},
    "KNN": {'n_neighbors': [3, 5, 7], 'metric': ['euclidean', 'manhattan']},
    "Naive Bayes": {},  # Naive Bayes typically doesn't need much tuning
    "ANN (MLP)": {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']},
    "Gradient Boosting": {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]},
    "Extra Trees": {'n_estimators': [80, 97, 167], 'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]},
    "AdaBoost": {'n_estimators': [66, 86], 'learning_rate': [0.03, 0.4, 0.5]},
    "CatBoost": {'learning_rate': [0.01, 0.1, 0.3], 'iterations': [40, 89, 145]}
}

# Set up MLflow experiment
mlflow.set_experiment('Diabetes_Prediction_Experiment')

def plot_side_by_side(cm, fpr, tpr, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - {model_name}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    axes[1].plot(fpr, tpr, label=f'ROC Curve - {model_name}')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC Curve - {model_name}')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()
# Define distilled models (for example, using simpler models)
distilled_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Define sparse distilled models (make sure they are defined with appropriate hyperparameters)
sparse_distilled_models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Classifier': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}
def apply_sparsity(model, amount=0.2):
    """Apply sparsity to a model using pruning."""
    if hasattr(model, 'coef_'):
        mask = np.random.rand(*model.coef_.shape) > amount
        model.coef_ = model.coef_ * mask
    elif hasattr(model, 'feature_importances_'):
        mask = np.random.rand(len(model.feature_importances_)) > amount
        model.feature_importances_ = model.feature_importances_ * mask



def knowledge_distillation(teacher_model, student_model, X_train, y_train, temperature=2.0, epochs=10):
    """Train a student model using knowledge distillation."""
    student_model = clone(student_model)

    # Get teacher's soft predictions (probabilities)
    teacher_probs = teacher_model.predict_proba(X_train)

    # Convert soft labels to hard labels for models that do not support soft labels
    teacher_labels = np.argmax(teacher_probs, axis=1)

    # Train the student model
    if hasattr(student_model, 'partial_fit'):
        # For models that support incremental learning (e.g., SGDClassifier)
        student_model.partial_fit(X_train, teacher_labels, classes=np.unique(y_train))
    else:
        # For models that do not support incremental learning (e.g., LogisticRegression)
        student_model.fit(X_train, teacher_labels)

    return student_model

def tune_model(model, param_grid, X_train, y_train):
    """Tuning model using GridSearchCV or RandomizedSearchCV."""
    if len(param_grid) > 0:
        grid_search = GridSearchCV(model, param_grid, cv=3, verbose=1, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        return best_model, grid_search.best_params_
    else:
        return model, {}

def train_and_evaluate_base_models(X_train, X_test, y_train, y_test, dataset_type):
    """Train and evaluate base models."""
    metrics_table = []
    for name, model in models.items():
        print(f"Training {name} on {dataset_type} dataset...")
        start_time = time.time()

        # Hyperparameter tuning
        best_model, best_params = tune_model(model, param_grids.get(name, {}), X_train, y_train)

        # Ensure the model is properly fitted
        best_model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Evaluate base model
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        cm = confusion_matrix(y_test, y_pred)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        # Store metrics in table
        metrics_table.append([name, accuracy, precision, recall, f1, auc, sensitivity, specificity, training_time, best_params.get('learning_rate', 'NaN')])

        # Plot confusion matrix and ROC curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plot_side_by_side(cm, fpr, tpr, name)
        else:
            plot_side_by_side(cm, [], [], name)

    metrics_df = pd.DataFrame(metrics_table, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC', 'Sensitivity', 'Specificity', 'Training Time', 'Learning Rate'])
    return metrics_df

def train_and_evaluate_sparse_models(X_train, X_test, y_train, y_test, dataset_type):
    """Train and evaluate sparse models."""
    metrics_table = []
    for name, model in models.items():
        if name in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]:
            print(f"Training {name} (Sparse) on {dataset_type} dataset...")
            start_time = time.time()

            # Apply sparsity
            sparse_model = clone(model)
            apply_sparsity(sparse_model)

            # Train sparse model
            sparse_model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Evaluate sparse model
            y_pred_sparse = sparse_model.predict(X_test)
            accuracy_sparse = accuracy_score(y_test, y_pred_sparse)

            # Store metrics in table
            metrics_table.append([f"{name} (Sparse)", accuracy_sparse, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, training_time, np.nan])

    metrics_df = pd.DataFrame(metrics_table, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC', 'Sensitivity', 'Specificity', 'Training Time', 'Learning Rate'])
    return metrics_df

def train_and_evaluate_distilled_models(X_train, X_test, y_train, y_test, dataset_type):
    """Train and evaluate distilled models."""
    metrics_table = []
    for name, model in distilled_models.items():
        print(f"Training distilled {name} on {dataset_type} dataset...")
        start_time = time.time()

        # Hyperparameter tuning for distilled models (if needed)
        best_model, best_params = tune_model(model, param_grids.get(name, {}), X_train, y_train)

        # Ensure the model is fitted properly
        best_model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Evaluate distilled model
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        cm = confusion_matrix(y_test, y_pred)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        # Store metrics in table
        metrics_table.append([name, accuracy, precision, recall, f1, auc, sensitivity, specificity, training_time, best_params.get('learning_rate', 'NaN')])

        # Plot confusion matrix and ROC curve for distilled models
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plot_side_by_side(cm, fpr, tpr, name)
        else:
            plot_side_by_side(cm, [], [], name)

    metrics_df = pd.DataFrame(metrics_table, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC', 'Sensitivity', 'Specificity', 'Training Time', 'Learning Rate'])
    return metrics_df


def train_and_evaluate_sparse_distilled_models(X_train, X_test, y_train, y_test, dataset_type):
    """Train and evaluate sparse distilled models."""
    metrics_table = []
    for name, model in sparse_distilled_models.items():
        print(f"Training sparse distilled {name} on {dataset_type} dataset...")
        start_time = time.time()

        # Hyperparameter tuning for sparse distilled models (if needed)
        best_model, best_params = tune_model(model, param_grids.get(name, {}), X_train, y_train)

        # Ensure the model is fitted properly before evaluation
        best_model.fit(X_train, y_train)  # Fit the model here
        training_time = time.time() - start_time

        # Evaluate sparse distilled model
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        cm = confusion_matrix(y_test, y_pred)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        # Store metrics in table
        metrics_table.append([name, accuracy, precision, recall, f1, auc, sensitivity, specificity, training_time, best_params.get('learning_rate', 'NaN')])

        # Plot confusion matrix and ROC curve for sparse distilled models
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plot_side_by_side(cm, fpr, tpr, name)
        else:
            plot_side_by_side(cm, [], [], name)

    metrics_df = pd.DataFrame(metrics_table, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC', 'Sensitivity', 'Specificity', 'Training Time', 'Learning Rate'])
    return metrics_df


def get_best_model(results_df):
    """Identify the best model based on training time and return its full details."""
    best_model = results_df.loc[results_df['Training Time'].idxmin()]
    return best_model

# Train and evaluate on original dataset
print("\nTraining and evaluating base models on original dataset...")
original_base_results = train_and_evaluate_base_models(X_train, X_test, y_train, y_test, "Original")

print("\nTraining and evaluating sparse models on original dataset...")
original_sparse_results = train_and_evaluate_sparse_models(X_train, X_test, y_train, y_test, "Original")

print("\nTraining and evaluating distilled models on original dataset...")
original_distilled_results = train_and_evaluate_distilled_models(X_train, X_test, y_train, y_test, "Original")

print("\nTraining and evaluating sparse distilled models on original dataset...")
original_sparse_distilled_results = train_and_evaluate_sparse_distilled_models(X_train, X_test, y_train, y_test, "Original")

# Apply PCA for denoising
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)
X_train_pca, X_temp_pca, y_train_pca, y_temp_pca = train_test_split(X_pca, y, test_size=0.4, random_state=42)
X_val_pca, X_test_pca, y_val_pca, y_test_pca = train_test_split(X_temp_pca, y_temp_pca, test_size=0.5, random_state=42)

# Train and evaluate on denoised dataset
print("\nTraining and evaluating base models on denoised dataset...")
denoised_base_results = train_and_evaluate_base_models(X_train_pca, X_test_pca, y_train_pca, y_test_pca, "Denoised")

print("\nTraining and evaluating sparse models on denoised dataset...")
denoised_sparse_results = train_and_evaluate_sparse_models(X_train_pca, X_test_pca, y_train_pca, y_test_pca, "Denoised")

print("\nTraining and evaluating distilled models on denoised dataset...")
denoised_distilled_results = train_and_evaluate_distilled_models(X_train_pca, X_test_pca, y_train_pca, y_test_pca, "Denoised")

print("\nTraining and evaluating sparse distilled models on denoised dataset...")
denoised_sparse_distilled_results = train_and_evaluate_sparse_distilled_models(X_train_pca, X_test_pca, y_train_pca, y_test_pca, "Denoised")

# Print results and identify best models with full details
print("\nOriginal Dataset - Base Models:")
print(original_base_results.to_string(index=False))
best_base_original = get_best_model(original_base_results)
print("\nBest Base Model (Original Dataset) based on Training Time:")
print(best_base_original)

print("\nOriginal Dataset - Sparse Models:")
print(original_sparse_results.to_string(index=False))
best_sparse_original = get_best_model(original_sparse_results)
print("\nBest Sparse Model (Original Dataset) based on Training Time:")
print(best_sparse_original)

print("\nOriginal Dataset - Distilled Models:")
print(original_distilled_results.to_string(index=False))
best_distilled_original = get_best_model(original_distilled_results)
print("\nBest Distilled Model (Original Dataset) based on Training Time:")
print(best_distilled_original)

print("\nOriginal Dataset - Sparse Distilled Models:")
print(original_sparse_distilled_results.to_string(index=False))
best_sparse_distilled_original = get_best_model(original_sparse_distilled_results)
print("\nBest Sparse Distilled Model (Original Dataset) based on Training Time:")
print(best_sparse_distilled_original)

print("\nDenoised Dataset - Base Models:")
print(denoised_base_results.to_string(index=False))
best_base_denoised = get_best_model(denoised_base_results)
print("\nBest Base Model (Denoised Dataset) based on Training Time:")
print(best_base_denoised)

print("\nDenoised Dataset - Sparse Models:")
print(denoised_sparse_results.to_string(index=False))
best_sparse_denoised = get_best_model(denoised_sparse_results)
print("\nBest Sparse Model (Denoised Dataset) based on Training Time:")
print(best_sparse_denoised)

print("\nDenoised Dataset - Distilled Models:")
print(denoised_distilled_results.to_string(index=False))
best_distilled_denoised = get_best_model(denoised_distilled_results)
print("\nBest Distilled Model (Denoised Dataset) based on Training Time:")
print(best_distilled_denoised)

print("\nDenoised Dataset - Sparse Distilled Models:")
print(denoised_sparse_distilled_results.to_string(index=False))
best_sparse_distilled_denoised = get_best_model(denoised_sparse_distilled_results)
print("\nBest Sparse Distilled Model (Denoised Dataset) based on Training Time:")
print(best_sparse_distilled_denoised)

