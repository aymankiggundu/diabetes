import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
import optuna
from sklearn.metrics import make_scorer
import joblib
import os
from tabulate import tabulate

# Base path for Google Colab
BASE_PATH = '/content/drive/MyDrive/MACHINE LEARNING PROJECT/EGG_dataset/egg_datasbase'

# Create directories for saving models and results
os.makedirs(f'{BASE_PATH}/models', exist_ok=True)
os.makedirs(f'{BASE_PATH}/results', exist_ok=True)

def check_class_balance(y):
    """Check if dataset has enough samples of each class."""
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        raise ValueError("Dataset must contain at least 2 classes")
    if any(counts < 2):
        raise ValueError("Each class must have at least 2 samples")
    
    # Print class distribution for debugging
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} samples ({count/len(y)*100:.2f}%)")
        
    return True

def add_gaussian_noise(X, y, noise_level=0.05):
    """Add Gaussian noise to the dataset while preserving class distribution."""
    X_noisy = X.copy()
    unique_classes, counts = np.unique(y, return_counts=True)
    
    if len(unique_classes) < 2:
        raise ValueError("Dataset must contain at least 2 classes")
    
    # Calculate class weights to maintain balance
    class_weights = dict(zip(unique_classes, counts / len(y)))
    
    for class_label in unique_classes:
        class_mask = (y == class_label)
        X_class = X[class_mask]
        
        # Scale noise by feature standard deviation and class weight
        feature_std = np.std(X_class, axis=0)
        noise_scale = noise_level * feature_std * class_weights[class_label]
        noise = np.random.normal(0, noise_scale, X_class.shape)
        
        # Ensure noise doesn't push samples too far from their class
        X_noisy[class_mask] = X_class + np.clip(noise, -2*feature_std, 2*feature_std)
    
    return X_noisy

def check_stratification(y_train, y_test):
    """Check if stratification is maintained between splits."""
    train_dist = np.bincount(y_train) / len(y_train)
    test_dist = np.bincount(y_test) / len(y_test)
    
    # Check if distributions are similar
    dist_diff = np.abs(train_dist - test_dist).mean()
    if dist_diff > 0.1:  # 10% threshold
        raise ValueError("Class distribution is not maintained between splits")
    return True

# Load and preprocess data
df = pd.read_csv(f'{BASE_PATH}/extractecd_feature_labled.csv')

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
X = df.drop(columns=[df.columns[-1]])
y = df[df.columns[-1]]

# Handle missing values and scaling
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create noisy versions of the dataset
noise_levels = [0.01, 0.05, 0.1]
X_noisy_datasets = []

for noise_level in noise_levels:
    try:
        X_noisy = add_gaussian_noise(X, y, noise_level)
        X_noisy_datasets.append(X_noisy)
    except Exception as e:
        print(f"Error creating noisy dataset with level {noise_level}: {str(e)}")
        continue

# Split each version of the dataset
datasets = {'original': X}
for i, X_noisy in enumerate(X_noisy_datasets):
    datasets[f'noisy_{noise_levels[i]}'] = X_noisy

def objective(trial, X, y, model_name):
    """Optuna objective function for hyperparameter optimization."""
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_name == "XGBoost":
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0)
        }
        model = XGBClassifier(
            **params,
            eval_metric='logloss',
            verbose=0
        )
        # Fit with validation data
        eval_set = [(X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        # Use validation set for evaluation
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)
    
    if model_name == "Logistic Regression":
        params = {
            'C': trial.suggest_float('C', 1e-5, 1.0, log=True),
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
    elif model_name == "Random Forest":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
            'class_weight': 'balanced'
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

    # Use validation set for evaluation
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)
    else:
        y_pred = model.predict(X_val)
        return accuracy_score(y_val, y_pred)

# Modify the models dictionary to include regularization and prevent overfitting
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "SVM": SVC(probability=True, C=0.1, kernel='rbf', class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced'
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='logloss'
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "ANN (MLP)": MLPClassifier(max_iter=500, early_stopping=True, validation_fraction=0.2),
    "Gradient Boosting": GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=0.8),
    "Extra Trees": ExtraTreesClassifier(min_samples_split=5, min_samples_leaf=2),
    "AdaBoost": AdaBoostClassifier(learning_rate=0.1, n_estimators=100),
    "CatBoost": CatBoostClassifier(verbose=0, learning_rate=0.1, early_stopping_rounds=20)
}

# Set up MLflow experiment
mlflow.set_experiment('Diabetes_Prediction_Experiment')

def apply_sparsity(model, amount=0.2):
    """Apply sparsity to a model using pruning."""
    if hasattr(model, 'coef_'):
        mask = np.random.rand(*model.coef_.shape) > amount
        model.coef_ = model.coef_ * mask
    elif hasattr(model, 'feature_importances_'):
        mask = np.random.rand(len(model.feature_importances_)) > amount
        model.feature_importances_ = model.feature_importances_ * mask

def knowledge_distillation(teacher_model, student_model, X_train, y_train, temperature=2.0, epochs=10):
    """Train a student model using knowledge distillation while preserving class balance."""
    # Check class balance before distillation
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError("Need at least 2 classes for distillation")
    
    # First fit the teacher model
    teacher_model.fit(X_train, y_train)
    
    # Get teacher's soft predictions
    teacher_probs = teacher_model.predict_proba(X_train)
    
    # Apply temperature scaling to soften the probabilities
    scaled_probs = np.exp(np.log(teacher_probs) / temperature)
    scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)
    
    # Initialize student model
    student_model = clone(student_model)
    
    # Set class weights for balanced learning
    class_counts = np.bincount(y_train)
    class_weights = dict(enumerate(len(y_train) / (len(unique_classes) * class_counts)))
    
    if hasattr(student_model, 'class_weight'):
        student_model.set_params(class_weight=class_weights)
    
    # Train student using soft targets
    student_model.fit(X_train, np.argmax(scaled_probs, axis=1))
    
    # Verify class balance in student predictions
    student_preds = student_model.predict(X_train)
    if len(np.unique(student_preds)) < 2:
        raise ValueError("Student model failed to learn multiple classes")
    
    return student_model

def train_and_evaluate_base_models(X_train, X_test, y_train, y_test, dataset_type):
    """Train and evaluate base models with cross-validation."""
    # Create validation set for early stopping
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    metrics_table = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Training {name} on {dataset_type} dataset...")
        start_time = time.time()

        try:
            # Check stratification before training
            check_stratification(y_train, y_test)
            
            # Special handling for XGBoost
            if name == "XGBoost":
                study = optuna.create_study(direction='maximize')
                study.optimize(
                    lambda trial: objective(trial, X_train_final, y_train_final, name),
                    n_trials=10,
                    show_progress_bar=True
                )
                
                # Create and train final model with best parameters
                best_params = study.best_params
                model = XGBClassifier(**best_params, eval_metric='logloss')
                
                # Train with validation set
                eval_set = [(X_val, y_val)]
                model.fit(
                    X_train_final,
                    y_train_final,
                    eval_set=eval_set,
                    verbose=False
                )
            
            # Handle other models normally
            elif name in ["Logistic Regression", "Random Forest"]:
                study = optuna.create_study(direction='maximize')
                study.optimize(
                    lambda trial: objective(trial, X_train_final, y_train_final, name),
                    n_trials=10,
                    show_progress_bar=True
                )
                
                if name == "Logistic Regression":
                    model = LogisticRegression(**study.best_params)
                else:
                    model = RandomForestClassifier(**study.best_params)
                model.fit(X_train_final, y_train_final)
            
            else:
                # For other models, just fit normally
                model.fit(X_train_final, y_train_final)

            # Perform cross-validation without early stopping for all models
            if name == "XGBoost":
                # For XGBoost, create a temporary model without early stopping for CV
                temp_model = XGBClassifier(
                    **{k: v for k, v in model.get_params().items() if k != 'early_stopping_rounds'},
                    verbose=0
                )
                cv_scores = cross_val_score(temp_model, X_train_final, y_train_final, cv=cv, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, X_train_final, y_train_final, cv=cv, scoring='roc_auc')

            # Train final model
            model.fit(X_train_final, y_train_final)
            training_time = time.time() - start_time

            # Evaluate base model
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
            cm = confusion_matrix(y_test, y_pred)
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

            # Store metrics in table
            metrics_table.append([
                name, accuracy, precision, recall, f1, auc, 
                sensitivity, specificity, training_time, 
                model.get_params().get('learning_rate', 'NaN'),
                cv_scores.mean(), cv_scores.std()
            ])

        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue

    metrics_df = pd.DataFrame(metrics_table, columns=[
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 
        'AUC', 'Sensitivity', 'Specificity', 'Training Time', 
        'Learning Rate', 'CV Mean Score', 'CV Std'
    ])
    return metrics_df

def train_and_evaluate_sparse_models(X_train, X_test, y_train, y_test, dataset_type):
    """Train and evaluate sparse models."""
    metrics_table = []
    for name, model in models.items():
        if name in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]:
            print(f"Training {name} (Sparse) on {dataset_type} dataset...")
            start_time = time.time()
            
            try:
                # Apply sparsity
                sparse_model = clone(model)
                apply_sparsity(sparse_model)

                # Train sparse model
                sparse_model.fit(X_train, y_train)
                training_time = time.time() - start_time

                # Evaluate sparse model
                y_pred_sparse = sparse_model.predict(X_test)
                accuracy_sparse = accuracy_score(y_test, y_pred_sparse)
                
                # Save sparse model
                model_filename = f'{BASE_PATH}/models/{dataset_type}_{name}_sparse.joblib'
                joblib.dump(sparse_model, model_filename)
                print(f"Saved sparse model to {model_filename}")

                # Store metrics in table
                metrics_table.append([
                    f"{name} (Sparse)", 
                    accuracy_sparse, 
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                    training_time, 
                    np.nan,
                    np.nan, np.nan  # Add placeholders for CV metrics
                ])
            except Exception as e:
                print(f"Error training sparse {name}: {str(e)}")
                continue

    metrics_df = pd.DataFrame(metrics_table, columns=[
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 
        'AUC', 'Sensitivity', 'Specificity', 'Training Time', 
        'Learning Rate', 'CV Mean Score', 'CV Std'
    ])
    return metrics_df

def train_and_evaluate_distilled_models(X_train, X_test, y_train, y_test, dataset_type):
    """Train and evaluate distilled models with class balance preservation."""
    metrics_table = []
    
    # Verify initial class balance
    check_class_balance(y_train)
    check_class_balance(y_test)
    
    # Only use Random Forest for distillation
    name = "Random Forest"
    model = models[name]
    print(f"Training {name} (Distilled) on {dataset_type} dataset...")
    start_time = time.time()

    try:
        # Create stratified train/val split for distillation
        X_dist_train, X_dist_val, y_dist_train, y_dist_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=42
        )
        
        # Train teacher model
        model.fit(X_dist_train, y_dist_train)
        
        # Train distilled model with class balance preservation
        student_model = clone(models["Logistic Regression"])
        distilled_model = knowledge_distillation(
            model, student_model, X_train, y_train,
            temperature=2.0
        )
        
        training_time = time.time() - start_time

        # Evaluate distilled model
        y_pred_distilled = distilled_model.predict(X_test)
        y_prob_distilled = distilled_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy_distilled = accuracy_score(y_test, y_pred_distilled)
        auc_distilled = roc_auc_score(y_test, y_prob_distilled)
        
        # Save distilled model
        model_filename = f'{BASE_PATH}/models/{dataset_type}_{name}_distilled.joblib'
        joblib.dump(distilled_model, model_filename)
        print(f"Saved distilled model to {model_filename}")
        
        metrics_table.append([
            f"{name} (Distilled)", 
            accuracy_distilled,
            np.nan, np.nan, np.nan,
            auc_distilled,
            np.nan, np.nan,
            training_time,
            np.nan,
            np.nan, np.nan  # Add placeholders for CV metrics
        ])

    except Exception as e:
        print(f"Error in distillation for {name}: {str(e)}")

    metrics_df = pd.DataFrame(metrics_table, columns=[
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 
        'AUC', 'Sensitivity', 'Specificity', 'Training Time', 
        'Learning Rate', 'CV Mean Score', 'CV Std'
    ])
    return metrics_df

def train_and_evaluate_sparse_distilled_models(X_train, X_test, y_train, y_test, dataset_type):
    """Train and evaluate sparse distilled models."""
    metrics_table = []
    
    # Only use Random Forest for sparse distillation
    name = "Random Forest"
    model = models[name]
    print(f"Training {name} (Sparse Distilled) on {dataset_type} dataset...")
    start_time = time.time()

    try:
        # Train sparse distilled model
        student_model = clone(models["Logistic Regression"])
        apply_sparsity(student_model)
        distilled_model = knowledge_distillation(model, student_model, X_train, y_train)
        training_time = time.time() - start_time

        # Evaluate sparse distilled model
        y_pred_sparse_distilled = distilled_model.predict(X_test)
        accuracy_sparse_distilled = accuracy_score(y_test, y_pred_sparse_distilled)
        
        # Save sparse distilled model
        model_filename = f'{BASE_PATH}/models/{dataset_type}_{name}_sparse_distilled.joblib'
        joblib.dump(distilled_model, model_filename)
        print(f"Saved sparse distilled model to {model_filename}")

        # Store metrics in table
        metrics_table.append([
            f"{name} (Sparse Distilled)", 
            accuracy_sparse_distilled, 
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
            training_time, 
            np.nan,
            np.nan, np.nan  # Add placeholders for CV metrics
        ])

    except Exception as e:
        print(f"Error in sparse distillation for {name}: {str(e)}")

    metrics_df = pd.DataFrame(metrics_table, columns=[
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 
        'AUC', 'Sensitivity', 'Specificity', 'Training Time', 
        'Learning Rate', 'CV Mean Score', 'CV Std'
    ])
    return metrics_df

def get_best_model(results_df):
    """Identify the best model based on training time and return its full details."""
    if results_df.empty:
        return None
    best_model = results_df.loc[results_df['Training Time'].idxmin()]
    return best_model

# Function to save comparison tables
def save_comparison_tables(all_results):
    # Create dataframes for comparison tables
    accuracy_comparison = pd.DataFrame()
    time_comparison = pd.DataFrame()
    auc_comparison = pd.DataFrame()
    
    for result_key, result_df in all_results.items():
        dataset_name = result_key.split('_')[0]
        model_type = '_'.join(result_key.split('_')[1:])
        
        if not result_df.empty:
            # Add accuracy data
            if 'Accuracy' in result_df.columns:
                accuracy_data = result_df[['Model', 'Accuracy']]
                accuracy_data['Dataset'] = dataset_name
                accuracy_data['Model Type'] = model_type
                accuracy_comparison = pd.concat([accuracy_comparison, accuracy_data])
            
            # Add timing data
            if 'Training Time' in result_df.columns:
                time_data = result_df[['Model', 'Training Time']]
                time_data['Dataset'] = dataset_name
                time_data['Model Type'] = model_type
                time_comparison = pd.concat([time_comparison, time_data])
            
            # Add AUC data
            if 'AUC' in result_df.columns:
                auc_data = result_df[['Model', 'AUC']].dropna()
                if not auc_data.empty:
                    auc_data['Dataset'] = dataset_name
                    auc_data['Model Type'] = model_type
                    auc_comparison = pd.concat([auc_comparison, auc_data])
    
    # Save tables to CSV
    if not accuracy_comparison.empty:
        accuracy_comparison.to_csv(f'{BASE_PATH}/results/accuracy_comparison.csv', index=False)
    if not time_comparison.empty:
        time_comparison.to_csv(f'{BASE_PATH}/results/time_comparison.csv', index=False)
    if not auc_comparison.empty:
        auc_comparison.to_csv(f'{BASE_PATH}/results/auc_comparison.csv', index=False)
    
    return accuracy_comparison, time_comparison, auc_comparison

# Function to create pretty tables for display
def create_comparison_tables(all_results):
    # Group results by model and dataset
    model_datasets = {}
    dataset_models = {}
    
    for result_key, result_df in all_results.items():
        if result_df.empty:
            continue
            
        dataset_name = result_key.split('_')[0]
        model_type = '_'.join(result_key.split('_')[1:])
        
        # Extract key metrics
        for _, row in result_df.iterrows():
            model_name = row['Model']
            accuracy = row['Accuracy']
            training_time = row['Training Time']
            auc = row['AUC'] if 'AUC' in row and not pd.isna(row['AUC']) else "N/A"
            
            # Organize by model
            if model_name not in model_datasets:
                model_datasets[model_name] = []
            model_datasets[model_name].append({
                'Dataset': dataset_name,
                'Type': model_type,
                'Accuracy': accuracy,
                'AUC': auc,
                'Training Time': training_time
            })
            
            # Organize by dataset
            if dataset_name not in dataset_models:
                dataset_models[dataset_name] = []
            dataset_models[dataset_name].append({
                'Model': model_name,
                'Type': model_type,
                'Accuracy': accuracy,
                'AUC': auc,
                'Training Time': training_time
            })
    
    # Create tables
    model_tables = {}
    for model_name, datasets in model_datasets.items():
        table_data = [[d['Dataset'], d['Type'], f"{d['Accuracy']:.4f}", 
                      f"{d['AUC'] if d['AUC'] != 'N/A' else 'N/A'}", 
                      f"{d['Training Time']:.2f}s"] for d in datasets]
        model_tables[model_name] = tabulate(
            table_data, 
            headers=['Dataset', 'Type', 'Accuracy', 'AUC', 'Training Time'],
            tablefmt='grid'
        )
    
    dataset_tables = {}
    for dataset_name, models in dataset_models.items():
        table_data = [[m['Model'], m['Type'], f"{m['Accuracy']:.4f}", 
                      f"{m['AUC'] if m['AUC'] != 'N/A' else 'N/A'}", 
                      f"{m['Training Time']:.2f}s"] for m in models]
        dataset_tables[dataset_name] = tabulate(
            table_data, 
            headers=['Model', 'Type', 'Accuracy', 'AUC', 'Training Time'],
            tablefmt='grid'
        )
    
    return model_tables, dataset_tables

# Modify the main evaluation loop
all_results = {}

for dataset_name, X_current in datasets.items():
    print(f"\nEvaluating dataset: {dataset_name}")
    
    try:
        # Initial class balance check
        check_class_balance(y)
        
        # Split with shuffling and stratification
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_current, y, 
            test_size=0.4, 
            random_state=42, 
            stratify=y,
            shuffle=True
        )
        
        # Verify stratification
        check_stratification(y_train, y_temp)
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=0.5, 
            random_state=42, 
            stratify=y_temp,
            shuffle=True
        )
        
        # Verify stratification again
        check_stratification(y_val, y_test)
        
        # Store all results in a dictionary for this dataset
        dataset_results = {}
        
        print(f"\nTraining models on {dataset_name} dataset...")
        
        # Train and evaluate base models
        base_results = train_and_evaluate_base_models(X_train, X_test, y_train, y_test, dataset_name)
        dataset_results['base'] = base_results
        all_results[f'{dataset_name}_base'] = base_results
        
        # Train and evaluate sparse models
        sparse_results = train_and_evaluate_sparse_models(X_train, X_test, y_train, y_test, dataset_name)
        dataset_results['sparse'] = sparse_results
        all_results[f'{dataset_name}_sparse'] = sparse_results
        
        # Train and evaluate distilled models
        distilled_results = train_and_evaluate_distilled_models(X_train, X_test, y_train, y_test, dataset_name)
        dataset_results['distilled'] = distilled_results
        all_results[f'{dataset_name}_distilled'] = distilled_results
        
        # Train and evaluate sparse distilled models
        sparse_distilled_results = train_and_evaluate_sparse_distilled_models(X_train, X_test, y_train, y_test, dataset_name)
        dataset_results['sparse_distilled'] = sparse_distilled_results
        all_results[f'{dataset_name}_sparse_distilled'] = sparse_distilled_results
        
        # Print concise results table for this dataset
        print(f"\n=== Results for {dataset_name} dataset ===")
        for model_type, result_df in dataset_results.items():
            if not result_df.empty:
                print(f"\n{model_type.upper()} MODELS:")
                # Only select columns that exist
                cols_to_show = ['Model', 'Accuracy', 'Training Time']
                if 'AUC' in result_df.columns:
                    cols_to_show.append('AUC')
                if 'CV Mean Score' in result_df.columns:
                    cols_to_show.append('CV Mean Score')
                
                summary = result_df[cols_to_show]
                print(tabulate(summary, headers='keys', tablefmt='grid', showindex=False))
                
                # Save best model for each type
                if not result_df.empty:
                    best_model_row = get_best_model(result_df)
                    if best_model_row is not None:
                        best_model_name = best_model_row['Model']
                        print(f"Best {model_type} model: {best_model_name}")
            
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        continue

# Train and save Naive Bayes model on 0.05 noise level dataset
noise_level = 0.05
if f'noisy_{noise_level}' in datasets:
    X_noisy = datasets[f'noisy_{noise_level}']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_noisy, y, test_size=0.4, random_state=42, stratify=y, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp, shuffle=True
    )
    
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    
    # Save the model
    model_filename = f'{BASE_PATH}/models/noisy_{noise_level}_Naive_Bayes.joblib'
    joblib.dump(nb_model, model_filename)
    print(f"Saved Naive Bayes model trained on 0.05 noise level dataset to {model_filename}")

# Create and display comparison tables
print("\n=== MODEL COMPARISON TABLES ===")
model_tables, dataset_tables = create_comparison_tables(all_results)

print("\n=== COMPARISON BY MODEL ===")
for model_name, table in model_tables.items():
    print(f"\n{model_name}:")
    print(table)

print("\n=== COMPARISON BY DATASET ===")
for dataset_name, table in dataset_tables.items():
    print(f"\n{dataset_name}:")
    print(table)

# Save comparison tables to CSV
print("\nSaving comparison tables to CSV...")
accuracy_comparison, time_comparison, auc_comparison = save_comparison_tables(all_results)
print("Tables saved successfully!")

# Print overall best models
print("\n=== OVERALL BEST MODELS ===")
for metric, comparison in [
    ('Accuracy', accuracy_comparison), 
    ('Training Time', time_comparison),
    ('AUC', auc_comparison)
]:
    if not comparison.empty:
        if metric == 'Training Time':
            best_idx = comparison[metric].idxmin()
        else:
            best_idx = comparison[metric].idxmax()
            
        best_row = comparison.iloc[best_idx]
        print(f"Best model by {metric}: {best_row['Model']} on {best_row['Dataset']} dataset ({metric}: {best_row[metric]:.4f})")
