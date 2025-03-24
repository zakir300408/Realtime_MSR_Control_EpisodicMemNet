import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, 
    balanced_accuracy_score  # Added this import explicitly
)
from collections import Counter
import xgboost as xgb
from scipy.stats import wilcoxon, friedmanchisquare, ttest_rel
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import json
import os
import sys
from io import StringIO
import contextlib

def majority_class_baseline(y_true):
    """
    Return predictions that always choose the majority class.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    
    Returns:
    --------
    array-like : Predictions always choosing the majority class
    """
    counts = Counter(y_true)
    majority_label = counts.most_common(1)[0][0]
    predictions = np.full_like(y_true, majority_label)
    return predictions

def random_classifier_baseline(y_true, class_distribution=None):
    """
    Return random predictions based on the provided class distribution.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    class_distribution : array-like, optional
        Probability distribution over classes. If None, computes from y_true
    
    Returns:
    --------
    array-like : Random predictions
    """
    unique, counts = np.unique(y_true, return_counts=True)
    if class_distribution is None:
        class_distribution = counts / counts.sum()
    predictions = np.random.choice(unique, size=len(y_true), p=class_distribution)
    return predictions

def evaluate_model(model_name, y_true, y_pred, print_results=True):
    """
    Evaluate a model's predictions using standard classification metrics.
    
    Parameters:
    -----------
    model_name : str
        Name of the model being evaluated
    y_true : array-like
        True labels for the test set
    y_pred : array-like
        Predicted labels from the model (MUST be predictions on test data only)
    print_results : bool, default=True
        Whether to print the results
        
    Returns:
    --------
    dict : Dictionary containing the evaluation metrics
    """
    # Verify that y_true and y_pred have the same length
    if len(y_true) != len(y_pred):
        raise ValueError(f"Mismatch in lengths: y_true {len(y_true)} != y_pred {len(y_pred)}")
    
    # Verify input types and shape
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Verify there's no NaN or infinity values in predictions
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError(f"Invalid values detected in predictions: NaN or infinity found in {model_name}")
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Calculate balanced accuracy to handle class imbalance
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    try:
        conf_mat = confusion_matrix(y_true, y_pred)
    except:
        conf_mat = None
    
    # Use zero_division=0 for classification_report to avoid warnings
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    
    if print_results:
        print(f"\n--- Evaluation for {model_name} ---")
        print(f"Standard Accuracy: {acc:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print("Confusion Matrix:")
        print(conf_mat)
        print("Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
    
    result = {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': conf_mat,
        'classification_report': report,
        'kappa': kappa
    }
    
    return result

def evaluate_baseline_models(y_true):
    """
    Evaluate simple baseline models (majority and random).
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics for baseline models
    """
    metrics = {}
    
    # Majority baseline
    maj_preds = majority_class_baseline(y_true)
    metrics['majority'] = evaluate_model("Majority Class Baseline", y_true, maj_preds)
    
    # Random classifier baseline
    unique, counts = np.unique(y_true, return_counts=True)
    class_dist = counts / counts.sum()
    rand_preds = random_classifier_baseline(y_true, class_distribution=class_dist)
    metrics['random'] = evaluate_model("Random Classifier Baseline", y_true, rand_preds)
    
    # Print class distribution
    print("\nClass distribution:")
    for cls, freq in zip(unique, class_dist):
        print(f"  Class {cls}: {freq:.3f}")
        
    return metrics

def evaluate_multinomial_logistic_regression(X_train, y_train, X_test, y_test, output_name=""):
    """
    Train and evaluate multinomial logistic regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training input features
    y_train : array-like
        Training target variable
    X_test : array-like
        Test input features
    y_test : array-like
        Test target variable
    output_name : str, optional
        Name of the output column for display purposes
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics and trained model
    """
    # Initialize and train model
    lr_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = lr_model.predict(X_test)
    
    # Calculate and return metrics
    model_name = f"Multinomial Logistic Regression{' for ' + output_name if output_name else ''}"
    metrics = evaluate_model(model_name, y_test, y_pred)
    return {'metrics': metrics, 'model': lr_model}

def evaluate_random_forest(X_train, y_train, X_test, y_test, output_name=""):
    """
    Train and evaluate random forest classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training input features
    y_train : array-like
        Training target variable
    X_test : array-like
        Test input features
    y_test : array-like
        Test target variable
    output_name : str, optional
        Name of the output column for display purposes
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics and trained model
    """
    # Initialize and train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = rf_model.predict(X_test)
    
    # Calculate and return metrics
    model_name = f"Random Forest{' for ' + output_name if output_name else ''}"
    metrics = evaluate_model(model_name, y_test, y_pred)
    return {'metrics': metrics, 'model': rf_model}

def evaluate_decision_tree(X_train, y_train, X_test, y_test, output_name=""):
    """
    Train and evaluate a decision tree (CART) classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training input features
    y_train : array-like
        Training target variable
    X_test : array-like
        Test input features
    y_test : array-like
        Test target variable
    output_name : str, optional
        Name of the output column for display purposes
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics and trained model
    """
    # Initialize and train model
    dt_model = DecisionTreeClassifier(
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = dt_model.predict(X_test)
    
    # Calculate and return metrics
    model_name = f"Decision Tree (CART){' for ' + output_name if output_name else ''}"
    metrics = evaluate_model(model_name, y_test, y_pred)
    return {'metrics': metrics, 'model': dt_model}

def evaluate_knn(X_train, y_train, X_test, y_test, k=5, output_name=""):
    """
    Train and evaluate a k-Nearest Neighbors classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training input features
    y_train : array-like
        Training target variable
    X_test : array-like
        Test input features
    y_test : array-like
        Test target variable
    k : int, default=5
        Number of neighbors to use
    output_name : str, optional
        Name of the output column for display purposes
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics and trained model
    """
    # Initialize and train model
    knn_model = KNeighborsClassifier(
        n_neighbors=k,
        n_jobs=-1
    )
    knn_model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = knn_model.predict(X_test)
    
    # Calculate and return metrics
    model_name = f"k-NN (k={k}){' for ' + output_name if output_name else ''}"
    metrics = evaluate_model(model_name, y_test, y_pred)
    return {'metrics': metrics, 'model': knn_model}

def evaluate_xgboost(X_train, y_train, X_test, y_test, output_name=""):
    """
    Train and evaluate an XGBoost classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training input features
    y_train : array-like
        Training target variable
    X_test : array-like
        Test input features
    y_test : array-like
        Test target variable
    output_name : str, optional
        Name of the output column for display purposes
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics and trained model
    """
    # Initialize and train model - removed deprecated use_label_encoder parameter
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = xgb_model.predict(X_test)
    
    # Calculate and return metrics
    model_name = f"XGBoost{' for ' + output_name if output_name else ''}"
    metrics = evaluate_model(model_name, y_test, y_pred)
    return {'metrics': metrics, 'model': xgb_model}

def evaluate_all_models(X_train, y_train_dict, X_test, y_test_dict):
    """
    Evaluate all models (logistic regression, random forest) for each output column.
    
    Parameters:
    -----------
    X_train : array-like
        Training input features
    y_train_dict : dict
        Dictionary mapping column names to training target arrays
    X_test : array-like
        Test input features
    y_test_dict : dict
        Dictionary mapping column names to test target arrays
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics for all models and outputs
    """
    results = {}
    
    print("\n=== Evaluating Models for Each Output ===")
    for col_name, y_train in y_train_dict.items():
        print(f"\n--- Evaluating models for {col_name} ---")
        y_test = y_test_dict[col_name]
        
        # Store results for this output
        results[col_name] = {}
        
        # Baseline models
        baseline_metrics = evaluate_baseline_models(y_test)
        results[col_name]['baseline'] = baseline_metrics
        
        # Multinomial Logistic Regression
        lr_result = evaluate_multinomial_logistic_regression(X_train, y_train, X_test, y_test, col_name)
        results[col_name]['logistic_regression'] = lr_result
        
        # Random Forest
        rf_result = evaluate_random_forest(X_train, y_train, X_test, y_test, col_name)
        results[col_name]['random_forest'] = rf_result
        
        # Decision Tree
        dt_result = evaluate_decision_tree(X_train, y_train, X_test, y_test, col_name)
        results[col_name]['decision_tree'] = dt_result
        
        # k-Nearest Neighbors
        knn_result = evaluate_knn(X_train, y_train, X_test, y_test, k=5, output_name=col_name)
        results[col_name]['knn'] = knn_result
        
        # XGBoost
        try:
            xgb_result = evaluate_xgboost(X_train, y_train, X_test, y_test, col_name)
            results[col_name]['xgboost'] = xgb_result
        except Exception as e:
            print(f"Error evaluating XGBoost: {e}")
            print("Skipping XGBoost evaluation. Make sure the xgboost package is installed.")
        
    return results

def compare_model_performance(nn_metrics, all_model_results):
    """
    Compare performance between neural network and other models.
    
    Parameters:
    -----------
    nn_metrics : dict
        Dictionary containing neural network metrics for each output
    all_model_results : dict
        Dictionary containing metrics for all other models
    """
    print("\n=== Model Performance Comparison ===")
    
    for col_name, nn_result in nn_metrics.items():
        print(f"\n--- {col_name} ---")
        if col_name not in all_model_results:
            continue
            
        nn_acc = nn_result['accuracy']
        print(f"Neural Network Accuracy: {nn_acc:.4f}")
        
        # Compare with logistic regression
        if 'logistic_regression' in all_model_results[col_name]:
            lr_acc = all_model_results[col_name]['logistic_regression']['metrics']['accuracy']
            print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
            print(f"Difference (NN - LR): {nn_acc - lr_acc:.4f}")
        
        # Compare with random forest
        if 'random_forest' in all_model_results[col_name]:
            rf_acc = all_model_results[col_name]['random_forest']['metrics']['accuracy']
            print(f"Random Forest Accuracy: {rf_acc:.4f}")
            print(f"Difference (NN - RF): {nn_acc - rf_acc:.4f}")
        
        # Compare with decision tree
        if 'decision_tree' in all_model_results[col_name]:
            dt_acc = all_model_results[col_name]['decision_tree']['metrics']['accuracy']
            print(f"Decision Tree Accuracy: {dt_acc:.4f}")
            print(f"Difference (NN - DT): {nn_acc - dt_acc:.4f}")
        
        # Compare with k-NN
        if 'knn' in all_model_results[col_name]:
            knn_acc = all_model_results[col_name]['knn']['metrics']['accuracy']
            print(f"k-NN Accuracy: {knn_acc:.4f}")
            print(f"Difference (NN - kNN): {nn_acc - knn_acc:.4f}")
        
        # Compare with XGBoost
        if 'xgboost' in all_model_results[col_name]:
            xgb_acc = all_model_results[col_name]['xgboost']['metrics']['accuracy']
            print(f"XGBoost Accuracy: {xgb_acc:.4f}")
            print(f"Difference (NN - XGB): {nn_acc - xgb_acc:.4f}")
        
        # Compare with baseline
        if 'baseline' in all_model_results[col_name]:
            if 'majority' in all_model_results[col_name]['baseline']:
                maj_acc = all_model_results[col_name]['baseline']['majority']['accuracy']
                print(f"Majority Baseline Accuracy: {maj_acc:.4f}")
                print(f"Improvement over majority: {nn_acc - maj_acc:.4f}")

def perform_statistical_test(y_true, predictions1, predictions2, model1_name, model2_name, alpha=0.05):
    """
    Perform statistical significance test to compare two models.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    predictions1 : array-like
        Predictions from first model
    predictions2 : array-like
        Predictions from second model
    model1_name : str
        Name of first model
    model2_name : str
        Name of second model
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    dict : Dictionary containing test results
    """
    # Check for identical predictions
    if np.array_equal(predictions1, predictions2):
        print(f"Models {model1_name} and {model2_name} made identical predictions")
        return {
            'test': 'identical',
            'p_value': 1.0,
            'significant': False,
            'better_model': 'tied'
        }
    
    # Calculate accuracy for each model
    acc1 = accuracy_score(y_true, predictions1)
    acc2 = accuracy_score(y_true, predictions2)
    
    # Create binary arrays indicating correct (1) or incorrect (0) predictions
    correct1 = (predictions1 == y_true).astype(int)
    correct2 = (predictions2 == y_true).astype(int)
    
    # Contingency table for McNemar's test
    # [both correct, model1 correct & model2 incorrect]
    # [model1 incorrect & model2 correct, both incorrect]
    contingency = np.zeros((2, 2))
    contingency[0, 0] = np.sum((correct1 == 1) & (correct2 == 1))
    contingency[0, 1] = np.sum((correct1 == 1) & (correct2 == 0))
    contingency[1, 0] = np.sum((correct1 == 0) & (correct2 == 1))
    contingency[1, 1] = np.sum((correct1 == 0) & (correct2 == 0))
    
    # Perform McNemar's test (with or without correction based on sample size)
    if contingency[0, 1] + contingency[1, 0] > 25:
        # Use chi-square approximation
        result = mcnemar(contingency, exact=False, correction=True)
        test_name = "McNemar's test (with correction)"
    else:
        # Use exact binomial test
        result = mcnemar(contingency, exact=True)
        test_name = "McNemar's exact test"
    
    # Get p-value
    p_value = result.pvalue
    
    # Determine if difference is statistically significant
    significant = p_value < alpha
    
    # Determine which model is better (if significant)
    if significant:
        better_model = model1_name if acc1 > acc2 else model2_name
    else:
        better_model = "No significant difference"
    
    print(f"\n{test_name} comparing {model1_name} vs {model2_name}:")
    print(f"  Accuracy: {model1_name} = {acc1:.4f}, {model2_name} = {acc2:.4f}")
    print(f"  Contingency table:")
    print(f"    Both correct: {contingency[0, 0]}")
    print(f"    {model1_name} correct, {model2_name} incorrect: {contingency[0, 1]}")
    print(f"    {model1_name} incorrect, {model2_name} correct: {contingency[1, 0]}")
    print(f"    Both incorrect: {contingency[1, 1]}")
    print(f"  p-value: {p_value:.4f} {'(significant)' if significant else '(not significant)'}")
    print(f"  Better model: {better_model}")
    
    return {
        'test': test_name,
        'p_value': p_value,
        'significant': significant,
        'better_model': better_model,
        'contingency': contingency
    }

def compare_model_performance_with_stats(nn_metrics, all_model_results, y_test_dict, X_test, significant_threshold=0.05):
    """
    Compare performance between neural network and other models with statistical tests.
    
    Parameters:
    -----------
    nn_metrics : dict
        Dictionary containing neural network metrics for each output
    all_model_results : dict
        Dictionary containing metrics for all other models
    y_test_dict : dict
        Dictionary containing ground truth labels for test set
    X_test : array-like
        Test input features needed for predictions
    significant_threshold : float, default=0.05
        Threshold to determine statistical significance
    """
    print("\n=== Model Performance Comparison with Statistical Tests ===")
    
    # Import balanced_accuracy_score locally to ensure it's available in this function
    from sklearn.metrics import balanced_accuracy_score as bal_acc_score
    
    # Store all model predictions and statistical test results
    all_comparisons = {}
    
    # Track the consistency of performance across all outputs
    consistency_metrics = {}
    
    for col_name, nn_result in nn_metrics.items():
        print(f"\n--- {col_name} ---")
        
        if col_name not in all_model_results or col_name not in y_test_dict:
            print(f"Missing results or ground truth for {col_name}, skipping.")
            continue
        
        # Get ground truth
        y_true = y_test_dict[col_name]
        
        # Get neural network predictions
        nn_acc = nn_result['accuracy']
        nn_bal_acc = nn_result.get('balanced_accuracy', nn_acc)  # Use balanced accuracy if available
        nn_preds = nn_result.get('predictions', None)
        
        if nn_preds is None:
            print("Neural network predictions not available, statistical tests skipped.")
            continue
        
        print(f"Neural Network - Standard Accuracy: {nn_acc:.4f}, Balanced Accuracy: {nn_bal_acc:.4f}")
        
        # Store all model performances for this output
        comparisons = {}
        
        # Dictionary mapping model type to model data
        model_data = {
            'logistic_regression': ('Logistic Regression', 'lr'),
            'random_forest': ('Random Forest', 'rf'), 
            'decision_tree': ('Decision Tree', 'dt'),
            'knn': ('k-NN', 'knn'),
            'xgboost': ('XGBoost', 'xgb')
        }
        
        # Get predictions for each model
        models_info = {}
        for model_key, (model_name, short_name) in model_data.items():
            if model_key in all_model_results[col_name]:
                result = all_model_results[col_name][model_key]
                
                # Get model predictions
                if 'model' in result:
                    model = result['model']
                    
                    try:
                        # Ensure predictions are made properly
                        model_preds = model.predict(X_test)
                        
                        # Verify predictions are valid (not NaN or inf)
                        if np.any(np.isnan(model_preds)) or np.any(np.isinf(model_preds)):
                            print(f"WARNING: {model_name} produced invalid predictions for {col_name}")
                            continue
                            
                        # Calculate metrics
                        acc = accuracy_score(y_true, model_preds)
                        bal_acc = balanced_accuracy_score(y_true, model_preds)
                        
                        # Track model performance consistency across different outputs
                        if model_name not in consistency_metrics:
                            consistency_metrics[model_name] = {}
                        consistency_metrics[model_name][col_name] = {
                            'accuracy': acc, 
                            'balanced_accuracy': bal_acc
                        }
                        
                        models_info[model_name] = {
                            'accuracy': acc,
                            'balanced_accuracy': bal_acc,
                            'predictions': model_preds
                        }
                        
                        # Use balanced accuracy for more robust comparison
                        print(f"{model_name} - Standard: {acc:.4f}, Balanced: {bal_acc:.4f}")
                        print(f"Difference (NN - {short_name}): Standard: {nn_acc - acc:.4f}, Balanced: {nn_bal_acc - bal_acc:.4f}")
                    except Exception as e:
                        print(f"Error evaluating {model_name}: {e}")
                        continue
        
        # Identify close performances based on balanced accuracy (within 3%)
        close_models = {}
        for model_name, info in models_info.items():
            acc_diff = abs(nn_bal_acc - info['balanced_accuracy'])
            if acc_diff <= 0.03:  # 3% threshold
                close_models[model_name] = info
                print(f"Performance difference between Neural Network and {model_name} is close ({acc_diff:.4f})")
        
        # Perform statistical tests for close performances
        if close_models:
            print("\nPerforming statistical significance tests for close performances:")
            for model_name, info in close_models.items():
                test_result = perform_statistical_test(
                    y_true, 
                    nn_preds,
                    info['predictions'],
                    "Neural Network",
                    model_name,
                    alpha=significant_threshold
                )
                comparisons[model_name] = test_result
        
        # Compare between other models too if they have close performances
        model_names = list(models_info.keys())
        for i, model1_name in enumerate(model_names):
            for model2_name in model_names[i+1:]:
                acc_diff = abs(models_info[model1_name]['balanced_accuracy'] - models_info[model2_name]['balanced_accuracy'])
                if acc_diff <= 0.03:  # 3% threshold
                    print(f"\nPerformance difference between {model1_name} and {model2_name} is close ({acc_diff:.4f})")
                    test_result = perform_statistical_test(
                        y_true, 
                        models_info[model1_name]['predictions'],
                        models_info[model2_name]['predictions'],
                        model1_name,
                        model2_name,
                        alpha=significant_threshold
                    )
                    comparison_key = f"{model1_name} vs {model2_name}"
                    comparisons[comparison_key] = test_result
        
        # Store comparisons for this output
        all_comparisons[col_name] = comparisons
    
    # Add consistency analysis
    if consistency_metrics:
        print("\n=== Model Performance Consistency Analysis ===")
        for model_name, results in consistency_metrics.items():
            accuracies = [metrics['accuracy'] for metrics in results.values()]
            balanced_accs = [metrics['balanced_accuracy'] for metrics in results.values()]
            
            # Calculate standard deviation as a measure of consistency
            acc_std = np.std(accuracies)
            bal_acc_std = np.std(balanced_accs)
            
            print(f"{model_name} consistency:")
            print(f"  Standard Accuracy - Mean: {np.mean(accuracies):.4f}, Std Dev: {acc_std:.4f}")
            print(f"  Balanced Accuracy - Mean: {np.mean(balanced_accs):.4f}, Std Dev: {bal_acc_std:.4f}")
    
    # Return all statistical comparison results
    all_comparisons['consistency_metrics'] = consistency_metrics
    return all_comparisons

def perform_friedman_test(performances_by_model, alpha=0.05):
    """
    Perform Friedman test to compare multiple models across multiple datasets.
    
    Parameters:
    -----------
    performances_by_model : dict
        Dictionary mapping model names to lists of performance scores across datasets
    alpha : float, default=0.05
        Significance level
    
    Returns:
    --------
    dict : Dictionary containing test results and ranking information
    """
    # Extract model names and performance lists
    model_names = list(performances_by_model.keys())
    performance_lists = list(performances_by_model.values())
    
    # Ensure all lists have the same length
    list_lengths = [len(lst) for lst in performance_lists]
    if len(set(list_lengths)) > 1:
        raise ValueError("All performance lists must have the same length")
    
    # Perform Friedman test
    statistic, p_value = friedmanchisquare(*performance_lists)
    
    # Determine if differences are statistically significant
    significant = p_value < alpha
    
    print("\n=== Friedman Test for Overall Model Comparison ===")
    print(f"Test statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.4f} {'(significant)' if significant else '(not significant)'}")
    
    # Calculate model ranks
    n_datasets = len(performance_lists[0])
    ranks = np.zeros((n_datasets, len(model_names)))
    avg_ranks = None
    rank_dict = {}
    
    if significant:
        print("There are significant differences among the models.")
        
        # For each dataset, rank the models (1 = best)
        for i in range(n_datasets):
            dataset_scores = [performances[i] for performances in performance_lists]
            # Higher score = better rank (lower number)
            sorted_indices = np.argsort(-np.array(dataset_scores))
            for rank, idx in enumerate(sorted_indices):
                ranks[i, idx] = rank + 1  # +1 because ranks start at 1
        
        # Calculate average rank for each model
        avg_ranks = np.mean(ranks, axis=0)
        
        # Create a dictionary of model ranks
        rank_dict = {model_names[i]: avg_ranks[i] for i in range(len(model_names))}
        
        # Print models sorted by average rank
        sorted_indices = np.argsort(avg_ranks)
        print("\nModels ranked by average performance:")
        for idx in sorted_indices:
            print(f"{model_names[idx]}: Average rank = {avg_ranks[idx]:.2f}")
    else:
        print("No significant differences found among the models.")
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': significant,
        'model_names': model_names,
        'ranks': ranks,
        'avg_ranks': avg_ranks,
        'rank_dict': rank_dict,
        'performances_by_model': performances_by_model
    }

def summarize_model_rankings(nn_metrics, all_model_results, y_test_dict=None):
    """
    Provide a summary table of model rankings based on balanced accuracy.
    
    Parameters:
    -----------
    nn_metrics : dict
        Dictionary containing neural network metrics for each output
    all_model_results : dict
        Dictionary containing metrics for all other models
    y_test_dict : dict, optional
        Dictionary containing ground truth labels for test set (for statistical tests)
    
    Returns:
    --------
    dict : Dictionary containing ranking information and statistical test results
    """
    print("\n=== Model Ranking Summary ===")
    
    all_rankings = {}
    performances_by_model = {}
    balanced_performances_by_model = {}  # Add tracking for balanced accuracy
    
    # Dictionary to store neural network predictions
    nn_predictions = {}
    
    for col_name, nn_result in nn_metrics.items():
        if col_name not in all_model_results:
            continue
            
        # Get class distribution for this output column
        if y_test_dict and col_name in y_test_dict:
            y_test = y_test_dict[col_name]
            unique, counts = np.unique(y_test, return_counts=True)
            class_dist = counts / counts.sum()
            print(f"\n{col_name} Class Distribution:")
            for cls, freq in zip(unique, class_dist):
                print(f"  Class {cls}: {freq:.3f}")
            
            # Calculate imbalance metric
            imbalance = max(class_dist) - min(class_dist)
            print(f"  Class Imbalance: {imbalance:.3f}")
        
        # Collect accuracies for each model
        standard_accuracies = {"Neural Network": nn_result['accuracy']}
        balanced_accuracies = {"Neural Network": nn_result.get('balanced_accuracy', nn_result['accuracy'])}
        
        # Update performances_by_model dict for Friedman test
        if "Neural Network" not in performances_by_model:
            performances_by_model["Neural Network"] = []
            balanced_performances_by_model["Neural Network"] = []
        performances_by_model["Neural Network"].append(nn_result['accuracy'])
        balanced_performances_by_model["Neural Network"].append(
            nn_result.get('balanced_accuracy', nn_result['accuracy'])
        )
        
        # Collect accuracies from other models
        model_types = [
            'logistic_regression', 'random_forest', 'decision_tree', 'knn', 'xgboost'
        ]
        model_names = [
            "Logistic Regression", "Random Forest", "Decision Tree", "k-NN", "XGBoost"
        ]
        
        for model_type, model_name in zip(model_types, model_names):
            if model_type in all_model_results[col_name]:
                metrics = all_model_results[col_name][model_type]['metrics']
                acc = metrics['accuracy']
                bal_acc = metrics.get('balanced_accuracy', acc)
                
                standard_accuracies[model_name] = acc
                balanced_accuracies[model_name] = bal_acc
                
                # Update performances_by_model
                if model_name not in performances_by_model:
                    performances_by_model[model_name] = []
                    balanced_performances_by_model[model_name] = []
                performances_by_model[model_name].append(acc)
                balanced_performances_by_model[model_name].append(bal_acc)
        
        # Include baseline if available
        if 'baseline' in all_model_results[col_name] and 'majority' in all_model_results[col_name]['baseline']:
            maj_metrics = all_model_results[col_name]['baseline']['majority']
            maj_acc = maj_metrics['accuracy']
            maj_bal_acc = maj_metrics.get('balanced_accuracy', maj_acc)
            
            standard_accuracies["Majority Baseline"] = maj_acc
            balanced_accuracies["Majority Baseline"] = maj_bal_acc
            
            if "Majority Baseline" not in performances_by_model:
                performances_by_model["Majority Baseline"] = []
                balanced_performances_by_model["Majority Baseline"] = []
            performances_by_model["Majority Baseline"].append(maj_acc)
            balanced_performances_by_model["Majority Baseline"].append(maj_bal_acc)
        
        # Sort by balanced accuracy in descending order (more robust to imbalanced classes)
        sorted_balanced = sorted(balanced_accuracies.items(), key=lambda x: x[1], reverse=True)
        sorted_standard = sorted(standard_accuracies.items(), key=lambda x: x[1], reverse=True)
        
        all_rankings[col_name] = {
            'standard': sorted_standard,
            'balanced': sorted_balanced
        }
        
        # Print rankings for this output
        print(f"\n{col_name} Rankings (Balanced Accuracy):")
        for i, (model_name, acc) in enumerate(sorted_balanced, 1):
            std_acc = standard_accuracies[model_name]
            print(f"{i}. {model_name}: Balanced={acc:.4f}, Standard={std_acc:.4f}")
    
    # Overall average ranking
    print("\nAverage Model Performance Across All Outputs (Balanced Accuracy):")
    model_avg_balanced_acc = {}
    model_avg_std_acc = {}
    
    for col_rankings in all_rankings.values():
        for model_name, acc in col_rankings['balanced']:
            if model_name not in model_avg_balanced_acc:
                model_avg_balanced_acc[model_name] = []
            model_avg_balanced_acc[model_name].append(acc)
        
        for model_name, acc in col_rankings['standard']:
            if model_name not in model_avg_std_acc:
                model_avg_std_acc[model_name] = []
            model_avg_std_acc[model_name].append(acc)
    
    for model_name, accs in model_avg_balanced_acc.items():
        avg_bal_acc = sum(accs) / len(accs)
        model_avg_balanced_acc[model_name] = avg_bal_acc
    
    for model_name, accs in model_avg_std_acc.items():
        avg_std_acc = sum(accs) / len(accs)
        model_avg_std_acc[model_name] = avg_std_acc
    
    sorted_avg_bal = sorted(model_avg_balanced_acc.items(), key=lambda x: x[1], reverse=True)
    for i, (model_name, avg_bal_acc) in enumerate(sorted_avg_bal, 1):
        avg_std_acc = model_avg_std_acc.get(model_name, 0)
        print(f"{i}. {model_name}: Balanced={avg_bal_acc:.4f}, Standard={avg_std_acc:.4f}")
    
    # Perform Friedman test to compare all models across all outputs (using balanced accuracy)
    friedman_result_balanced = None
    try:
        print("\n=== Friedman Test using Balanced Accuracy ===")
        friedman_result_balanced = perform_friedman_test(balanced_performances_by_model)
    except Exception as e:
        print(f"Friedman test could not be performed: {e}")
    
    # Return complete ranking information for file output
    return {
        'rankings': all_rankings,
        'average_std_accuracy': model_avg_std_acc,
        'average_balanced_accuracy': model_avg_balanced_acc,
        'friedman_test': friedman_result_balanced,
        'performances_by_model': performances_by_model,
        'balanced_performances_by_model': balanced_performances_by_model,
        'model_ranks': friedman_result_balanced['rank_dict'] if friedman_result_balanced and 'rank_dict' in friedman_result_balanced else {}
    }

def save_results_to_file(results_data, filename):
    """
    Save evaluation results to a text file.
    
    Parameters:
    -----------
    results_data : dict
        Dictionary containing all results to save
    filename : str
        Path to the output file
    
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    try:
        # Create a string buffer to capture printed output
        buffer = StringIO()
        with contextlib.redirect_stdout(buffer):
            print("=" * 80)
            print(f"MODEL EVALUATION RESULTS")
            print(f"Generated at: {pd.Timestamp.now()}")
            print("=" * 80)
            
            # 1. Print neural network performance
            print("\n" + "=" * 80)
            print("NEURAL NETWORK PERFORMANCE")
            print("=" * 80)
            nn_metrics = results_data.get("neural_network_metrics", {})
            for col_name, metrics in nn_metrics.items():
                print(f"\n--- {col_name} ---")
                print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                print("Classification Report:")
                if isinstance(metrics.get('classification_report'), str):
                    print(metrics.get('classification_report'))
                elif isinstance(metrics.get('classification_report'), dict):
                    print(pd.DataFrame(metrics.get('classification_report')).T)
            
            # 2. Print model comparison results
            print("\n" + "=" * 80)
            print("MODEL COMPARISON RESULTS")
            print("=" * 80)
            all_model_results = results_data.get("all_model_results", {})
            for col_name, model_results in all_model_results.items():
                print(f"\n--- {col_name} ---")
                accuracies = []
                
                # Neural network accuracy
                nn_acc = nn_metrics.get(col_name, {}).get('accuracy', 'N/A')
                if nn_acc != 'N/A':
                    accuracies.append(("Neural Network", nn_acc))
                
                # Get accuracies for other models
                for model_type, result in model_results.items():
                    if model_type == 'baseline':
                        if 'majority' in result:
                            maj_acc = result['majority'].get('accuracy', 'N/A')
                            if maj_acc != 'N/A':
                                accuracies.append(("Majority Baseline", maj_acc))
                    else:
                        model_acc = result.get('metrics', {}).get('accuracy', 'N/A')
                        if model_acc != 'N/A':
                            model_name = model_type.replace('_', ' ').title()
                            accuracies.append((model_name, model_acc))
                
                # Sort by accuracy in descending order
                accuracies.sort(key=lambda x: x[1] if isinstance(x[1], (int, float)) else -float('inf'), reverse=True)
                
                # Print accuracies
                print("Model Accuracies:")
                for model_name, acc in accuracies:
                    print(f"  {model_name}: {acc:.4f}")
            
            # 3. Print statistical test results
            print("\n" + "=" * 80)
            print("STATISTICAL SIGNIFICANCE TESTS")
            print("=" * 80)
            stat_results = results_data.get("statistical_comparisons", {})
            for col_name, comparisons in stat_results.items():
                print(f"\n--- {col_name} ---")
                if not comparisons:
                    print("  No statistical comparisons performed")
                    continue
                
                for comparison_key, result in comparisons.items():
                    p_value = result.get('p_value', 'N/A')
                    significant = result.get('significant', False)
                    better_model = result.get('better_model', 'N/A')
                    
                    if p_value != 'N/A':
                        print(f"  {comparison_key}:")
                        print(f"    p-value: {p_value:.4f} {'(significant)' if significant else '(not significant)'}")
                        print(f"    Better model: {better_model}")
            
            # 4. Print ranking results
            print("\n" + "=" * 80)
            print("MODEL RANKINGS")
            print("=" * 80)
            rankings = results_data.get("rankings", {})
            
            # Print overall average performance
            avg_acc = results_data.get("ranks", {})  # Use the top-level ranks field
            if not avg_acc:  # Fall back to rankings.average_accuracy if needed
                avg_acc = rankings.get("average_accuracy", {})
                
            if avg_acc:
                print("\nOverall Average Performance:")
                sorted_avg = sorted(avg_acc.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else -float('inf'), reverse=True)
                for i, (model_name, acc) in enumerate(sorted_avg, 1):
                    if isinstance(acc, (int, float)):
                        print(f"  {i}. {model_name}: {acc:.4f}")
                    else:
                        print(f"  {i}. {model_name}: {acc}")
            
            # Print Friedman test results
            friedman_results = rankings.get("friedman_test", {})
            if friedman_results:
                p_value = friedman_results.get('p_value', 'N/A')
                significant = friedman_results.get('significant', False)
                
                print("\nFriedman Test Results:")
                if p_value != 'N/A':
                    print(f"  p-value: {p_value:.4f} {'(significant)' if significant else '(not significant)'}")
                    
                    if significant:
                        print("  There are significant differences among the models")
                        
                        # Print model ranks if available
                        model_ranks = results_data.get('model_ranks', {})
                        if model_ranks:
                            print("\n  Models ranked by average performance:")
                            for model_name, rank in model_ranks.items():
                                print(f"    {model_name}: Average rank = {rank:.2f}")
                    else:
                        print("  No significant differences found among the models")
            
            # 5. Print environment and model information
            print("\n" + "=" * 80)
            print("ENVIRONMENT AND MODEL INFORMATION")
            print("=" * 80)
            model_info = results_data.get("model_info", {})
            
            # Neural network information
            nn_info = model_info.get("neural_network", {})
            print("\nNeural Network Information:")
            for key, value in nn_info.items():
                print(f"  {key}: {value}")
            
            # Task information
            task_info = model_info.get("task_info", {})
            print("\nTask Information:")
            for key, value in task_info.items():
                print(f"  {key}: {value}")
        
        # Make sure the directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Ensure filename is valid
        try:
            # Check if we can open the file
            with open(filename, 'w') as f_test:
                pass
                
            # Write the captured output to file
            with open(filename, 'w') as f:
                f.write(buffer.getvalue())
            return True
            
        except (IOError, OSError) as e:
            print(f"Invalid filename: {e}")
            # Create a safe filename in the current directory
            base_filename = os.path.basename(filename)
            filename = f"./model_eval_{base_filename}"
            
            # Write the captured output to file
            with open(filename, 'w') as f:
                f.write(buffer.getvalue())
            return True
        
    except Exception as e:
        print(f"Error saving results to file: {e}")
        return False
