import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
from torch.utils.data import TensorDataset, DataLoader
import sys  
import random  # Added for reproducibility seed
import os  # For file path operations
import datetime  # For timestamping output files
from evaluation_models import (
    evaluate_baseline_models,
    evaluate_multinomial_logistic_regression,
    evaluate_random_forest,
    evaluate_decision_tree,
    evaluate_knn,
    evaluate_xgboost, 
    compare_model_performance,
    summarize_model_rankings,
    compare_model_performance_with_stats,
    save_results_to_file,  
    evaluate_model  
)
from hyperparameter_optimization import (
    optimize_neural_network,
    optimize_multiple_xgboost,
    optimize_multiple_models
)

def prepare_data(file_path):
    """
    Data preparation with multiple inputs and discrete outputs.
    Maps each numeric output to a discrete index by matching to a known discrete set.
    Now stores scaling factors for deltas.
    """
    df = pd.read_csv(file_path)
    
    # Store scaling factors for deltas
    delta_scaling_factors = {}
    delta_columns = ['delta_x', 'delta_y', 'delta_angle']
    for col in delta_columns:
        max_abs = np.abs(df[col]).max()
        delta_scaling_factors[col] = max_abs if max_abs > 0 else 1.0
        if max_abs > 0:
            df[col] = df[col] / max_abs
    
    # Enforce physical constraint: zero amplitude => zero phase.
    for axis in ['x', 'y', 'z']:
        mask = df[f'next_amplitude_value_{axis}'] == 0
        df.loc[mask, f'next_phase_value_{axis}'] = 0
    
    # Grouping based on inputs (including previous phase and amplitude values)
    input_cols = [
        'pre_normalized_angle', 'delta_x', 'delta_y', 'delta_angle',
        'prev_phase_value_x', 'prev_phase_value_y', 'prev_phase_value_z',
        'prev_amplitude_value_x', 'prev_amplitude_value_y', 'prev_amplitude_value_z'
    ]
    output_columns = [
        'next_phase_value_x', 'next_phase_value_y', 'next_phase_value_z',
        'next_amplitude_value_x', 'next_amplitude_value_y', 'next_amplitude_value_z'
    ]
    multi_output_groups = df.groupby(input_cols)[output_columns].apply(lambda x: x.drop_duplicates().shape[0])
    multi_output_count = (multi_output_groups > 1).sum()
    print(f"Number of inputs with multiple distinct outputs: {multi_output_count}")
    # Removed filtering for multiple distinct outputs:
    df = df.groupby(input_cols).filter(lambda g: g[output_columns].drop_duplicates().shape[0] == 1)
    
    # Re-compute inputs on filtered data.
    inputs = np.column_stack((
        df['pre_normalized_angle'],
        df['delta_x'],
        df['delta_y'],
        df['delta_angle'],
        df['prev_phase_value_x'],
        df['prev_phase_value_y'],
        df['prev_phase_value_z'],
        df['prev_amplitude_value_x'],
        df['prev_amplitude_value_y'],
        df['prev_amplitude_value_z']
    ))
    
    # Define outputs and discrete labels for each output column.
    outputs = {}
    discrete_labels = {}
    
    for col in output_columns:
        dvals = np.sort(df[col].unique())
        discrete_labels[col] = dvals
        # For each sample, store the index corresponding to the value in dvals.
        col_indices = df[col].apply(lambda x: np.where(dvals == x)[0][0]).values
        outputs[col] = col_indices
    
    return inputs, outputs, discrete_labels, delta_scaling_factors

# Multi-head network with a shared backbone.
class MultiHeadClassifier(nn.Module):
    def __init__(self, n_inputs=10, n_hidden=64, output_info=None, discrete_labels=None, 
                 max_values=None, delta_scaling_factors=None):
        super(MultiHeadClassifier, self).__init__()
        # Shared feature extractor
        self.shared_fc1 = nn.Linear(n_inputs, n_hidden)
        self.shared_fc2 = nn.Linear(n_hidden, n_hidden)
        # Task-specific heads
        self.heads = nn.ModuleDict({
            col: nn.Linear(n_hidden, num_classes) for col, num_classes in output_info.items()
        })
        self.criterion = nn.CrossEntropyLoss()
        # Save class mapping information in the model
        self.discrete_labels = discrete_labels if discrete_labels is not None else {}
        self.max_values = max_values if max_values is not None else {}
        self.delta_scaling_factors = delta_scaling_factors if delta_scaling_factors is not None else {}

    def forward(self, x):
        x = torch.relu(self.shared_fc1(x))
        shared_repr = torch.relu(self.shared_fc2(x))
        outputs = {}
        for col, head in self.heads.items():
            outputs[col] = head(shared_repr)
        return outputs

    def train_step(self, X, y_dict):
        self.train()
        self.zero_grad()  # Single optimizer for the entire network.
        outputs = self.forward(X)
        total_loss = 0.0
        losses = {}
        for col in outputs:
            loss = self.criterion(outputs[col], y_dict[col])
            losses[col] = loss.item()
            total_loss += loss
        total_loss.backward()
        self.optimizer.step()
        return losses

    def scale_deltas(self, X):
        """Scale delta values in input array."""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        X_scaled = X.copy()
        delta_indices = {'delta_x': 1, 'delta_y': 2, 'delta_angle': 3}
        for col, idx in delta_indices.items():
            if col in self.delta_scaling_factors and self.delta_scaling_factors[col] > 0:
                X_scaled[:, idx] = X[:, idx] / self.delta_scaling_factors[col]
        return torch.tensor(X_scaled, dtype=torch.float32)

    def descale_deltas(self, X):
        """Descale delta values in input array."""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        X_descaled = X.copy()
        delta_indices = {'delta_x': 1, 'delta_y': 2, 'delta_angle': 3}
        for col, idx in delta_indices.items():
            if col in self.delta_scaling_factors and self.delta_scaling_factors[col] > 0:
                X_descaled[:, idx] = X[:, idx] * self.delta_scaling_factors[col]
        return X_descaled

    def predict(self, X, scaled_input=False):
        """
        Make predictions with the model.
        
        Parameters:
        -----------
        X : array-like or tensor
            Input features. Can be either scaled or unscaled.
        scaled_input : bool, default=False
            If True, assumes deltas in X are already scaled.
            If False, assumes deltas in X are in their original scale and need scaling.
        
        Returns:
        --------
        dict : Predictions with (logits, probabilities) for each output.
        """
        self.eval()
        if not scaled_input:
            X = self.scale_deltas(X)
        elif not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.forward(X)
            predictions = {}
            for col, logits in outputs.items():
                logits_np = logits.cpu().numpy()
                probs_np = torch.softmax(logits, dim=1).cpu().numpy()
                predictions[col] = (logits_np, probs_np)
        return predictions

    def predict_constrained(self, X, scaled_input=False):
        """
        Predict discrete outputs subject to the constraint that for each sample, at most one output column 
        can change relative to its corresponding previous value (present in the input X).
        Previous value mapping:
           next_phase_value_x  -> input index 4
           next_phase_value_y  -> input index 5
           next_phase_value_z  -> input index 6
           next_amplitude_value_x -> input index 7
           next_amplitude_value_y -> input index 8
           next_amplitude_value_z -> input index 9
        If more than one output differs from its previous value, only the one with the highest 
        predicted confidence remains; the rest are forced to have the previous value.
        Modified to handle descaling of delta values in the input.
        """
        predictions = self.predict(X, scaled_input=scaled_input)
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        if not scaled_input:
            X_np = self.descale_deltas(X_np)  # Work with descaled values for comparison

        batch_size = X_np.shape[0]
        final_predictions = {}  # col -> list of discrete predicted values
        # mapping of output column to previous value index in X.
        prev_map = {
            "next_phase_value_x": 4,
            "next_phase_value_y": 5,
            "next_phase_value_z": 6,
            "next_amplitude_value_x": 7,
            "next_amplitude_value_y": 8,
            "next_amplitude_value_z": 9
        }
        # Process each sample in the batch.
        for i in range(batch_size):
            pred_indices = {}  # column -> predicted index
            changes = []      # list of (col, confidence, predicted_index, previous_value)
            for col, (logits, probs) in predictions.items():
                logits_sample = logits[i]
                probs_sample = probs[i]
                pred_idx = np.argmax(logits_sample)
                pred_indices[col] = pred_idx
                predicted_value = self.discrete_labels[col][pred_idx]
                # Retrieve previous value from input X using mapping.
                prev_idx = prev_map.get(col)
                prev_value = X_np[i, prev_idx]
                if predicted_value != prev_value:
                    changes.append((col, probs_sample[pred_idx], pred_idx, prev_value))
            # If more than one change, keep only the most confident change.
            if len(changes) > 1:
                # Select the output column with highest confidence.
                best = max(changes, key=lambda x: x[1])
                for col, conf, p_idx, prev_val in changes:
                    if col != best[0]:
                        # Force predicted index to that corresponding to previous value.
                        disc_arr = self.discrete_labels[col]
                        forced_idx = int(np.where(disc_arr == prev_val)[0][0])
                        pred_indices[col] = forced_idx
            # Append final discrete predictions per column.
            for col in predictions.keys():
                if col not in final_predictions:
                    final_predictions[col] = []
                final_predictions[col].append(self.discrete_labels[col][pred_indices[col]])
        # Convert lists to numpy arrays.
        for col in final_predictions:
            final_predictions[col] = np.array(final_predictions[col])
        return final_predictions

    def get_discrete_prediction(self, col, logits):
        """
        Computes the predicted normalized value and confidence scores using the saved discrete_labels and max_values.
        """
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        probs = torch.softmax(logits_tensor, dim=0).numpy()
        pred_index = np.argmax(probs)
        dvals = self.discrete_labels.get(col)
        max_val = self.max_values.get(col, 1)
        norm_vals = dvals / max_val if max_val != 0 else dvals
        predicted = norm_vals[pred_index]
        confidence_scores = {norm_vals[i]: probs[i] for i in range(len(probs))}
        return predicted, confidence_scores

def get_discrete_prediction(logits, discrete_vals, max_val):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probs = torch.softmax(logits_tensor, dim=0).numpy()
    pred_index = np.argmax(probs)
    norm_vals = discrete_vals / max_val if max_val != 0 else discrete_vals
    predicted = norm_vals[pred_index]
    confidence_scores = {norm_vals[i]: probs[i] for i in range(len(probs))}
    return predicted, confidence_scores

# Memory module - Fix to use only training data
def build_memory_train_only(X_train_tensor, y_train_tensors):
    """
    Create memory module using only training data to prevent test data leakage
    """
    memory = {}
    memory["full_input"] = X_train_tensor.cpu().numpy()
    for col in ["next_phase_value_x", "next_phase_value_y", "next_phase_value_z", 
                "next_amplitude_value_x", "next_amplitude_value_y", "next_amplitude_value_z"]:
        memory[col] = y_train_tensors[col].cpu().numpy()
    return memory

def predict_with_memory(model, X, memory, discrete_labels):
    """
    For each input sample, if the signal state (indices 4 onward) exactly matches a stored memory sample's signal state,
    and the other features (indices 0 to 4) are within ±2% difference, override the network's prediction with memorized values.
    """
    net_preds = model.predict(X)  # Get network predictions as fallback
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    
    # Descale delta values for comparison if needed
    if model.delta_scaling_factors:
        delta_indices = {'delta_x': 1, 'delta_y': 2, 'delta_angle': 3}
        X_descaled = X_np.copy()
        mem_inputs_descaled = memory["full_input"].copy()
        for col, idx in delta_indices.items():
            if col in model.delta_scaling_factors:
                scale = model.delta_scaling_factors[col]
                X_descaled[:, idx] *= scale
                mem_inputs_descaled[:, idx] *= scale
    else:
        X_descaled = X_np
        mem_inputs_descaled = memory["full_input"]

    num_samples = X_np.shape[0]
    memory_match_found = False
    tol = 0.02  # 2% tolerance
    
    for i in range(num_samples):
        test_sample = X_descaled[i]
        match_found = False
        best_index = None
        
        for j in range(mem_inputs_descaled.shape[0]):
            mem_sample = mem_inputs_descaled[j]
            if np.array_equal(test_sample[4:], mem_sample[4:]):
                differences = np.abs(test_sample[:4] - mem_sample[:4])
                tolerances = tol * (np.abs(test_sample[:4]) + 1e-6)
                if np.all(differences <= tolerances):
                    match_found = True
                    memory_match_found = True
                    best_index = j
                    break
                    
        if match_found:
            for col in discrete_labels:
                mem_index = memory[col][best_index]
                num_classes = len(discrete_labels[col])
                logits = np.full((num_classes,), -1e9)
                logits[mem_index] = 1e9
                probs = np.zeros((num_classes,))
                probs[mem_index] = 1.0
                net_preds[col][0][i] = logits
                net_preds[col][1][i] = probs
    
    if not memory_match_found:
        print("No memory match found")
        net_preds["memory_miss"] = True
    
    return net_preds

# New function for K-fold cross validation to get more robust performance estimates
def cross_validate_sk_model(model, X, y, cv=5):
    """
    Perform k-fold cross-validation for scikit-learn models to get more robust accuracy estimates.
    
    Parameters:
    -----------
    model : sklearn estimator
        The model to evaluate
    X : array-like
        Input features
    y : array-like
        Target values
    cv : int, default=5
        Number of cross-validation folds
    
    Returns:
    --------
    dict : Dictionary with cross-validation results
    """
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score
    
    # Define metrics to track
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'balanced_accuracy': make_scorer(balanced_accuracy_score)
    }
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y, 
        cv=cv,
        scoring=scoring,
        return_train_score=True
    )
    
    # Calculate mean and std of test scores
    results = {
        'test_accuracy_mean': np.mean(cv_results['test_accuracy']),
        'test_accuracy_std': np.std(cv_results['test_accuracy']),
        'test_balanced_accuracy_mean': np.mean(cv_results['test_balanced_accuracy']),
        'test_balanced_accuracy_std': np.std(cv_results['test_balanced_accuracy']),
        'train_accuracy_mean': np.mean(cv_results['train_accuracy']),
        'train_accuracy_std': np.std(cv_results['train_accuracy']),
        'train_balanced_accuracy_mean': np.mean(cv_results['train_balanced_accuracy']),
        'train_balanced_accuracy_std': np.std(cv_results['train_balanced_accuracy']),
    }
    
    # Check for overfitting (large difference between train and test scores)
    train_test_diff = results['train_accuracy_mean'] - results['test_accuracy_mean']
    results['overfitting_gap'] = train_test_diff
    results['overfitting_risk'] = 'High' if train_test_diff > 0.1 else 'Medium' if train_test_diff > 0.05 else 'Low'
    
    return results

if __name__ == "__main__":
    # Set seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parse command-line arguments.
    use_memory = "--use-memory" in sys.argv
    load_checkpoint = "--load" in sys.argv
    optimize_params = "--optimize" in sys.argv  # New flag for hyperparameter optimization
    search_type = "random"  # Default search type
    for arg in sys.argv:
        if arg.startswith("--search="):
            search_type = arg.split("=")[1]
    
    # New flag for optimizing all models
    optimize_all = "--optimize-all" in sys.argv  # Set to True to optimize all models

    # Load and prepare data.
    file_path = "RL_data_raw_post_processed.csv"
    X, y_dict, discrete_labels, delta_scaling_factors = prepare_data(file_path)
    
    # Create a consistent train/test split.
    X_train_full, X_test_full, idx_train, idx_test = train_test_split(
        X, range(len(X)), test_size=0.2, random_state=42, stratify=None
    )
    X_train = X[idx_train]
    X_test = X[idx_test]
    
    # Build y_train_dict and y_test_dict.
    y_train_dict = {}
    y_test_dict = {}
    for col_name, y_values in y_dict.items():
        y_train_dict[col_name] = y_values[idx_train]
        y_test_dict[col_name] = y_values[idx_test]

    # Convert to torch tensors.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensors = {col: torch.tensor(vals, dtype=torch.long) for col, vals in y_train_dict.items()}
    y_test_tensors = {col: torch.tensor(vals, dtype=torch.long) for col, vals in y_test_dict.items()}

    # Convert PyTorch tensors to NumPy for sklearn models - moved earlier to fix the error
    numpy_y_train_dict = {col: vals.numpy() for col, vals in y_train_tensors.items()}
    numpy_y_test_dict = {col: vals.numpy() for col, vals in y_test_tensors.items()}

    # Build the multi-output model.
    output_info = {col_name: len(discrete_labels[col_name]) for col_name in discrete_labels}
    
    # Load checkpoint if requested.
    if load_checkpoint:
        checkpoint = torch.load("checkpoint.pth")
        multi_head_net = MultiHeadClassifier(
            n_inputs=10, 
            n_hidden=64, 
            output_info=output_info, 
            discrete_labels=discrete_labels, 
            max_values={},  # Replace with appropriate max_values if available
            delta_scaling_factors=delta_scaling_factors
        )
        multi_head_net.load_state_dict(checkpoint["model_state_dict"])
        multi_head_net.optimizer = optim.Adam(multi_head_net.parameters(), lr=1e-3)
        multi_head_net.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        multi_head_net.discrete_labels = checkpoint.get("discrete_labels", {})
        multi_head_net.max_values = checkpoint.get("max_values", {})
        multi_head_net.delta_scaling_factors = checkpoint.get("delta_scaling_factors", {})
        memory_size = checkpoint["memory"]["full_input"].shape[0]
        train_size = len(X_train)
        if memory_size > train_size:
            print("WARNING: Loaded checkpoint contains memory built with test data!")
            print(f"Memory size: {memory_size}, Train size: {train_size}")
            print("Rebuilding memory with training data only to prevent test data leakage...")
            memory = build_memory_train_only(X_train_tensor, y_train_tensors)
            print("Memory rebuilt with training data only.")
        else:
            memory = checkpoint["memory"]
            print("Loaded memory contains only training data.")
    else:
        # Dictionary to store optimized models
        optimized_models = {}
        
        # Always run neural network optimization - no default option
        print("\n=== Starting Neural Network Hyperparameter Optimization ===")
        nn_best_params, multi_head_net, nn_best_score = optimize_neural_network(
            X_train, y_train_dict, output_info, discrete_labels, {}, delta_scaling_factors,
            search_type=search_type, 
            n_iter=50 if search_type == 'random' else None,  # More iterations
            cv_folds=3, 
            verbose=2  # More verbose output
        )
        
        # Print optimization results
        print(f"\nBest Neural Network Parameters: {nn_best_params}")
        print(f"Best Neural Network Validation Score: {nn_best_score:.4f}")
        
        # Always do fine-tuning with the best parameters
        print("\n=== Fine-tuning Neural Network with Best Parameters ===")
        # Ensure batch size is int
        batch_size = int(nn_best_params.get('batch_size', 64))
        # Create DataLoader for efficient batching
        train_dataset = TensorDataset(X_train_tensor, *list(y_train_tensors.values()))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Use best learning rate
        lr = nn_best_params.get('lr', 0.001)
        # Setup optimizer based on best parameters
        if nn_best_params.get('optimizer_type', 'adam').lower() == 'adam':
            multi_head_net.optimizer = optim.Adam(multi_head_net.parameters(), lr=lr)
        else:
            momentum = nn_best_params.get('momentum', 0.9)
            multi_head_net.optimizer = optim.SGD(multi_head_net.parameters(), lr=lr, momentum=momentum)
        
        # Additional training epochs for fine-tuning
        additional_epochs = 200  # Extended fine-tuning period
        best_val_loss = float('inf')
        patience = 20  # Increased patience for fine-tuning
        patience_counter = 0
        best_model_state = None
        
        print(f"Fine-tuning for up to {additional_epochs} epochs with batch size {batch_size}, learning rate {lr}")
        print(f"Using optimizer: {nn_best_params.get('optimizer_type', 'adam')}")
        
        for epoch in range(additional_epochs):
            multi_head_net.train()
            epoch_losses = {col_name: 0.0 for col_name in y_train_dict}
            
            for batch_X, *batch_y_values in train_loader:
                batch_y = {col: val for col, val in zip(y_train_tensors.keys(), batch_y_values)}
                losses = multi_head_net.train_step(batch_X, batch_y)
                for col_name, loss_val in losses.items():
                    epoch_losses[col_name] += loss_val
            
            # Print epoch results
            print(f"\nFine-tuning Epoch {epoch+1}/{additional_epochs}")
            for col_name in epoch_losses:
                avg_loss = epoch_losses[col_name] / len(train_loader)
                print(f"  {col_name} loss: {avg_loss:.4f}")
            
            # Evaluate on validation set
            multi_head_net.eval()
            val_losses = []
            with torch.no_grad():
                outputs = multi_head_net.forward(X_test_tensor)
                for col_name in outputs:
                    loss = multi_head_net.criterion(outputs[col_name], y_test_tensors[col_name])
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model during fine-tuning
                best_model_state = multi_head_net.state_dict().copy()
                print(f"New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement, patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    # Restore best model
                    if best_model_state is not None:
                        multi_head_net.load_state_dict(best_model_state)
                        print("Restored best model state from epoch with lowest validation loss")
                    break
        
        # Optimize other models if requested
        if optimize_all:
            print("\n=== Starting Optimization for All Models ===")
            all_optimized_models = optimize_multiple_models(
                X_train, numpy_y_train_dict,
                search_type=search_type,
                n_iter=50 if search_type == 'random' else None,
                cv=3,
                verbose=1
            )
            
            # Store optimized models for later use
            optimized_models.update(all_optimized_models)
            print("\n=== Model Optimization Complete ===")

        # Fix: Build memory using training data only
        memory = build_memory_train_only(X_train_tensor, y_train_tensors)
        print("Memory built from training data only to prevent test data leakage.")
        
        checkpoint = {
            "model_state_dict": multi_head_net.state_dict(),
            "optimizer_state_dict": multi_head_net.optimizer.state_dict(),
            "memory": memory,
            "discrete_labels": multi_head_net.discrete_labels,
            "max_values": multi_head_net.max_values,
            "delta_scaling_factors": delta_scaling_factors
        }
        torch.save(checkpoint, "checkpoint.pth")
        print("Checkpoint saved to checkpoint.pth")
    
    # Evaluation on the test set (without memory override).
    print("\n=== Evaluation on Test Set (Network Predictions Only) ===")
    all_metrics = {}
    nn_predictions = {}  # Store neural network predictions for statistical testing
    multi_head_net.eval()
    with torch.no_grad():
        outputs = multi_head_net.forward(X_test_tensor)
        for col_name in outputs:
            logits = outputs[col_name]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true = y_test_tensors[col_name].cpu().numpy()
            
            # Store predictions for statistical testing
            nn_predictions[col_name] = preds
            
            acc = accuracy_score(true, preds)
            conf_mat = confusion_matrix(true, preds)
            report = classification_report(true, preds, zero_division=0)
            print(f"\n--- Evaluation for {col_name} ---")
            print(f"Accuracy: {acc:.4f}")
            print("Confusion Matrix:")
            print(conf_mat)
            print("Classification Report:")
            print(report)
            all_metrics[col_name] = {
                'accuracy': acc,
                'confusion_matrix': conf_mat,
                'classification_report': report,
                'predictions': preds  # Add predictions for statistical testing
            }
    
    # Compute overall Top 2 Accuracy.
    print("\n=== Overall Top 2 Accuracy on Test Set ===")
    with torch.no_grad():
        outputs = multi_head_net.forward(X_test_tensor)
        for col_name in outputs:
            logits = outputs[col_name]
            top2_indices = torch.topk(logits, 2, dim=1).indices.cpu().numpy()
            true_indices = y_test_tensors[col_name].cpu().numpy()
            correct_top2 = sum(true in top2 for true, top2 in zip(true_indices, top2_indices))
            accuracy_top2 = correct_top2 / len(true_indices)
            print(f"{col_name} Top 2 Accuracy: {accuracy_top2:.4f}")
    

    # Test predictions on the first 5 samples.
    if use_memory:
        print("\nTest Predictions (first 5 samples) WITH Memory Integration:")
        test_predictions = predict_with_memory(multi_head_net, X_test_tensor[:5], memory, discrete_labels)
    else:
        print("\nTest Predictions (first 5 samples) WITHOUT Memory Integration:")
        test_predictions = multi_head_net.predict(X_test_tensor[:5])
    
    for i in range(5):
        print(f"\n--- Sample {i} ---")
        for col_name in test_predictions:
            if col_name == "memory_miss":
                continue
            logits_i, probs_i = test_predictions[col_name]
            logits_sample = logits_i[i]
            predicted_norm, confidences = multi_head_net.get_discrete_prediction(
                col=col_name,
                logits=logits_sample
            )
            true_index = y_test_tensors[col_name][i].item()
            if multi_head_net.max_values.get(col_name, 0) != 0:
                true_norm = discrete_labels[col_name][true_index] / multi_head_net.max_values.get(col_name, 1)
            else:
                true_norm = discrete_labels[col_name][true_index]
            print(f"{col_name}:")
            print(f"  Ground Truth (normalized) = {true_norm:.4f}")
            print(f"  Predicted (normalized)    = {predicted_norm:.4f}")
            top_2 = sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:2]
            print("  Top 2 Confidence Scores:")
            for val, conf in top_2:
                print(f"    Value={val:.4f}, Prob={conf:.3f}")
            is_top2_correct = any(abs(val - true_norm) < 1e-6 for val, _ in top_2)
            print(f"  Top 2 Prediction Correct: {'Yes' if is_top2_correct else 'No'}")
    
    # Example of how to use predict with unscaled deltas.
    print("\nTest Predictions with unscaled (original) deltas:")
    preds = multi_head_net.predict(X_test_tensor, scaled_input=False)

    # Example of how to use predict with pre-scaled deltas.
    print("\nTest Predictions with pre-scaled deltas:")
    X_scaled = multi_head_net.scale_deltas(X_test_tensor)
    preds = multi_head_net.predict(X_scaled, scaled_input=True)

    # Use the evaluation_models module for model evaluation
    print("\n=== Running Model Evaluation Using evaluation_models ===")
    # Removed the redundant conversion since it's done earlier now
    # numpy_y_train_dict and numpy_y_test_dict are already defined
    
    # Optimize XGBoost hyperparameters if requested
    optimized_xgb_models = {}
    if optimize_params:
        print("\n=== Starting XGBoost Hyperparameter Optimization ===")
        xgb_results = optimize_multiple_xgboost(
            X_train, numpy_y_train_dict, 
            search_type=search_type, 
            n_iter=50 if search_type == 'random' else None,  # Increased from 15 to be consistent
            cv=3, 
            verbose=1
        )
        
        # Store optimized XGBoost models for each output
        for col_name, (params, model, score) in xgb_results.items():
            optimized_xgb_models[col_name] = model
            print(f"\nBest XGBoost parameters for {col_name}: {params}")
            print(f"Validation score: {score:.4f}")
    
    # Add validation that there's no data leakage in model evaluation
    print("\n=== Testing and comparing all models on the same test set ===")
    print(f"Test set size: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}% of total)")
    print(f"Train set size: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}% of total)")
    print("Ensuring fair comparison across all model types with proper data separation...")
    
    # Validate train/test split is balanced and representative
    from sklearn.metrics import balanced_accuracy_score
    
    # First check if there are any samples that appear in both train and test sets
    try:
        # First check if there are any samples that appear in both train and test sets
        X_train_tuples = set(tuple(row) for row in X_train)
        X_test_tuples = set(tuple(row) for row in X_test)
        overlap = X_train_tuples.intersection(X_test_tuples)
        overlap_count = len(overlap)
        
        if overlap_count > 0:
            print(f"WARNING: Data leakage detected! {overlap_count} samples appear in both train and test sets.")
        else:
            print("Data integrity check: Train and test sets are properly separated with no overlap.")
        
        # Check that memory only contains training data
        memory_inputs = set(tuple(row) for row in memory["full_input"])
        non_train_memory = memory_inputs - X_train_tuples
        if len(non_train_memory) > 0:
            print(f"WARNING: Memory contains {len(non_train_memory)} samples not in training data!")
        else:
            print("Memory integrity check: Memory contains only training data samples.")
            
        # Check class balance in train and test sets
        print("\n=== Class Balance Validation ===")
        for col_name in y_train_dict:
            train_values, train_counts = np.unique(y_train_dict[col_name], return_counts=True)
            test_values, test_counts = np.unique(y_test_dict[col_name], return_counts=True)
            
            train_dist = train_counts / train_counts.sum()
            test_dist = test_counts / test_counts.sum()
            
            print(f"\n{col_name} class distribution:")
            print("  Train set: ", end="")
            for val, freq in zip(train_values, train_dist):
                print(f"Class {val}: {freq:.3f}, ", end="")
            print("\n  Test set:  ", end="")
            for val, freq in zip(test_values, test_dist):
                print(f"Class {val}: {freq:.3f}, ", end="")
            print()
            
            # Check if all classes in test are also in train
            missing_classes = set(test_values) - set(train_values)
            if missing_classes:
                print(f"  WARNING: Classes {missing_classes} appear in test set but not in training set!")
    except Exception as e:
        print(f"Unable to perform complete data integrity check: {e}")
    
    # Run evaluations for all models across all outputs
    print("\n=== Testing and comparing all models on the same test set ===")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Train set size: {len(X_train)} samples")
    print("Using balanced accuracy for fair comparison across all models...")
    
    # Print hashes of datasets to confirm consistency
    print(f"Training data hash: {hash(str(X_train))}")
    print(f"Test data hash: {hash(str(X_test))}")
    
    all_model_results = {}
    model_cv_results = {}  # Store cross-validation results
    
    for col_name, y_train in numpy_y_train_dict.items():
        y_test = numpy_y_test_dict[col_name]
        print(f"\n--- Evaluating models for {col_name} ---")
        
        # Verify true separation of train/test labels
        train_unique = set(y_train)
        test_unique = set(y_test)
        print(f"Training set unique labels: {train_unique}")
        print(f"Test set unique labels: {test_unique}")
        
        # Check class imbalance for this target
        unique_values, counts = np.unique(y_train, return_counts=True)
        proportions = counts / len(y_train)
        max_prop = proportions.max()
        min_prop = proportions.min()
        imbalance_ratio = max_prop / min_prop
        print(f"Class imbalance ratio: {imbalance_ratio:.2f} (max={max_prop:.2f}, min={min_prop:.2f})")
        
        # Store results for this output
        all_model_results[col_name] = {}
        model_cv_results[col_name] = {}
        
        # Baseline models
        baseline_metrics = evaluate_baseline_models(y_test)
        all_model_results[col_name]['baseline'] = baseline_metrics
        
        # Multinomial Logistic Regression
        if optimize_all and 'logistic_regression' in optimized_models and col_name in optimized_models['logistic_regression']:  # Using correct test set for evaluation
            # Use optimized logistic regression model
            _, lr_model, _ = optimized_models['logistic_regression'][col_name]
            y_pred = lr_model.predict(X_test)
            lr_metrics = evaluate_model(f"Optimized Logistic Regression for {col_name}", y_test, y_pred)
            
            # Add cross-validation results if feasible
            try:
                if len(y_train) > 10000:
                    print("  Skipping cross-validation due to large dataset...")
                    cv_results = None
                else:
                    cv_results = cross_validate_sk_model(lr_model, X_train, y_train)
                    print(f"  CV balanced accuracy: {cv_results['test_balanced_accuracy_mean']:.4f} ± {cv_results['test_balanced_accuracy_std']:.4f}")
                    if cv_results['overfitting_risk'] != 'Low':
                        print(f"  Warning: {cv_results['overfitting_risk']} risk of overfitting detected (gap: {cv_results['overfitting_gap']:.4f})")
            except Exception as e:
                print(f"  Error in cross-validation: {e}")
                cv_results = None
                
            lr_result = {'metrics': lr_metrics, 'model': lr_model, 'cv_results': cv_results}
            model_cv_results[col_name]['logistic_regression'] = cv_results
        else:
            # Use default logistic regression
            lr_result = evaluate_multinomial_logistic_regression(X_train, y_train, X_test, y_test, col_name)
        all_model_results[col_name]['logistic_regression'] = lr_result
        
        # Random Forest
        if optimize_all and 'random_forest' in optimized_models and col_name in optimized_models['random_forest']:
            # Use optimized random forest model
            _, rf_model, _ = optimized_models['random_forest'][col_name]
            y_pred = rf_model.predict(X_test)
            rf_metrics = evaluate_model(f"Optimized Random Forest for {col_name}", y_test, y_pred)
            
            # Add cross-validation results if feasible
            try:
                if len(y_train) > 1000:
                    print("  Skipping cross-validation due to large dataset...")
                    cv_results = None
                else:
                    cv_results = cross_validate_sk_model(rf_model, X_train, y_train)
                    print(f"  CV balanced accuracy: {cv_results['test_balanced_accuracy_mean']:.4f} ± {cv_results['test_balanced_accuracy_std']:.4f}")
                    if cv_results['overfitting_risk'] != 'Low':
                        print(f"  Warning: {cv_results['overfitting_risk']} risk of overfitting detected (gap: {cv_results['overfitting_gap']:.4f})")
            except Exception as e:
                print(f"  Error in cross-validation: {e}")
                cv_results = None
                
            rf_result = {'metrics': rf_metrics, 'model': rf_model, 'cv_results': cv_results}
            model_cv_results[col_name]['random_forest'] = cv_results
        else:
            # Use default random forest
            rf_result = evaluate_random_forest(X_train, y_train, X_test, y_test, col_name)
        all_model_results[col_name]['random_forest'] = rf_result
        
        # Decision Tree
        if optimize_all and 'decision_tree' in optimized_models and col_name in optimized_models['decision_tree']:
            # Use optimized decision tree model
            _, dt_model, _ = optimized_models['decision_tree'][col_name]
            y_pred = dt_model.predict(X_test)
            dt_metrics = evaluate_model(f"Optimized Decision Tree for {col_name}", y_test, y_pred)
            
            # Add cross-validation results if feasible
            try:
                if len(y_train) > 1000:
                    print("  Skipping cross-validation due to large dataset...")
                    cv_results = None
                else:
                    cv_results = cross_validate_sk_model(dt_model, X_train, y_train)
                    print(f"  CV balanced accuracy: {cv_results['test_balanced_accuracy_mean']:.4f} ± {cv_results['test_balanced_accuracy_std']:.4f}")
                    if cv_results['overfitting_risk'] != 'Low':
                        print(f"  Warning: {cv_results['overfitting_risk']} risk of overfitting detected (gap: {cv_results['overfitting_gap']:.4f})")
            except Exception as e:
                print(f"  Error in cross-validation: {e}")
                cv_results = None
                
            dt_result = {'metrics': dt_metrics, 'model': dt_model, 'cv_results': cv_results}
            model_cv_results[col_name]['decision_tree'] = cv_results
        else:
            # Use default decision tree
            dt_result = evaluate_decision_tree(X_train, y_train, X_test, y_test, col_name)
        all_model_results[col_name]['decision_tree'] = dt_result
        
        # k-Nearest Neighbors
        if optimize_all and 'knn' in optimized_models and col_name in optimized_models['knn']:
            # Use optimized kNN model
            _, knn_model, _ = optimized_models['knn'][col_name]
            y_pred = knn_model.predict(X_test)
            knn_metrics = evaluate_model(f"Optimized k-NN for {col_name}", y_test, y_pred)
            
            # Add cross-validation results if feasible
            try:
                if len(y_train) > 1000:
                    print("  Skipping cross-validation due to large dataset...")
                    cv_results = None
                else:
                    cv_results = cross_validate_sk_model(knn_model, X_train, y_train)
                    print(f"  CV balanced accuracy: {cv_results['test_balanced_accuracy_mean']:.4f} ± {cv_results['test_balanced_accuracy_std']:.4f}")
                    if cv_results['overfitting_risk'] != 'Low':
                        print(f"  Warning: {cv_results['overfitting_risk']} risk of overfitting detected (gap: {cv_results['overfitting_gap']:.4f})")
            except Exception as e:
                print(f"  Error in cross-validation: {e}")
                cv_results = None
                
            knn_result = {'metrics': knn_metrics, 'model': knn_model, 'cv_results': cv_results}
            model_cv_results[col_name]['knn'] = cv_results
        else:
            # Use default kNN
            knn_result = evaluate_knn(X_train, y_train, X_test, y_test, k=5, output_name=col_name)
        all_model_results[col_name]['knn'] = knn_result
        
        # XGBoost
        try:
            if (optimize_params or optimize_all) and 'xgboost' in optimized_models and col_name in optimized_models['xgboost']:
                # Use optimized XGBoost model
                _, xgb_model, _ = optimized_models['xgboost'][col_name]
                y_pred = xgb_model.predict(X_test)
                xgb_metrics = evaluate_model(f"Optimized XGBoost for {col_name}", y_test, y_pred)
                
                # Add cross-validation results if feasible
                try:
                    if len(y_train) > 1000:
                        print("  Skipping cross-validation due to large dataset...")
                        cv_results = None
                    else:
                        cv_results = cross_validate_sk_model(xgb_model, X_train, y_train)
                        print(f"  CV balanced accuracy: {cv_results['test_balanced_accuracy_mean']:.4f} ± {cv_results['test_balanced_accuracy_std']:.4f}")
                        if cv_results['overfitting_risk'] != 'Low':
                            print(f"  Warning: {cv_results['overfitting_risk']} risk of overfitting detected (gap: {cv_results['overfitting_gap']:.4f})")
                except Exception as e:
                    print(f"  Error in cross-validation: {e}")
                    cv_results = None
                    
                xgb_result = {'metrics': xgb_metrics, 'model': xgb_model, 'cv_results': cv_results}
                model_cv_results[col_name]['xgboost'] = cv_results
            else:
                # Use default XGBoost
                xgb_result = evaluate_xgboost(X_train, y_train, X_test, y_test, col_name)
            
            all_model_results[col_name]['xgboost'] = xgb_result
            
        except Exception as e:
            print(f"Error evaluating XGBoost: {e}")
            print("Skipping XGBoost evaluation. Make sure the xgboost package is installed.")
    
    # Compare neural network performance with other models (basic comparison)
    compare_model_performance(all_metrics, all_model_results)
    
    # Compare performance with statistical tests for close performances
    stat_results = compare_model_performance_with_stats(
        all_metrics, 
        all_model_results, 
        numpy_y_test_dict,
        X_test,  # Pass X_test to the function
        significant_threshold=0.05
    )
    
    # Generate a summary ranking of all models with statistical analysis
    ranking_results = summarize_model_rankings(all_metrics, all_model_results, numpy_y_test_dict)
    
    # Save all results to a text file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a valid file path using os.path.join and normpath to handle path correctly
    results_dir = os.path.normpath("e:/RL_realtime_IEEE TSME")
    results_filename = os.path.join(results_dir, f"model_evaluation_{timestamp}.txt")
    
    # Ensure the directory exists
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
            print(f"Created directory: {results_dir}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            # Fallback to current directory
            results_filename = f"model_evaluation_{timestamp}.txt"
    
    # Collect all results to be saved
    results_data = {
        "neural_network_metrics": all_metrics,
        "all_model_results": all_model_results,
        "statistical_comparisons": stat_results,
        "rankings": ranking_results,
        # Include rank information directly in the top level for easier access
        "ranks": ranking_results.get('model_ranks', {}),
        "performances_by_model": ranking_results.get('performances_by_model', {}),
        "model_info": {
            "neural_network": {
                "type": "MultiHeadClassifier",
                "hidden_size": 64,
                "num_outputs": len(output_info)
            },
            "task_info": {
                "num_samples": len(X),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "input_features": X.shape[1],
                "output_columns": list(discrete_labels.keys())
            }
        }
    }
    
    # Save results
    save_success = save_results_to_file(results_data, results_filename)
    if save_success:
        print(f"\nResults saved to: {results_filename}")
    else:
        print(f"\nFailed to save results to: {results_filename}")
        # Try saving to a simpler location as fallback
        fallback_filename = f"model_evaluation_{timestamp}.txt"
        fallback_success = save_results_to_file(results_data, fallback_filename)
        if fallback_success:
            print(f"Results saved to fallback location: {fallback_filename}")
