import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from scipy.stats import randint, uniform
import time
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_evaluate_nn(model, X_train, y_train, X_val, y_val, params, device='cpu', verbose=0):
    """
    Train and evaluate a neural network with given hyperparameters
    
    Parameters:
    -----------
    model : torch.nn.Module
        Neural network model
    X_train : array-like
        Training features
    y_train : dict
        Training labels for each output
    X_val : array-like
        Validation features
    y_val : dict
        Validation labels for each output
    params : dict
        Hyperparameters including:
        - lr: learning rate
        - batch_size: batch size
        - optimizer_type: optimizer type ('adam', 'sgd')
        - hidden_size: size of hidden layers
        - dropout_rate: dropout rate (if applicable)
    device : str
        Device to run training on ('cpu' or 'cuda')
    verbose : int, default=0
        Verbosity level
        
    Returns:
    --------
    tuple : (fitted model, validation accuracy, training time)
    """
    # Extract parameters
    lr = params.get('lr', 0.001)
    # Ensure batch_size is a Python int, not numpy int
    batch_size = int(params.get('batch_size', 64))
    optimizer_type = params.get('optimizer_type', 'adam')
    
    # Move data to device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    y_train_tensors = {col: torch.tensor(vals, dtype=torch.long).to(device) 
                      for col, vals in y_train.items()}
    y_val_tensors = {col: torch.tensor(vals, dtype=torch.long).to(device) 
                    for col, vals in y_val.items()}
    
    # Setup optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'sgd':
        momentum = params.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.optimizer = optimizer
    
    # Create DataLoader for batching with explicit Python int type
    train_dataset = TensorDataset(X_train_tensor, *list(y_train_tensors.values()))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=5)
    
    # Training loop
    epochs = params.get('max_epochs', 100)
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, *batch_y_values in train_loader:
            batch_y = {col: val for col, val in zip(y_train_tensors.keys(), batch_y_values)}
            
            # Forward pass
            outputs = model(batch_X)
            
            # Calculate loss
            loss = 0
            for col in outputs:
                loss += model.criterion(outputs[col], batch_y[col])
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            outputs = model(X_val_tensor)
            for col in outputs:
                val_loss += model.criterion(outputs[col], y_val_tensors[col])
        
        # Print progress info during optimization if verbose >= 2
        if verbose >= 2 and epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            
        # Early stopping check
        if early_stopping(val_loss):
            if verbose >= 2:
                print(f"Early stopping triggered at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    
    # Calculate validation accuracy
    model.eval()
    val_accs = {}
    with torch.no_grad():
        outputs = model(X_val_tensor)
        for col, logits in outputs.items():
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_accs[col] = accuracy_score(y_val_tensors[col].cpu().numpy(), preds)
    
    avg_val_acc = sum(val_accs.values()) / len(val_accs)
    
    return model, avg_val_acc, training_time


def optimize_neural_network(X_train, y_train_dict, output_info, discrete_labels=None, 
                           max_values=None, delta_scaling_factors=None, search_type='random',
                           n_iter=50, cv_folds=3, verbose=1):
    """
    Perform hyperparameter optimization for the MultiHeadClassifier neural network
    with fewer parameter combinations for efficiency.
    """
    from sklearn.model_selection import KFold
    from episodicMemNet_train import MultiHeadClassifier
    
    # Define a more comprehensive parameter grid
    if search_type == 'grid':
        param_grid = {
            'lr': [0.01, 0.001, 0.0001],
            'batch_size': [32, 64, 128],
            'hidden_size': [64, 128, 256],
            'optimizer_type': ['adam'],
            'max_epochs': [150, 300],
        }
    else:  # random search
        # Define ranges for random search without using rv_discrete objects
        param_grid = {
            'lr': uniform(0.0001, 0.1),
            'batch_size': [32, 64, 96, 128, 192, 256],  # More batch size options
            'hidden_size': [32, 64, 96, 128, 192, 256, 384, 512],  # More hidden size options
            'optimizer_type': ['adam', 'sgd'],
            'max_epochs': [100, 150, 200, 250, 300],  # More epochs options
        }
    
    # Setup cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Track best score and parameters
    best_score = -float('inf')
    best_params = None
    best_model = None
    
    # Determine number of iterations - REMOVE the limit of 10
    if search_type == 'random':
        n_iterations = n_iter  # Use the full requested number of iterations
    else:
        n_iterations = len(param_grid['lr']) * len(param_grid['batch_size']) * \
                     len(param_grid['hidden_size']) * len(param_grid['optimizer_type']) * \
                     len(param_grid['max_epochs'])
    
    if verbose >= 1:
        print(f"Starting {search_type} search with {n_iterations} parameter combinations")
    
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose >= 1:
        print(f"Using device: {device}")
    
    # Generate parameter combinations
    param_combinations = []
    if search_type == 'random':
        for _ in range(n_iterations):
            params = {
                'lr': float(uniform.rvs(loc=param_grid['lr'].args[0], 
                                       scale=param_grid['lr'].args[1])),
                # Convert numpy int to Python int
                'batch_size': int(np.random.choice(param_grid['batch_size'])),
                'hidden_size': int(np.random.choice(param_grid['hidden_size'])),
                'optimizer_type': str(np.random.choice(param_grid['optimizer_type'])),
                'max_epochs': int(np.random.choice(param_grid['max_epochs']))
            }
            if params['optimizer_type'] == 'sgd':
                params['momentum'] = float(uniform.rvs(0.7, 0.3))
            param_combinations.append(params)
    else:  # grid search
        from itertools import product
        param_keys = list(param_grid.keys())
        for values in product(*(param_grid[key] for key in param_keys)):
            params = dict(zip(param_keys, values))
            # Ensure batch_size is a Python int, not numpy int
            if 'batch_size' in params:
                params['batch_size'] = int(params['batch_size'])
            if 'hidden_size' in params:
                params['hidden_size'] = int(params['hidden_size'])
            if 'max_epochs' in params:
                params['max_epochs'] = int(params['max_epochs'])
            param_combinations.append(params)
    
    # Perform search
    for i, params in enumerate(param_combinations):
        if verbose >= 1:
            print(f"\nTrying parameters {i+1}/{len(param_combinations)}: {params}")
        
        cv_scores = []
        
        # Cross-validation loop
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold = {col: vals[train_idx] for col, vals in y_train_dict.items()}
            y_val_fold = {col: vals[val_idx] for col, vals in y_train_dict.items()}
            
            # Initialize model with current parameters
            hidden_size = params['hidden_size']
            model = MultiHeadClassifier(
                n_inputs=X_train.shape[1],
                n_hidden=hidden_size,
                output_info=output_info,
                discrete_labels=discrete_labels,
                max_values=max_values,
                delta_scaling_factors=delta_scaling_factors
            ).to(device)
            
            # Train and evaluate model
            try:
                _, val_acc, _ = train_evaluate_nn(
                    model, X_train_fold, y_train_fold, X_val_fold, y_val_fold, params, device
                )
                cv_scores.append(val_acc)
                if verbose >= 2:
                    print(f"Fold validation accuracy: {val_acc:.4f}")
            except Exception as e:
                if verbose >= 1:
                    print(f"Error with parameters {params}: {e}")
                cv_scores.append(0.0)
        
        # Calculate mean CV score
        mean_cv_score = sum(cv_scores) / len(cv_scores) if cv_scores else 0.0
        
        if verbose >= 1:
            print(f"Mean validation accuracy: {mean_cv_score:.4f}")
        
        # Update best score if better
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_params = params
            
            # Train a model on the full training set with best params
            if verbose >= 1:
                print(f"New best parameters found! Training model on full training set...")
            
            model = MultiHeadClassifier(
                n_inputs=X_train.shape[1],
                n_hidden=best_params['hidden_size'],
                output_info=output_info,
                discrete_labels=discrete_labels,
                max_values=max_values,
                delta_scaling_factors=delta_scaling_factors
            ).to(device)
            
            best_model, _, _ = train_evaluate_nn(
                model, X_train, y_train_dict, X_train, y_train_dict, best_params, device
            )
            
            if verbose >= 1:
                print(f"Best validation accuracy so far: {best_score:.4f}")
    
    if verbose >= 1:
        print(f"\nBest parameters: {best_params}")
        print(f"Best validation accuracy: {best_score:.4f}")
    
    # Move best model back to CPU for easier handling
    if best_model is not None:
        best_model = best_model.cpu()
    
    return best_params, best_model, best_score


def optimize_xgboost(X_train, y_train, X_val=None, y_val=None, search_type='random', 
                    n_iter=50, cv=3, verbose=1):
    """
    Perform hyperparameter optimization for XGBoost with fewer parameter combinations.
    """
    # Define a more focused parameter grid
    if search_type == 'grid':
        param_grid = {
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    else:  # random search
        param_grid = {
            'max_depth': [3, 4, 5, 6, 8, 10],
            'learning_rate': uniform(0.01, 0.19),
            'n_estimators': [50, 100, 150, 200, 250],
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.7, 0.3)
        }
    
    # Create XGBoost classifier - removed deprecated use_label_encoder parameter
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss'
    )
    
    # Setup search - REMOVE the limit of 10
    # n_iter = min(n_iter, 10)  # Removed limitation
    
    start_time = time.time()
    if search_type == 'random':
        search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_grid,
            n_iter=n_iter,  # Use the full requested number of iterations
            scoring='accuracy',
            cv=cv,
            verbose=verbose,
            random_state=42,
            n_jobs=-1
        )
    else:
        search = GridSearchCV(
            xgb_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            verbose=verbose,
            n_jobs=-1
        )
    
    # Perform search
    search.fit(X_train, y_train)
    
    # Get best parameters and model
    best_params = search.best_params_
    best_model = search.best_estimator_
    best_score = search.best_score_
    
    search_time = time.time() - start_time
    
    if verbose >= 1:
        print(f"\nBest XGBoost parameters: {best_params}")
        print(f"Best validation accuracy: {best_score:.4f}")
        print(f"Search completed in {search_time:.2f} seconds")
    
    return best_params, best_model, best_score


def optimize_multiple_xgboost(X_train, y_train_dict, X_val=None, y_val_dict=None, 
                            search_type='random', n_iter=10, cv=3, verbose=1):
    """
    Perform hyperparameter optimization for multiple XGBoost models (one per output)
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train_dict : dict
        Dictionary mapping column names to training labels
    X_val : array-like, optional
        Validation features
    y_val_dict : dict, optional
        Dictionary mapping column names to validation labels
    search_type : str, default='random'
        Type of search, 'random' or 'grid'
    n_iter : int, default=10
        Number of iterations for random search
    cv : int, default=3
        Number of cross-validation folds
    verbose : int, default=1
        Verbosity level
        
    Returns:
    --------
    dict : Dictionary mapping column names to tuples of (best_params, best_model, best_score)
    """
    results = {}
    
    for col_name, y_train in y_train_dict.items():
        if verbose >= 1:
            print(f"\nOptimizing XGBoost for {col_name}")
        
        y_val = y_val_dict.get(col_name) if y_val_dict else None
        
        best_params, best_model, best_score = optimize_xgboost(
            X_train, y_train, X_val, y_val, search_type, n_iter, cv, verbose
        )
        
        results[col_name] = (best_params, best_model, best_score)
    
    return results


def optimize_random_forest(X_train, y_train, X_val=None, y_val=None, search_type='random',
                          n_iter=10, cv=3, verbose=1):
    """
    Perform hyperparameter optimization for Random Forest.
    
    Parameters:
    -----------
    X_train, y_train, X_val, y_val : Data for training and validation
    search_type : str, default='random'
        'random' or 'grid'
    n_iter : int, default=10
        Number of parameter combinations
    cv : int, default=3
        Number of cross-validation folds
    verbose : int, default=1
        Verbosity level
        
    Returns:
    --------
    tuple : (best_params, best_model, best_score)
    """
    # Define parameter grid
    if search_type == 'grid':
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    else:  # random search
        param_grid = {
            'n_estimators': [50, 75, 100, 125, 150],
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 3, 5, 7, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        }
    
    # Create model
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Setup search - REMOVE the limit of 10
    # n_iter = min(n_iter, 10)  # Removed limitation
    
    start_time = time.time()
    if search_type == 'random':
        search = RandomizedSearchCV(
            rf_model,
            param_distributions=param_grid,
            n_iter=n_iter,  # Use the full requested number of iterations
            scoring='accuracy',
            cv=cv,
            verbose=verbose,
            random_state=42,
            n_jobs=-1
        )
    else:
        search = GridSearchCV(
            rf_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            verbose=verbose,
            n_jobs=-1
        )
    
    # Perform search
    search.fit(X_train, y_train)
    
    # Get best parameters and model
    best_params = search.best_params_
    best_model = search.best_estimator_
    best_score = search.best_score_
    
    search_time = time.time() - start_time
    
    if verbose >= 1:
        print(f"\nBest Random Forest parameters: {best_params}")
        print(f"Best validation accuracy: {best_score:.4f}")
        print(f"Search completed in {search_time:.2f} seconds")
    
    return best_params, best_model, best_score


def optimize_decision_tree(X_train, y_train, X_val=None, y_val=None, search_type='random',
                          n_iter=10, cv=3, verbose=1):
    """
    Perform hyperparameter optimization for Decision Tree.
    """
    # Define parameter grid
    if search_type == 'grid':
        param_grid = {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'criterion': ['gini', 'entropy']
        }
    else:  # random search
        param_grid = {
            'max_depth': [None, 3, 5, 8, 10, 15],
            'min_samples_split': [2, 3, 5, 8, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'criterion': ['gini', 'entropy']
        }
    
    # Create model
    dt_model = DecisionTreeClassifier(random_state=42)
    
    # Setup search - REMOVE the limit of 10 
    # n_iter = min(n_iter, 10)  # Removed limitation
    
    start_time = time.time()
    if search_type == 'random':
        search = RandomizedSearchCV(
            dt_model,
            param_distributions=param_grid,
            n_iter=n_iter,  # Use the full requested number of iterations
            scoring='accuracy',
            cv=cv,
            verbose=verbose,
            random_state=42,
            n_jobs=-1
        )
    else:
        search = GridSearchCV(
            dt_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            verbose=verbose,
            n_jobs=-1
        )
    
    # Perform search
    search.fit(X_train, y_train)
    
    # Get best parameters and model
    best_params = search.best_params_
    best_model = search.best_estimator_
    best_score = search.best_score_
    
    search_time = time.time() - start_time
    
    if verbose >= 1:
        print(f"\nBest Decision Tree parameters: {best_params}")
        print(f"Best validation accuracy: {best_score:.4f}")
        print(f"Search completed in {search_time:.2f} seconds")
    
    return best_params, best_model, best_score


def optimize_knn(X_train, y_train, X_val=None, y_val=None, search_type='random',
               n_iter=10, cv=3, verbose=1):
    """
    Perform hyperparameter optimization for k-NN.
    """
    # Define parameter grid
    if search_type == 'grid':
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
        }
    else:  # random search
        param_grid = {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 19],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    
    # Create model
    knn_model = KNeighborsClassifier(n_jobs=-1)
    
    # Setup search - REMOVE the limit of 10
    # n_iter = min(n_iter, 10)  # Removed limitation
    
    start_time = time.time()
    if search_type == 'random':
        search = RandomizedSearchCV(
            knn_model,
            param_distributions=param_grid,
            n_iter=n_iter,  # Use the full requested number of iterations
            scoring='accuracy',
            cv=cv,
            verbose=verbose,
            random_state=42,
            n_jobs=-1
        )
    else:
        search = GridSearchCV(
            knn_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            verbose=verbose,
            n_jobs=-1
        )
    
    # Perform search
    search.fit(X_train, y_train)
    
    # Get best parameters and model
    best_params = search.best_params_
    best_model = search.best_estimator_
    best_score = search.best_score_
    
    search_time = time.time() - start_time
    
    if verbose >= 1:
        print(f"\nBest k-NN parameters: {best_params}")
        print(f"Best validation accuracy: {best_score:.4f}")
        print(f"Search completed in {search_time:.2f} seconds")
    
    return best_params, best_model, best_score


def optimize_multiple_models(X_train, y_train_dict, X_val=None, y_val_dict=None,
                           search_type='random', n_iter=10, cv=3, verbose=1):
    """
    Optimize hyperparameters for multiple models and outputs.
    
    Returns:
    --------
    dict : Dictionary with structure {model_type: {output_name: (best_params, best_model, best_score)}}
    """
    results = {
        'random_forest': {},
        'decision_tree': {},
        'knn': {},
        'xgboost': {}
    }
    
    optimize_functions = {
        'random_forest': optimize_random_forest,
        'decision_tree': optimize_decision_tree,
        'knn': optimize_knn,
        'xgboost': optimize_xgboost
    }
    
    if verbose >= 1:
        print("\n=== Starting optimization for multiple models ===")
        
    for model_type, optimize_func in optimize_functions.items():
        if verbose >= 1:
            print(f"\n--- Optimizing {model_type} ---")
            
        for col_name, y_train in y_train_dict.items():
            if verbose >= 1:
                print(f"\nOutput: {col_name}")
                
            y_val = y_val_dict.get(col_name) if y_val_dict else None
            
            try:
                best_params, best_model, best_score = optimize_func(
                    X_train, y_train, X_val, y_val,
                    search_type=search_type, n_iter=n_iter, cv=cv, verbose=max(0, verbose-1)
                )
                results[model_type][col_name] = (best_params, best_model, best_score)
                
                if verbose >= 1:
                    print(f"Best {model_type} score for {col_name}: {best_score:.4f}")
            except Exception as e:
                print(f"Error optimizing {model_type} for {col_name}: {e}")
    
    return results
