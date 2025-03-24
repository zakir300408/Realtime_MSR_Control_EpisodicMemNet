# Reinforcement Learning for Real-time Signal Control

## Project Overview

This project implements a reinforcement learning system for real-time signal control of robotic movements. It uses a multi-head neural network architecture to predict optimal phase and amplitude values across three axes (X, Y, Z) based on the current state and desired movement direction.

## Key Components

### Core Models

- `episodicMemNet_train.py`: Implements the multi-head classifier neural network with memory integration
- `model.py`: Contains the UNet architecture for image-based processing
- `execution.py`: Handles real-time control and execution using trained models
- `Markov_Env_for_training.py`/`Markov_Env_for_execution.py`: Environment models for training and execution

### Data Processing

- `extract_dataset_h5.py`: Extracts and processes data from H5 files
- `Generate_experimental_data.py`: Generates experimental data for training
- `generate_movement_classes.py`: Creates movement class definitions

### Model Evaluation

- `evaluation_models.py`: Provides functions for model evaluation and comparison
- `hyperparameter_optimization.py`: Implements hyperparameter optimization techniques
- `benchmark_model_inference.py`: Benchmarks model inference performance
- `plot_confusion_matrix.py`: Generates confusion matrices for model evaluation

### Visualization

- `visualization/create_video.py`: Creates visualizations of model performance and robot movements

## Usage

### Training a Model

```bash
python episodicMemNet_train.py
```

Add flags to control training behavior:
- `--optimize`: Use hyperparameter optimization
- `--optimize-all`: Optimize all supported model types
- `--search=random`: Select search strategy (random or grid)

### Model Execution

```bash
python execution.py
```

Use with optional flags:
- `--use-memory`: Enable memory-based prediction enhancement

### Model Evaluation

```bash
python benchmark_model_inference.py
```

### Visualization

```bash
python visualization/create_video.py
```

## Model Architecture

The core model is a multi-head neural network that learns optimal signal patterns for robotic movement control. It features:

- Shared backbone network for feature extraction
- Task-specific output heads for different signal parameters
- Memory integration for improved performance on previously seen patterns
- Support for constrained prediction to ensure physically valid outputs

## Data Structure

The model operates on input features including:
- Normalized angle
- Delta position values (x, y)
- Delta angle
- Previous signal states (phase and amplitude values)

Outputs include:
- Phase values for X, Y, Z axes
- Amplitude values for X, Y, Z axes

## Performance Evaluation

The system includes comprehensive evaluation tools that compare the neural network against traditional machine learning approaches including:
- XGBoost
- Random Forest
- Decision Trees
- k-Nearest Neighbors
- Multinomial Logistic Regression

Evaluation metrics include accuracy, confusion matrices, and statistical significance testing.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Pygame (for controller input)
- OpenCV (for visualization)
- Matplotlib
- H5py


