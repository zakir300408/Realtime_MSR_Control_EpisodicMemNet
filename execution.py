import torch
from episodicMemNet_train import MultiHeadClassifier, predict_with_memory
from xbox import get_movement
import time
import pygame
import numpy as np
from Markov_Env_for_execution import RobotEnv  # Add this import

__all__ = ['input']

# Explicitly set weights_only to False to avoid the future deprecation warning.
checkpoint = torch.load("checkpoint.pth", map_location=torch.device("cpu"), weights_only=False)
memory = checkpoint["memory"]
discrete_labels = checkpoint["discrete_labels"]
# Remove max_values loading
delta_scaling_factors = checkpoint.get("delta_scaling_factors", {})  # Add this line
output_info = { key: len(torch.unique(torch.tensor(memory[key]))) for key in memory if key != "full_input"}

# Initialize RobotEnv
env = RobotEnv()
initial_state, _ = env.reset()  # Get initial state

_model = None
def get_model():
    global _model
    if _model is None:
        _model = MultiHeadClassifier(n_inputs=10, n_hidden=64, output_info=output_info,
                                    discrete_labels=discrete_labels,
                                    delta_scaling_factors=delta_scaling_factors)  # Removed max_values
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()
    return _model

def recurrsive_input_call(normalized_angle, movement, signal_state):
    """Modified to use the current state from environment"""
    rep = get_representative_deltas(movement)
    outputs = []
    model = get_model()
    print("\nModel inputs for each delta variant:")
    for delta_x, delta_y, delta_angle in zip(rep["delta_x"], rep["delta_y"], rep["delta_angle"]):
        # Create unscaled features first
        features = [normalized_angle, delta_x, delta_y, delta_angle] + signal_state
        print(f"Original (unscaled) input: [angle={normalized_angle:.3f}, dx={delta_x:.3f}, "
              f"dy={delta_y:.3f}, dθ={delta_angle:.3f}, signal_state={[f'{x:.3f}' for x in signal_state]}]")
        
        features_tensor = torch.tensor([features], dtype=torch.float32)
        scaled_tensor = model.scale_deltas(features_tensor)
        
        scaled_features = scaled_tensor.numpy()[0]
        print(f"Scaled input: [angle={scaled_features[0]:.3f}, dx={scaled_features[1]:.3f}, "
              f"dy={scaled_features[2]:.3f}, dθ={scaled_features[3]:.3f}, signal_state={[f'{x:.3f}' for x in scaled_features[4:]]}]")
        
        outputs.append(scaled_tensor)
    return outputs

def get_representative_deltas(category):
    """
    Return representative (10%, 50%, 90%) quantile values for delta_x, delta_y, and delta_angle
    based on the given movement category.
    Values are now kept in their original (unscaled) form.
    """
    mapping = {
        "Backward High": {
            "delta_x": (-0.015, 0.001, 0.016),
            "delta_y": (-0.010, 0.004, 0.013),
            "delta_angle": (-0.006, 0.000, 0.006)
        },
        "Backward Medium": {
            "delta_x": (-0.006, 0.001, 0.007),
            "delta_y": (-0.005, 0.001, 0.006),
            "delta_angle": (-0.006, 0.000, 0.006)
        },
        "Backward with Anticlockwise Rotation": {
            "delta_x": (-0.015, 0.000, 0.015),
            "delta_y": (-0.010, 0.002, 0.011),
            "delta_angle": (0.011, 0.019, 0.042)
        },
        "Backward with Clockwise Rotation": {
            "delta_x": (-0.014, 0.002, 0.015),
            "delta_y": (-0.009, 0.002, 0.011),
            "delta_angle": (-0.044, -0.019, -0.011)
        },
        "Forward High": {
            "delta_x": (-0.018, -0.007, 0.016),
            "delta_y": (-0.013, -0.003, 0.011),
            "delta_angle": (-0.006, 0.000, 0.006)
        },
        "Forward Medium": {
            "delta_x": (-0.007, -0.002, 0.006),
            "delta_y": (-0.006, -0.001, 0.005),
            "delta_angle": (-0.006, 0.000, 0.006)
        },
        "Forward with Anticlockwise Rotation": {
            "delta_x": (-0.019, -0.003, 0.015),
            "delta_y": (-0.012, -0.001, 0.011),
            "delta_angle": (0.011, 0.017, 0.035)
        },
        "Forward with Clockwise Rotation": {
            "delta_x": (-0.016, -0.001, 0.015),
            "delta_y": (-0.013, -0.003, 0.009),
            "delta_angle": (-0.033, -0.017, -0.011)
        },
        "Rotation Left": {
            "delta_x": (-0.003, 0.000, 0.002),
            "delta_y": (-0.003, 0.000, 0.003),
            "delta_angle": (0.011, 0.017, 0.034)
        },
        "Rotation Right": {
            "delta_x": (-0.002, 0.000, 0.003),
            "delta_y": (-0.004, 0.000, 0.003),
            "delta_angle": (-0.033, -0.017, -0.011)
        }
    }
    if category not in mapping:
        raise ValueError(f"Unknown movement category: {category}")
    return mapping[category]

def _value_to_action(param_str, value):
    # Simplified private function for parameter conversion.
    parts = param_str.split('_')
    if len(parts) != 4:
        raise ValueError(f"Invalid parameter string format: {param_str}")
    
    param_type = parts[1]
    channel = parts[-1].upper()
    channels = ['X', 'Y', 'Z']
    
    if channel not in channels:
        raise ValueError(f"Invalid channel: {channel}")
    
    channel_index = channels.index(channel)
    
    # Convert normalized value to actual value
    if param_type == "phase":
        actual_value = int(round(value * 315))  # Convert normalized value back to degrees
        phase_values = [0, 90, 180, 315]
        # Find closest valid phase value
        actual_value = min(phase_values, key=lambda x: abs(x - actual_value))
        if actual_value not in phase_values:
            raise ValueError(f"Invalid phase value after conversion: {actual_value}")
        return channel_index * 4 + phase_values.index(actual_value)
    
    elif param_type == "amplitude":
        actual_value = int(round(value * 9))  # Convert normalized value back to amplitude
        amplitude_values = [0, 5, 7, 9]
        # Find closest valid amplitude value
        actual_value = min(amplitude_values, key=lambda x: abs(x - actual_value))
        if actual_value not in amplitude_values:
            raise ValueError(f"Invalid amplitude value after conversion: {actual_value}")
        return 12 + channel_index * 4 + amplitude_values.index(actual_value)
    
    else:
        raise ValueError(f"Invalid parameter type: {param_type}")

def _predict_change(sample):
    # Remove the shape check and bias term addition as it's no longer needed
    model = get_model()
    # Since sample is already scaled from recurrsive_input_call, we set scaled_input=True
    outputs = predict_with_memory(model, sample, memory, discrete_labels)
    
    if outputs.get("memory_miss", False):
        print("No memory match, using network prediction")
        outputs = model.predict(sample, scaled_input=False)
    
    prev_map = {
        "next_phase_value_x": 4,
        "next_phase_value_y": 5,
        "next_phase_value_z": 6,
        "next_amplitude_value_x": 7,
        "next_amplitude_value_y": 8,
        "next_amplitude_value_z": 9
    }

    valid_phase_values = {0, 90, 180, 315}
    valid_amplitude_values = {0, 5, 7, 9}
    
    input_vals = sample.cpu().numpy()[0]
    tolerance = 1e-6
    changes = []
    
    for col, (logits, _) in outputs.items():
        if col == "memory_miss":
            continue
            
        predicted, conf_dict = model.get_discrete_prediction(col=col, logits=logits[0])
        prev_value = input_vals[prev_map[col]]
        
        # Validate and adjust predicted values silently
        if 'phase' in col:
            predicted_phase = min(valid_phase_values, key=lambda x: abs(x - (predicted * 315)))
            predicted = predicted_phase / 315.0
        elif 'amplitude' in col:
            predicted_amp = min(valid_amplitude_values, key=lambda x: abs(x - (predicted * 9)))
            predicted = predicted_amp / 9.0
        
        if abs(predicted - prev_value) > tolerance:
            prob = conf_dict.get(predicted, 0)
            changes.append((col, predicted, prob))
    
    if changes:
        return max(changes, key=lambda x: x[2])
    return None

if __name__ == "__main__":
    pygame.init()
    pygame.joystick.init()

    current_state = initial_state  # Store the initial state
    last_action = None  # Store the last action
    print("Starting control loop...")
    try:
        while True:
            movement = get_movement()
            if movement:
                print(f"\nReceived movement: {movement}")
                
                # Extract normalized_angle and signal_state from current_state
                normalized_angle = current_state[0]  # First element is normalized_angle
                signal_state = current_state[1:].tolist()  # Rest are phase and amplitude values
                
                # Generate input variants based on movement using current state
                input_variants = recurrsive_input_call(normalized_angle, movement, signal_state)
                
                # Process each variant and collect changes
                best_change = None
                best_confidence = -1
                
                for input_tensor in input_variants:
                    change = _predict_change(input_tensor)
                    if change and change[2] > best_confidence:
                        best_change = change
                        best_confidence = change[2]
                
                # Apply the best change if found, otherwise use last action
                if best_change:
                    param, value, confidence = best_change
                    print(f"Predicted change: {param} = {value:.3f} (confidence: {confidence:.3f})")
                    
                    # Convert to action and apply to environment
                    action = _value_to_action(param, value)
                    last_action = action  # Store the new action
                else:
                    action = last_action  # Use the last action if no new action is predicted
                    if action is not None:
                        print(f"No new action predicted, reusing last action: {action}")
                    else:
                        print("No previous action available, using default action 0")
                        last_action = action

                print(f"Action: {action}")
                
                # Get new state from environment
                current_state, reward, done, truncated, info = env.step(action)
                print(f"Reward: {reward:.3f}")
                
                if done or truncated:
                    print(f"Episode ended: {info['termination_reason']}")
                    current_state, _ = env.reset()
                    last_action = None  # Reset last_action on episode end
                
                print(f"Current state: angle={normalized_angle:.3f}, signal_state={[f'{x:.3f}' for x in signal_state]}")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        env.close()
        pygame.quit()
