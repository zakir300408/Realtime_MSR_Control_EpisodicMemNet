import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Markov_Env import RobotEnv  # Now it will find the module
from Mlp_ensemble_model.model import FFNModel  # Updated import

# Define the causing action values
causing_action_values = np.linspace(0, 1, 8)

# Global list to accumulate positions
all_visited_positions = []
all_considered_positions = []
all_best_positions = []  # New global list to accumulate best positions

import numpy as np
import torch

# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y1 - y2)**2)

def denormalize_action(causing_action):
    """
    Convert the normalized action (0 to 1) into discrete channel/parameter changes.
    """
    causing_action = int(round(causing_action * 7))
    if causing_action < 4:  # phase adjustments
        channel_index = causing_action // 2
        phase_value = [0, 180][causing_action % 2]
        param_type = 'phase'
        channel = ['X', 'Y'][channel_index]
        # Normalize
        value = phase_value / 180.0
    else:  # amplitude adjustments
        channel_index = (causing_action - 4) // 2
        amplitude_value = [0, 8][(causing_action - 4) % 2]
        param_type = 'amplitude'
        channel = ['X', 'Y'][channel_index]
        # Normalize
        value = amplitude_value / 8.0

    return param_type, channel, value

def calculate_cost_with_orientation(x, y, orientation, target_x, target_y):
    """
    Calculate a cost using the same relative angle calculation as Markov_Env.
    """
    distance = euclidean_distance(x, y, target_x, target_y)
    
    # We assume RobotEnv.calculate_normalized_relative_angle exists
    # orientation is normalized (0..1), so multiply by 360 to get degrees
    target_vector = RobotEnv.calculate_normalized_relative_angle(
        None,  # self is not needed if it's a @staticmethod
        [x, y],  
        orientation * 360.0,  
        [target_x, target_y]
    )

    if target_vector is None:
        return float('inf')

    # Example cost: angle_cost = min distance to 0.0 or 0.5
    angle_cost = min(abs(target_vector - 0.0), abs(target_vector - 0.5))

    # Combine distance and orientation costs
    return distance

# --------------------------------------------------------------------
# Model inference
# --------------------------------------------------------------------

def predict_with_model(features):
    """
    Evaluate the ensemble of models and return averaged predictions.
    Loads all models once, stores them on the predict_with_model function object.
    """
    if not hasattr(predict_with_model, "models"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize list to store ensemble models
        models = []
        
        # Load each model in the ensemble
        for seed in [101, 102, 103, 104, 105]:  # Same seeds as in training
            model = FFNModel(
                input_dim=8,
                hidden_dim=512,
                output_dim=3,
                num_blocks=4
            ).to(device)
            
            # Load the state dict for this model
            model_path = f'results/best_model_{seed}.pth'
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
        
        predict_with_model.models = models
        predict_with_model.device = device

    # Process input features
    features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    features[:, 4:7] = np.trunc(features[:, 4:7] * 100) / 100.0
    device = predict_with_model.device
    features_tensor = torch.FloatTensor(features).to(device)
    
    # Get predictions from all models
    with torch.no_grad():
        all_predictions = []
        for model in predict_with_model.models:
            pred = model(features_tensor).cpu().numpy()
            all_predictions.append(pred)
        
        # Average the predictions
        ensemble_prediction = sum(all_predictions) / len(all_predictions)
        ensemble_prediction = np.trunc(ensemble_prediction * 100) / 100.0
    
    return ensemble_prediction

def predict_all_actions_data(initial_features, causing_action_values):
    """
    Returns a dictionary:
        causing_action -> {
            'position': (x_position, y_position),
            'angle': angle,
            'distance_change': distance,
            'next_features': next_initial_feats
        }
    """
    num_actions = len(causing_action_values)
    # Tile the initial features to build a batch
    all_features = np.tile(initial_features, (num_actions, 1))
    # Overwrite the last column with each action
    all_features[:, 7] = causing_action_values

    predictions_array = predict_with_model(all_features)
    all_predictions = {}

    for idx, causing_action in enumerate(causing_action_values):
        x_position, y_position, angle = predictions_array[idx]  # Predictions are already truncated

        # Get current x,y from the original features
        current_x = all_features[idx][5]
        current_y = all_features[idx][6]
        distance = euclidean_distance(current_x, current_y, x_position, y_position)

        # Build next features similarly to how 'predict_next_state' did
        components = initial_features[:4]  # [phaseX, phaseY, amplitudeX, amplitudeY]
        param_type, channel, value = denormalize_action(causing_action)
        index_map = {'X': 0, 'Y': 1}
        if param_type == 'phase':
            components[index_map[channel]] = value
        else:
            components[index_map[channel] + 2] = value

        # Append angle, x, y, action
        next_initial_feats = components + [angle, x_position, y_position, causing_action]

        all_predictions[causing_action] = {
            'position': (x_position, y_position),
            'angle': angle,
            'distance_change': distance,
            'next_features': next_initial_feats
        }

    return all_predictions

# --------------------------------------------------------------------
# Depth-limited search ("A*"-like) function
# --------------------------------------------------------------------

def a_star_search_segmental(
    initial_features, 
    causing_action_values, 
    target_x, 
    target_y, 
    max_levels=4
):
    """
    Depth-limited search enumerating all paths up to 'max_levels'. 
    Uses a cumulative cost (sum of costs at each node).
    """

    def get_available_actions(level):
        if level == 0:  # First level
            return [causing_action_values[5], causing_action_values[7]]
        else:  # Subsequent levels
            return causing_action_values

    def predict_all_levels(current_features, level=0, cumulative_cost=0.0):
        # Base case: if we've reached the maximum depth, return what we have
        if level >= max_levels:
            # No more children; return a "leaf" with empty path and positions
            return [(current_features, [], [], cumulative_cost)]

        # Filter actions based on current level
        available_actions = get_available_actions(level)
        
        # Get predictions for all available actions from the current node
        all_predictions = predict_all_actions_data(current_features, available_actions)

        all_paths = []
        for causing_action in available_actions:
            pred = all_predictions[causing_action]
            next_features = pred['next_features']

            # The next_features last 4 are [angle, x, y, causing_action]
            angle, x, y, _ = next_features[-4:]

            # Evaluate the local cost at this node
            # You may want something else for "local cost" if you have a different metric
            cost_to_target = calculate_cost_with_orientation(x, y, angle, target_x, target_y)
            
            # Accumulate the cost
            new_cumulative_cost = cumulative_cost + cost_to_target

            # Recurse to the next level
            children = predict_all_levels(next_features, level + 1, new_cumulative_cost)
            for (final_features, path, positions, child_cost) in children:
                # Add our current step at the front
                position = pred['position']  # (x, y) from the prediction
                all_paths.append((
                    final_features,
                    [causing_action] + path,
                    [position] + positions,
                    child_cost  # child_cost is already the cumulative cost
                ))

        return all_paths

    def find_best_path(all_paths):
        # Initialize with "infinite" cost
        best_cumulative_cost = float('inf')
        best_final_features = None
        best_path = None
        best_positions = None

        for (final_features, path, positions, path_cost) in all_paths:
            # Now we compare the *accumulated* path cost, not just the final node's cost
            if path_cost < best_cumulative_cost:
                best_cumulative_cost = path_cost
                best_final_features = final_features
                best_path = path
                best_positions = positions

        return best_path, best_positions, best_cumulative_cost, best_final_features

    # 1) Expand all possible paths (up to depth `max_levels`) from the initial features
    all_paths = predict_all_levels(initial_features)

    # 2) Find the best path based on the *lowest cumulative cost*
    best_path, best_positions, best_cost, best_final_features = find_best_path(all_paths)

    # 3) (Optional) collect/append visited & considered positions
    visited_positions = [(initial_features[5], initial_features[6])]  # start pos
    considered_positions = []
    for (final_features, path, positions, path_cost) in all_paths:
        considered_positions.extend(positions)
        if path == best_path:
            visited_positions.extend(positions)

    all_visited_positions.extend(visited_positions)
    all_considered_positions.extend(considered_positions)
    all_best_positions.append(best_positions)

    return best_path, best_positions, [best_cost], best_final_features, [
        denormalize_action(a) for a in best_path
    ]


def generate_circle_targets(current_x, current_y):
    targets = []
    for i in range(3):
        angle = i * (2 * np.pi) / 3
        target_x = current_x + 0.3 * np.cos(angle)
        target_y = current_y + 0.3 * np.sin(angle)
        targets.append((target_x, target_y))
    return targets
# Global variable to track if legends have been added
legends_added = False

def plot_results(visited_positions, considered_positions, best_positions_list, targets, threshold, initial_features):
    global legends_added  # Use the global variable

    if not visited_positions or not considered_positions:
        print("No positions visited.")
        return

    visited_x, visited_y = zip(*visited_positions)
    considered_x, considered_y = zip(*considered_positions)

    # Save the data for later use
    with open(f'plot_data_{time.time()}.pkl', 'wb') as f:
        pickle.dump(
            (visited_positions, considered_positions, best_positions_list, targets, threshold, initial_features), f)

    # Plot the positions
    plt.figure(figsize=(10, 10))
    plt.scatter(considered_x, considered_y, alpha=0.6, color='gray',
                label='Considered Positions' if not legends_added else "")  # Lighter plot for considered positions
    plt.scatter(visited_x, visited_y, alpha=1, color='blue',
                label='Visited Path' if not legends_added else "")  # Increased alpha for visited positions

    # Plot the best positions for each target
    for best_positions in best_positions_list:
        best_x, best_y = zip(*best_positions)  # Unpack best positions
        plt.scatter(best_x, best_y, alpha=1, color='green',
                    label='Best Path' if not legends_added else "")  # New scatter plot for best positions

    for target in targets:
        plt.plot(target[0], target[1], marker='*', color='red', markersize=15,
                 label='Target Position' if not legends_added else "")

    plt.plot(initial_features[5], initial_features[6], marker='s', color='black', markersize=15,
             label='Starting Position' if not legends_added else "")

    # Draw a circle around each target position with the threshold as the radius
    for target in targets:
        circle = plt.Circle((target[0], target[1]), threshold, color='red', fill=False, linestyle='--',
                            label='Threshold Circle' if not legends_added else "")
        plt.gca().add_artist(circle)

    # Set plot limits
    all_x = considered_x + visited_x + tuple(target[0] for target in targets) + (initial_features[5],)
    all_y = considered_y + visited_y + tuple(target[1] for target in targets) + (initial_features[6],)
    buffer = 0.1 + threshold
    plt.xlim(min(all_x) - buffer, max(all_x) + buffer)
    plt.ylim(min(all_y) - buffer, max(all_y) + buffer)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Exploration and Path to Targets')
    plt.grid(True)

    if not legends_added:
        plt.legend()  # Add a legend to distinguish the plots
        legends_added = True  # Update the global variable to indicate legends have been added

    # Save image with current timestamp
    plt.savefig(f'path_{time.time()}.png', dpi=600)
    plt.show()
if __name__ == "__main__":
    initial_features = [0.0, 0.0, 0.0, 0.0, 0.89, 0.28, 0.5, 0.0]

    # Generate triangle targets
    start_time = time.time()
  
    circle_targets = [(0.2, 0.4)]

    for target in circle_targets:
        print(f"\nTarget to reach: {target}")
        reached_target = False
        current_initial_features = initial_features
        complete_path = []  # Track complete path
        complete_positions = []  # Track all positions
        complete_costs = []  # Track all costs
        complete_denorm_path = []  # Track complete denormalized path

        while not reached_target:
            # Print all possible actions and their distance changes
            print("\nPossible actions and their distance changes:")
            all_predictions = predict_all_actions_data(current_initial_features, causing_action_values)
            for action in causing_action_values:
                pred = all_predictions[action]
                denorm = denormalize_action(action)
                print(f"Action {action:.3f} ({denorm}) -> Distance change: {pred['distance_change']:.3f}")
            
            result_path, best_positions, costs, final_features, best_path_denorm = a_star_search_segmental(
                current_initial_features, causing_action_values, target[0], target[1])

            # Accumulate the results
            complete_path.extend(result_path)
            complete_positions.extend(best_positions)
            complete_costs.extend(costs)
            complete_denorm_path.extend(best_path_denorm)

            # Print distance changes for current segment
            print("\nDistance changes for current segment:")
            current_features = current_initial_features
            for i, action in enumerate(result_path):
                predictions = predict_all_actions_data(current_features, causing_action_values)
                pred = predictions[action]
                print(f"Step {i+1}: Action {action:.3f} ({denormalize_action(action)}) -> Distance change: {pred['distance_change']:.3f}")
                current_features = pred['next_features']

            # Check the final position
            final_x, final_y = final_features[-3], final_features[-2]
            distance_to_target = euclidean_distance(final_x, final_y, target[0], target[1])

            if distance_to_target <= 0.05:
                reached_target = True
            else:
                current_initial_features = final_features

        print("\nComplete path summary:")
        print("Full path:", complete_path)
        print("Number of total actions:", len(complete_path))
        print("Total accumulated cost:", sum(complete_costs))
        print("Complete sequence of positions:", complete_positions)
        print("Complete path (denormalized):", complete_denorm_path)
        print("\nFinal features:", final_features)

    print("Time it took was", time.time() - start_time)

    # Plot all paths together after processing all targets
    plot_results(all_visited_positions, all_considered_positions, all_best_positions, circle_targets, 0.05, initial_features)
