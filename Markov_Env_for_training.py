import gymnasium as gym  # Change import from gym to gymnasium

from utils.FY8300 import SignalGenerator
from camera.camera import Camera
import cv2
import threading
import numpy as np
import h5py
import os
from datetime import datetime
from utils.predict_image import ResultVisualizer
import time
import torch
from gymnasium.spaces import Box, Discrete  # Change from gym.spaces to gymnasium.spaces
import gymnasium

import math

# Constants
DEVICE = "cuda"
MODEL_PATH = "Multiclass_2_model_12_13_23.pth"
SUCCESS_REWARD = 2.0          # Reward for reaching target
BOUNDARY_PENALTY = -1.0       # Penalty for boundary violation
BOUNDARY_THRESHOLD = 40       # pixels from boundary
REPOSITION_INTERVAL = 10      # Episodes between manual repositioning
PRINT_FREQUENCY = 50          # Only print every 50 steps
ALIGNMENT_SCALE = 0.05        # Small scale factor for alignment reward
ALIGNMENT_THRESHOLD = 0.1     # Threshold for considering good alignment (in normalized units)


class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()

        # Setup Action and Observation Spaces
        self.setup_spaces()
        self.cumulative_reward = 0.0
        self.low_level_done=False
        self.target = None

        self.prev_centers = []  # List to keep track of past centers
        self.max_prev_centers = 5
        self.components_initialized = False
        self.prev_phase_adjustments = None
        self.step_count = 0
        self.target_reached = True
        self.episode_end_reason = None  # <--- new instance variable

        # Initialize Hardware and Software Components
        self.initialize_components()

        # Initialize Internal State Variables
        self.center_trajectory = []  # List to store center positions for trajectory
        self.initialize_state()
        self.current_episode = 0  # Initialize current_episode here
        self.current_step = (0)
        self.prev_angle = None  # Initialize previous angle
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sg_parameters = {
            'X': {'phase': 0, 'amplitude': 0},
            'Y': {'phase': 0, 'amplitude': 0},
            'Z': {'phase': 0, 'amplitude': 0},
        }

        # Create a unique folder for this experiment within RL_experiments_data based on the date and time
        self.folder_name = os.path.join("Results", self.current_time)
        os.makedirs(self.folder_name, exist_ok=True)

        # Create hdf5 path within the folder
        self.hdf5_file_path = os.path.join(self.folder_name, f"{self.current_time}_PPO_data.h5")

        self.set_physical_boundary()

        self.init_episode()  # Now you can call init_episode safely
        self.target_index = 0  # To keep track of the current target
        self.predefined_targets = self.generate_random_targets(20, 80)

    def setup_spaces(self):
        # 6 variables each with 2 possible values
        self.action_space = gymnasium.spaces.Discrete(24)  # Use gymnasium.spaces.Discrete
        
        # Define Observation Space using gymnasium
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(8,),  # <-- changed to 8
            dtype=np.float32
        )

    def initialize_components(self):
        if not self.components_initialized:
            # Initialize Signal Generator
            self.sg = SignalGenerator()
            self.sg.initialize_channel(1, 1, 2)
            self.sg.initialize_channel(2, 1, 2)
            self.sg.initialize_channel(3, 1, 2)
            self.sg.first_init()
            time.sleep(10)
            # Load the model first
            model = torch.load(MODEL_PATH)
            # Initialize ResultVisualizer and Camera
            self.result_visualizer = ResultVisualizer(model, DEVICE)
            self.cam = Camera()  # Create a Camera object
            self.cam_thread = threading.Thread(target=self.cam.capture_video)
            self.cam_thread.start()
            self.components_initialized = True



    def initialize_state(self):
        self.prev_center = None
        self.staying_counter = 0
        self.previous_distance_to_target = None
        self.boundary_coordinates = None

    def get_and_process_frame(self):
        frame = self.cam.get_latest_frame()
        (
            frame_resized,
            centers,
            angle,
            pred_bgr,
            orientation_labels,
        ) = self.result_visualizer.process_frame(
            frame,
            self.result_visualizer.device,
            self.result_visualizer.color_map,
        )
        return frame,  centers, angle, pred_bgr
    # User Input Methods
    def set_physical_boundary(self):
        frame = self.cam.get_latest_frame()

        # Calculate the center of the frame
        frame_height, frame_width = frame.shape[:2]
        self.circle_center = (frame_width // 2, frame_height // 2)
        self.circle_radius = 250

        # Draw the circular boundary
        self.frame_with_boundary = cv2.circle(
            frame.copy(),
            self.circle_center,
            self.circle_radius,
            (0, 255, 0),
            2,
        )

        # Calculate the max_distance based on the physical boundary
        self.max_distance = self.circle_radius * np.sqrt(2)

    def generate_random_targets(self, num_targets=20, min_distance_to_boundary=80):
        targets = []
        safe_radius = self.circle_radius - min_distance_to_boundary
        for _ in range(num_targets):
            angle = 2 * np.pi * np.random.rand()  # Random angle between 0 and 2*pi
            distance = safe_radius * np.sqrt(np.random.rand())  # Random distance within the safe radius
            raw_target_x = self.circle_center[0] + distance * np.cos(angle)
            raw_target_y = self.circle_center[1] + distance * np.sin(angle)
            normalized_target_x = float(f"{(raw_target_x - (self.circle_center[0] - self.circle_radius)) / (2 * self.circle_radius):.2f}")
            normalized_target_y = float(f"{(raw_target_y - (self.circle_center[1] - self.circle_radius)) / (2 * self.circle_radius):.2f}")
            targets.append((normalized_target_x, normalized_target_y))
        return targets

    def generate_new_target(self):
        target = self.predefined_targets[self.target_index]
        self.target_index = (self.target_index + 1) % len(self.predefined_targets)  # Loop through the targets
        return target

    # Environment Logic
    def is_inside_boundary(self, center):
        cx, cy = center  # No need to flip the coordinates
        distance_to_center = np.linalg.norm(np.array([cx, cy]) - np.array(self.circle_center))

        return distance_to_center <= self.circle_radius

    def distance_to_boundary(self, normalized_center):
        # Denormalize the center coordinates
        denormalized_cx = (normalized_center[0] * 2 * self.circle_radius) + (self.circle_center[0] - self.circle_radius)
        denormalized_cy = (normalized_center[1] * 2 * self.circle_radius) + (self.circle_center[1] - self.circle_radius)
        denormalized_center = [denormalized_cx, denormalized_cy]

        # Calculate distance to center in the original scale
        distance_to_center = np.linalg.norm(np.array(denormalized_center) - np.array(self.circle_center))
        distance_to_boundary = self.circle_radius - distance_to_center

        # Normalize the distance to boundary
        normalized_distance_to_boundary = distance_to_boundary / self.circle_radius
        return normalized_distance_to_boundary


    def calculate_reward(self, current_center, prev_distance_to_target):
       
 

        current_distance_to_target = np.linalg.norm(np.array(current_center) - np.array(self.target))
        
        # Success reward (reaching target)
        if current_distance_to_target <= (30 / (2 * self.circle_radius)):
            return SUCCESS_REWARD
        

        # Calculate distance change
        distance_moved = prev_distance_to_target - current_distance_to_target
        
        # Combined reward calculation
        if distance_moved > 0:  # Moving towards target
            # Exponential reward scaled by alignment
            base_reward = np.exp(distance_moved * 30.0) - 1.0
           
            distance_reward = base_reward
        else:  # Moving away from target
            distance_reward = distance_moved

        
        total_reward = distance_reward
        if self.current_step % PRINT_FREQUENCY == 0:
            print(f"\nStep {self.current_step}:")
            print(f"Distance Moved: {distance_moved:.4f}")
  
            print(f"Distance Reward: {distance_reward:.4f}")

            print(f"Total Reward: {total_reward:.4f}")
            print(f"Current Distance: {current_distance_to_target:.4f}")
        
        return total_reward

    def initialize_if_needed(self):
            if not self.components_initialized:
                self.initialize_components()

    def calculate_current_center(self, front_center, back_center):
        if front_center is not None and back_center is not None:
            raw_center_x = (front_center[0] + back_center[0]) / 2
            raw_center_y = (front_center[1] + back_center[1]) / 2

            # Normalize the coordinates and increase precision to 3 decimal places
            normalized_center_x = float(f"{(raw_center_x - (self.circle_center[0] - self.circle_radius)) / (2 * self.circle_radius):.3f}")
            normalized_center_y = float(f"{(raw_center_y - (self.circle_center[1] - self.circle_radius)) / (2 * self.circle_radius):.3f}")

            return [normalized_center_x, normalized_center_y]
        else:
            return None



    def calculate_normalized_relative_angle(self, normalized_robot_center, robot_orientation, normalized_target):
  
        if normalized_robot_center is None or normalized_target is None:
            return None

        # Calculate vector to target
        target_vector_x = normalized_target[0] - normalized_robot_center[0]
        target_vector_y = normalized_target[1] - normalized_robot_center[1]
        
        # Calculate absolute angle to target in degrees
        target_angle = math.degrees(math.atan2(target_vector_y, target_vector_x))
        
        # Normalize robot orientation to [-180, 180]
        robot_orientation = robot_orientation % 360
        if robot_orientation > 180:
            robot_orientation -= 360
            
        # Calculate relative angle
        relative_angle = target_angle - robot_orientation
        
        # Normalize to [-180, 180]
        relative_angle = ((relative_angle + 180) % 360) - 180
        
        # Convert to [0, 1] range and increase precision to 3 decimal places
        normalized_relative_angle = float(f"{(relative_angle + 180) / 360.0:.3f}")
        
        return normalized_relative_angle

    def reset(self, seed=None, options=None):
        if self.episode_end_reason is not None:
            print(
                f"Reset called, reason: {self.episode_end_reason}, "
                f"last episode cumulative reward: {self.cumulative_reward}"
            )
        else:
            print(f"Reset called with no prior episode end reason.")

        # Reset variables for the new episode
        self.episode_end_reason = None
        self.cumulative_reward = 0.0

        super().reset(seed=seed)  # Reset using gym's standard method

        print(f"Reset called at step {self.current_step}")
        info = {}  # Initialize info dictionary

        self.initialize_if_needed()
        self.current_step = 0
        self.current_episode += 1
        self.low_level_done = False
        self.cumulative_reward = 0.0
        self.center_trajectory = []

        # Always generate a new target for each episode
        self.target = self.generate_new_target()
        print(f"New target generated for episode {self.current_episode}: {self.target}")

        # Add manual repositioning check
        if self.current_episode % REPOSITION_INTERVAL == 0:
            user_input = input("10th episode reached. Please reposition robot and press 'Y' to continue: ")
            while user_input.lower() != 'y':
                user_input = input("Press 'Y' after repositioning the robot: ")

        # Reset signal generator parameters
        self.set_sg_parameters('X', 'amplitude', 0)
        self.set_sg_parameters('Y', 'amplitude', 0)
        self.set_sg_parameters('Z', 'amplitude', 0)
        self.set_sg_parameters('X', 'phase', 0)
        self.set_sg_parameters('Y', 'phase', 0)
        self.set_sg_parameters('Z', 'phase', 0)

        

        # Capture the initial phase and amplitude values
        prev_phase_values = [self.sg_parameters[ch]['phase'] / 315 for ch in ['X', 'Y', 'Z']]
        prev_amplitude_values = [self.sg_parameters[ch]['amplitude'] / 9 for ch in ['X', 'Y', 'Z']]
        while True:
            frame, centers, angle, pred_bgr = self.get_and_process_frame()
            front_center = centers.get(1, None)
            back_center = centers.get(2, None)
            current_center = self.calculate_current_center(front_center, back_center)

            # Check if current center is found and angle is not zero
            if current_center is None or angle is None:
                action_required = "Current center not found" if current_center is None else "Angle is zero"
                user_input = input(f"{action_required}. Please adjust the robot and press 'Y' to continue: ")
                while user_input.lower() != 'y':
                    user_input = input("Press 'Y' after repositioning the robot: ")
            else:
                break

        normalized_angle = angle / 360.0
        normalized_angle = float(f"{normalized_angle:.3f}")

        # Calculate the distance to the boundary
        if current_center is not None:

            distance_to_boundary = self.distance_to_boundary(current_center)
        else:
            distance_to_boundary = np.inf  # Use a large number or some other default


        # User intervention for repositioning the robot
        normalized_boundary_threshold = 40 / self.circle_radius  # Normalized threshold for user intervention
        if distance_to_boundary <= normalized_boundary_threshold:
            user_input = input("Robot too close to boundary. Reposition robot and press 'Y' to continue: ")
            while user_input.lower() != 'y':
                user_input = input("Press 'Y' after repositioning the robot: ")
                frame, centers, angle, pred_bgr = self.get_and_process_frame()
                front_center = centers.get(1, None)
                back_center = centers.get(2, None)
                current_center = self.calculate_current_center(front_center, back_center)

        # Set a new target if the low level is done or it's the start of a new episode
        if self.target_reached or self.current_episode == 1:
            self.target = self.generate_new_target()
            self.target_reached = False  # Reset the flag

        if current_center is not None and self.target is not None:
            raw_distance_to_target = float(f"{np.linalg.norm(np.array(current_center) - np.array(self.target)):.3f}")
        else:
            normalized_distance_to_target = np.inf  # Or some suitable default value

        self.previous_distance_to_target = raw_distance_to_target

        # After all repositioning checks, capture a fresh frame before building final flattened state
        frame, centers, angle, pred_bgr = self.get_and_process_frame()
        front_center = centers.get(1, None)
        back_center = centers.get(2, None)
        current_center = self.calculate_current_center(front_center, back_center)
        normalized_angle = angle / 360.0 if angle else 0.0
        normalized_angle = float(f"{normalized_angle:.3f}")
        if current_center is not None:
            raw_distance_to_target = float(f"{np.linalg.norm(np.array(current_center) - np.array(self.target)):.3f}")
        else:
            raw_distance_to_target = np.inf

        # Angle to target (considering normalization)
        angle = normalized_angle * 360  # Convert back to degrees

        target_vector = self.calculate_normalized_relative_angle(current_center, angle, self.target)

        # Flatten the state into a single array
        flattened_state = np.concatenate([
            np.array([raw_distance_to_target], dtype=np.float32),
            np.array(prev_phase_values, dtype=np.float32),  # Include the captured phase values
            np.array(prev_amplitude_values, dtype=np.float32),  # Include the captured amplitude values
            np.array([target_vector], dtype=np.float32)  # Include the target vector
        ])

        self.prev_angle = normalized_angle
        self.prev_center = current_center

        # Store initial distance for efficiency calculations
        if current_center is not None and self.target is not None:
            self.initial_distance_to_target = np.linalg.norm(
                np.array(current_center) - np.array(self.target)
            )

        return flattened_state, info

    def init_episode(self):
        self.current_episode += 1  # Increment the episode number
        self.current_step = 0  # Reset the step count for the new episode


    def save_to_hdf5(
            self,
            hdf5_file_path,
            current_episode,
            current_step,
            current_center,
            reward,
            done,
            info_reason,
            frame,
            original_frame,  # <-- new line
            pred_bgr,
            prev_phase_values,  # New parameter
            prev_amplitude_values,  # New parameter
            centers,
            angle,
            distance_to_boundary,
            target,
            circle_radius,
            target_vector,
            raw_distance_to_target,
            action
    ):
        with h5py.File(hdf5_file_path, "a") as hf:
            # Check if the episode group already exists, create it if not
            episode_grp_name = f"episode_{current_episode}"
            if episode_grp_name in hf:
                episode_grp = hf[episode_grp_name]
            else:
                episode_grp = hf.create_group(episode_grp_name)

            # Now create a group for the current step within the episode group
            step_grp = episode_grp.create_group(f"step_{current_step}")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
                           :-3
                           ]  # Millisecond precision
            step_grp.create_dataset("timestamp", data=current_time)
            step_grp.create_dataset("current_center", data=current_center)
            step_grp.create_dataset("reward", data=reward)
            step_grp.create_dataset("done", data=done)
            step_grp.create_dataset("info_reason", data=str(info_reason))
            step_grp.create_dataset("frame", data=frame)
            step_grp.create_dataset("original_frame", data=original_frame)  # <-- new line
            step_grp.create_dataset("pred_gbr", data=pred_bgr)
            step_grp.create_dataset("prev_phase_values", data=prev_phase_values)
            step_grp.create_dataset("prev_amplitude_values", data=prev_amplitude_values)
            step_grp.create_dataset("centers", data=str(centers))
            step_grp.create_dataset("angle", data=angle)
            step_grp.create_dataset("distance_to_boundary", data=distance_to_boundary)
            step_grp.create_dataset("target", data=target)
            step_grp.create_dataset("circle_radius", data=circle_radius)
            step_grp.create_dataset("target_vector", data=target_vector)
            step_grp.create_dataset("raw_distance_to_target", data=raw_distance_to_target)
            step_grp.create_dataset("action", data=action)  # New dataset for action


    def save_to_hdf5_thread(self, current_center, reward, done, info, frame, original_frame, 
                           pred_bgr, phase_adjustments, amplitude_adjustments,
                           centers, angle, distance_to_boundary, target_vector,
                           raw_distance_to_target, action):
        threading.Thread(
            target=self.save_to_hdf5,
            args=(
                self.hdf5_file_path,
                self.current_episode,
                self.current_step,
                current_center,
                reward,
                done,
                info.get("reason", ""),
                frame,
                original_frame,
                pred_bgr,
                phase_adjustments,
                amplitude_adjustments,
                centers,
                angle,
                distance_to_boundary,
                self.target,
                self.circle_radius,
                target_vector,
                raw_distance_to_target,
                action
            ),
        ).start()

    def check_robot_state(self, current_center, angle):
        """
        Enhanced state checking with clear termination conditions
        """
        done = False  # Natural termination
        truncated = False  # Forced termination
        info = {"termination_reason": None}

        # Check for invalid angle
        if angle is None:
            truncated = True
            info["termination_reason"] = "Invalid angle"
            return 0.0, done, truncated, info

        normalized_angle = angle / 360.0
        normalized_angle = float(f"{normalized_angle:.3f}")

        # Check for invalid center position
        if current_center is None:
            truncated = True
            info["termination_reason"] = "Lost robot position"
            return normalized_angle, done, truncated, info

        # Check boundary violation
        distance_to_boundary = self.distance_to_boundary(current_center)
        if distance_to_boundary <= (BOUNDARY_THRESHOLD / self.circle_radius):
            truncated = True
            info["termination_reason"] = "Boundary violation"
            return normalized_angle, done, truncated, info

        # Check target reached (success condition)
        distance_to_target = np.linalg.norm(np.array(current_center) - np.array(self.target))
        if distance_to_target <= (30 / (2 * self.circle_radius)):
            done = True
            info["termination_reason"] = "Target reached"
            self.target_reached = True
            return normalized_angle, done, truncated, info

        # Check maximum steps (timeout)
        if self.current_step >= 1500:
            truncated = True
            info["termination_reason"] = "Maximum steps exceeded"
            return normalized_angle, done, truncated, info

        return normalized_angle, done, truncated, info

    def set_sg_parameters(self, channel, param_type, value):
        if param_type not in ['phase', 'amplitude']:
            raise ValueError("Parameter type must be 'phase' or 'amplitude'.")
        if channel not in self.sg_parameters:
            raise ValueError("Invalid channel. Must be 'X', 'Y', or 'Z'.")

        # Print before change
        # print(f"\n=== Signal Generator Parameter Change ===")
        # print(f"Previous {channel} {param_type}: {self.sg_parameters[channel][param_type]}")
        
        self.sg_parameters[channel][param_type] = value
        
        # Print after change
        # print(f"New {channel} {param_type}: {value}")

        # Map 'X', 'Y', 'Z' to 1, 2, 3 for hardware interaction
        channel_mapping = {'X': 1, 'Y': 2, 'Z': 3}
        hardware_channel = channel_mapping[channel]

        # Print current state of all parameters
        # print("\nCurrent Signal Generator State:")
        # for ch in ['X', 'Y', 'Z']:
        #     print(f"Channel {ch}:")
        #     print(f"  Phase: {self.sg_parameters[ch]['phase']}")
        #     print(f"  Amplitude: {self.sg_parameters[ch]['amplitude']}")

        # Print hardware command
        # print(f"\nSending to hardware:")
        # print(f"Channel: {hardware_channel}")
        # print(f"Parameter: {param_type}")
        # print(f"Value: {value}{'ampere' if param_type == 'amplitude' else ''}")
        
        # sg controls the signal generator
        self.sg.set_parameter(hardware_channel, param_type, f"{value}{'ampere' if param_type == 'amplitude' else ''}")

    def step(self, action):
        # print(f"\n=== Step {self.current_step} ===")
        # print(f"Action received: {action}")
        
        frame,centers, angle, pred_bgr = self.get_and_process_frame()
        original_frame = frame.copy()  # <-- new line
        front_center = centers.get(1, None)
        back_center = centers.get(2, None)
        current_center = self.calculate_current_center(front_center, back_center)
        normalized_angle, done, truncated, info = self.check_robot_state(current_center, angle)

        # Calculate distance to target before action
        raw_distance_to_target = float(f"{np.linalg.norm(np.array(current_center) - np.array(self.target)):.3f}")

        # Capture the current phase and amplitude values before the action
        prev_phase_values = [self.sg_parameters[ch]['phase'] / 315 for ch in ['X', 'Y', 'Z']]
        prev_amplitude_values = [self.sg_parameters[ch]['amplitude'] / 9 for ch in ['X', 'Y', 'Z']]

        phase_values = [0, 90, 180, 315]
        amplitude_values = [0, 5, 7, 9]

        if action < 12:  # Phase adjustments
            channel_index = action // 4
            phase_value = phase_values[action % 4]
            param_type = 'phase'
            # print(f"\nPhase adjustment:")
            # print(f"Channel index: {channel_index}")
            # print(f"New phase value: {phase_value}")
        else:  # Amplitude adjustments
            channel_index = (action - 12) // 4
            amplitude_value = amplitude_values[(action - 12) % 4]
            param_type = 'amplitude'
            # print(f"\nAmplitude adjustment:")
            # print(f"Channel index: {channel_index}")
            # print(f"New amplitude value: {amplitude_value}")

        channel = ['X', 'Y', 'Z'][channel_index]
        value = phase_value if param_type == 'phase' else amplitude_value
        
        # print(f"Selected channel: {channel}")
        # print(f"Parameter type: {param_type}")
        # print(f"Value to set: {value}")

        # Set the parameter
        self.set_sg_parameters(channel, param_type, value)

        # Map 'X', 'Y', 'Z' to 1, 2, 3 for hardware interaction
        channel_mapping = {'X': 1, 'Y': 2, 'Z': 3}
        hardware_channel = channel_mapping[channel]

        # Assuming self.sg is some object that controls your actual hardware
        self.sg.set_parameter(hardware_channel, param_type, f"{value}{'ampere' if param_type == 'amplitude' else ''}")

        # Get new state after action
        frame, centers, angle, pred_bgr = self.get_and_process_frame()
        front_center = centers.get(1, None)
        back_center = centers.get(2, None)
        current_center = self.calculate_current_center(front_center, back_center)
        normalized_angle, done, truncated, info = self.check_robot_state(current_center, angle)

        reward = self.calculate_reward(current_center, prev_distance_to_target=self.previous_distance_to_target)
        self.previous_distance_to_target = raw_distance_to_target
        self.cumulative_reward += reward
        self.current_step += 1
        target_vector = self.calculate_normalized_relative_angle(current_center, angle, self.target)

        # Save data and visualize
        self.save_to_hdf5_thread(
            current_center, reward, done, info, 
            frame, original_frame, pred_bgr,
            prev_phase_values, prev_amplitude_values,
            centers, angle, 
            self.distance_to_boundary(current_center),
            target_vector,
            raw_distance_to_target,
            action
        )

        self.visualize_frame(frame, centers, angle, reward, current_center)

        # Calculate flattened state using post-action values
        flattened_state = np.concatenate([
            
            np.array([np.linalg.norm(np.array(current_center) - np.array(self.target))], dtype=np.float32),
            
            np.array([self.sg_parameters[ch]['phase'] / 315 for ch in ['X', 'Y', 'Z']], dtype=np.float32),  # Include the captured phase values
            np.array([self.sg_parameters[ch]['amplitude'] / 9 for ch in ['X', 'Y', 'Z']], dtype=np.float32),  # Include the captured amplitude values
            np.array([target_vector], dtype=np.float32)  # Include the target vector
        ])

        self.prev_angle = normalized_angle
        self.prev_center = current_center

        if done or truncated:
            self.episode_end_reason = info["termination_reason"]  # <--- store reason


        return flattened_state, reward, done, truncated, info

    def visualize_frame(self, frame, centers, angle, reward, current_center):


        front_center = centers.get(1, None)
        back_center = centers.get(2, None)
        # Calculate penalty for boundary proximity
        denormalized_current_center = [
            (coord * 2 * self.circle_radius) + (self.circle_center[i] - self.circle_radius)
            for i, coord in enumerate(current_center)]
        BOUNDARY_THRESHOLD=60/self.circle_radius
        denormalized_target = [(coord * 2 * self.circle_radius) + (self.circle_center[i] - self.circle_radius) for
                               i, coord in enumerate(self.target)]

        # Draw the physical boundary
        if hasattr(self, 'circle_center') and hasattr(self, 'circle_radius'):
            cv2.circle(frame, self.circle_center, self.circle_radius, (255, 255, 255), 2)

        # Visualize the distance to boundary
        if denormalized_current_center is not None and hasattr(self, 'circle_center'):
            vec_to_center = np.array(self.circle_center) - np.array(denormalized_current_center)
            vec_to_center = vec_to_center / np.linalg.norm(vec_to_center)
            point_on_boundary = np.array(self.circle_center) - vec_to_center * self.circle_radius
            cv2.line(frame, tuple(map(int, denormalized_current_center)), tuple(map(int, point_on_boundary)), (255, 0, 255), 2)
            distance_to_boundary = self.distance_to_boundary(current_center)
            if distance_to_boundary < BOUNDARY_THRESHOLD:
                penalty = BOUNDARY_THRESHOLD - distance_to_boundary
                text_position = tuple(map(int, (denormalized_current_center + point_on_boundary) / 2))
                cv2.putText(frame, f"Penalty: {-penalty:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        cv2.imshow("Debug Frame", frame)
        cv2.waitKey(1)

        if denormalized_current_center is not None:
            self.center_trajectory.append(denormalized_current_center)
            arrow_length = 50  # Length of the arrow in pixels

            if angle is not None:
                angle_rad_corrected = -np.radians(angle)
                arrow_end = (int(denormalized_current_center[0] + arrow_length * np.cos(angle_rad_corrected)),
                             int(denormalized_current_center[1] + arrow_length * np.sin(angle_rad_corrected)),)

                if front_center is not None and back_center is not None:
                    cv2.arrowedLine(
                        frame,
                        tuple(map(int, back_center)),
                        tuple(map(int, front_center)),
                        (0, 0, 255),
                        1,
                        tipLength=0.2,
                    )
                    cv2.putText(
                        frame,
                        f"Angle: {angle:.2f} degrees",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        1,
                    )
            else:
                print("Warning: Not drawing arrow because angle is None.")

        for center in self.center_trajectory[::40]:
            cv2.circle(frame, tuple(map(int, center)), 3, (255, 255, 0), -1)

        if not np.all(denormalized_current_center == [0, 0]):
            cv2.circle(frame, tuple(map(int, denormalized_current_center)), 3, (0, 255, 0), -1)

        # Visualize the target and line to the target
        if denormalized_target is not None and denormalized_current_center is not None:
            cv2.circle(frame, tuple(map(int, denormalized_target)), 25, (0, 255, 0), 2)  # Draw target circle
            cv2.line(frame, tuple(map(int, denormalized_current_center)), tuple(map(int, denormalized_target)), (255, 0, 0),
                     2)  # Draw line to target

            # Display the reward
            text_position = (
            int((denormalized_current_center[0] + denormalized_target[0]) / 2), int((denormalized_current_center[1] + denormalized_target[1]) / 2))
            cv2.putText(frame, f"Reward: {reward:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        # Add target vector visualization
        if current_center is not None and angle is not None and self.target is not None:
            target_vector = self.calculate_normalized_relative_angle(current_center, angle, self.target)
            if target_vector is not None:
                # Display target vector value
                cv2.putText(
                    frame,
                    f"Target Vector: {target_vector:.3f}",
                    (50, 80),  # Position below angle display
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    1,
                )

                # Visualize relative angle as a colored arc
                if denormalized_current_center is not None:
                    center_point = tuple(map(int, denormalized_current_center))
                    radius = 40  # Radius of the arc
                    start_angle = -np.radians(angle)  # Robot's current orientation
                    end_angle = -np.radians(angle) + 2 * np.pi * target_vector  # Add the relative angle
                    cv2.ellipse(frame, center_point, (radius, radius),
                                0, np.degrees(start_angle), np.degrees(end_angle),
                                (0, 255, 255), 2)

        # Display current phase and amplitude values for all channels
        phase_values = [self.sg_parameters[ch]['phase'] for ch in ['X', 'Y', 'Z']]
        amplitude_values = [self.sg_parameters[ch]['amplitude'] for ch in ['X', 'Y', 'Z']]

        # Phase values display
        cv2.putText(
            frame,
            f"Phase X,Y,Z: ({phase_values[0]}, {phase_values[1]}, {phase_values[2]})",
            (50, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            1,
        )
        
        # Amplitude values display
        cv2.putText(
            frame,
            f"Amp X,Y,Z: ({amplitude_values[0]}, {amplitude_values[1]}, {amplitude_values[2]})",
            (50, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            1,
        )

        # Display last action taken with corrected value ranges
        if hasattr(self, 'last_action'):
            if self.last_action < 12:  # Phase adjustments
                channel_index = self.last_action // 4
                phase_value = [0, 90, 180, 315][self.last_action % 4]
                channel = ['X', 'Y', 'Z'][channel_index]
                cv2.putText(
                    frame,
                    f"Action: Phase {channel}={phase_value}°",
                    (50, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    1,
                )
            else:  # Amplitude adjustments
                channel_index = (self.last_action - 12) // 4
                amplitude_value = [0, 5, 7, 9][(self.last_action - 12) % 4]
                channel = ['X', 'Y', 'Z'][channel_index]
                cv2.putText(
                    frame,
                    f"Action: Amplitude {channel}={amplitude_value}A",
                    (50, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    1,
                )

        cv2.imshow("Debug Frame", frame)
        cv2.waitKey(1)

    def close(self):
        if hasattr(self, "cam_thread"):
            self.cam_thread.join()


