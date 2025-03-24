import h5py
import pandas as pd
import time

class HDF5CSVProcessor:
    def __init__(self, hdf5_file_path, output_csv_path):
        self.hdf5_file_path = hdf5_file_path
        self.output_csv_path = output_csv_path
        self.phase_values = [0, 90, 180, 315]
        self.amplitude_values = [0, 5, 7, 9]
        self.channels = ['X', 'Y', 'Z']
        self.delta_columns = ['delta_x', 'delta_y', 'delta_angle']

    def compute_signal_state(self, df):
        # Convert the channel values to a combined state
        df['signal_state'] = (
            df['prev_phase_value_x'].astype(int) * 32 + 
            df['prev_phase_value_y'].astype(int) * 16 + 
            df['prev_phase_value_z'].astype(int) * 8 +
            df['prev_amplitude_value_x'].astype(int) * 4 + 
            df['prev_amplitude_value_y'].astype(int) * 2 + 
            df['prev_amplitude_value_z'].astype(int)
        )
        # Normalize to 0-1 range and round to 3 decimal places
        df['signal_state'] = (df['signal_state'] / 63).round(3)
        return df

    def calculate_transitions(self, df):
        # Mark episode boundaries
        df['is_first'] = df.groupby('episode')['step'].transform('first') == df['step']
        df['is_last'] = df.groupby('episode')['step'].transform('last') == df['step']

        # Remove first and last rows before shifting
        df = df[~(df['is_first'] | df['is_last'])].copy()
        df = df.drop(['is_first', 'is_last'], axis=1)

        # Now it's safe to shift, because the first row of each episode is gone
        post_cols = ['current_center_x', 'current_center_y', 'angle']
        for col in post_cols:
            df[f'pre_{col}'] = df.groupby('episode')[col].shift(1)

        df['pre_normalized_angle'] = df['pre_angle'] / 360

        # Calculate next states for phase and amplitude values
        for channel in self.channels:
            for value_type in ['phase', 'amplitude']:
                col = f'prev_{value_type}_value_{channel.lower()}'
                df[f'next_{value_type}_value_{channel.lower()}'] = df.groupby('episode')[col].shift(-1)

        return df

    def calculate_deltas(self, df):
        df['delta_x'] = df['current_center_x'] - df['pre_current_center_x']
        df['delta_y'] = df['current_center_y'] - df['pre_current_center_y']
        df['delta_angle'] = (df['angle'] / 360) - df['pre_normalized_angle']
        df['delta_angle'] = (df['delta_angle'] + 0.5) % 1 - 0.5
        return df

    def process(self):
        with h5py.File(self.hdf5_file_path, 'r') as hf:
            all_data = []
            episodes = sorted(hf.keys(), key=lambda x: int(x.split('_')[1]))
            
            # Data collection loop
            for episode in episodes:
                episode_grp = hf[episode]
                episode_number = int(episode.split('_')[1])
                steps = sorted(episode_grp.keys(), key=lambda x: int(x.split('_')[1]))
                for step in steps:
                    step_grp = episode_grp[step]
                    step_number = int(step.split('_')[1])
                    current_time = step_grp['timestamp'][()]
                    current_center = step_grp['current_center'][()]
                    angle = step_grp['angle'][()]
                    prev_phase_values = step_grp['prev_phase_values'][()]  # Now includes Z
                    prev_amplitude_values = step_grp['prev_amplitude_values'][()]  # Now includes Z
                    action = step_grp['action'][()]
    
                    current_center_x, current_center_y = current_center
                    prev_phase_value_x, prev_phase_value_y, prev_phase_value_z = prev_phase_values
                    prev_amplitude_value_x, prev_amplitude_value_y, prev_amplitude_value_z = prev_amplitude_values
                    
                    all_data.append([
                        current_time, episode_number, step_number,
                        current_center_x, current_center_y, angle,
                        prev_phase_value_x, prev_phase_value_y, prev_phase_value_z,
                        prev_amplitude_value_x, prev_amplitude_value_y, prev_amplitude_value_z,
                        action
                    ])

            # Create DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'episode', 'step', 'current_center_x', 'current_center_y',
                'angle', 'prev_phase_value_x', 'prev_phase_value_y', 'prev_phase_value_z',
                'prev_amplitude_value_x', 'prev_amplitude_value_y', 'prev_amplitude_value_z',
                'action'
            ])

            # Process episodes
            initial_row_count = len(df)
            df = df.groupby('episode').filter(lambda x: len(x) >= 20)
            removed_rows = initial_row_count - len(df)
            print(f"Rows removed after filtering episodes: {removed_rows}")
            
            # Process timestamps
            df['timestamp'] = df['timestamp'].apply(lambda x: x.decode('utf-8'))
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
            df.sort_values(['episode', 'timestamp'], inplace=True)

            # Round and normalize angles
            df['angle'] = df['angle'].round().astype(int)
            df['normalized_angle'] = df['angle'] / 360

            # Calculate transitions and deltas
            df = self.calculate_transitions(df)
            df = self.calculate_deltas(df)

            # Normalize actions and compute signal state
            df['normalized_action'] = (df['action'] / 23).round(3)
            df = self.compute_signal_state(df)

            # Save processed data
            df.to_csv(self.output_csv_path, index=False)
            print("Data has been successfully written.")
            return df

hdf5_file_path= r"E:\RL_realtime_IEEE TSME\Results\20250125_210121\20250125_210121_PPO_data.h5"
output_csv_path = 'RL_data_raw.csv'
processor = HDF5CSVProcessor(hdf5_file_path, output_csv_path)
df = processor.process()
