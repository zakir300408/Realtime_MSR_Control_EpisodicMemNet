import logging
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from Markov_Env_for_training import RobotEnv
import torch

# Simple callback for model saving
class SaveModelCallback(BaseCallback):
    def __init__(self, save_path, save_freq=5000):
        super().__init__()
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"{self.save_path}/model_{self.n_calls}")
        return True

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create save directory
    save_path = "saved_models"
    os.makedirs(save_path, exist_ok=True)

    # Initialize environment
    env = RobotEnv()
    vec_env = DummyVecEnv([lambda: env])

    # Enhanced PPO configuration with better exploration
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,           # Reduced for more stable learning
        n_steps=2048,                 # Increased for better exploration
        batch_size=64,                # Smaller batches for better generalization
        n_epochs=10,
        gamma=0.99,                   # Standard discount factor
        ent_coef=0.2,               # Added entropy coefficient for exploration
        clip_range=0.2,              # Standard PPO clip range
        max_grad_norm=0.5,           # Added gradient clipping
        vf_coef=0.2,                 # Value function coefficient
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],        # Wider policy network
                vf=[256, 256]         # Wider value network
            ),
            activation_fn=torch.nn.ReLU,
            ortho_init=True           # Use orthogonal initialization
        )
    )

    # Setup callback with more frequent saving
    callback = SaveModelCallback(save_path, save_freq=1000)  # Save more frequently

    # Train model with increased timesteps
    try:
        model.learn(
            total_timesteps=100000,   # Increased training time
            callback=callback
        )
        model.save(f"{save_path}/final_model")
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training interrupted: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
