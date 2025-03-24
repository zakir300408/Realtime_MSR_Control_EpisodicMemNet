# main.py
from train import RobotSegmentation

if __name__ == "__main__":
    robot_segmentation = RobotSegmentation(dataset_dir='multilabel_dataset1_2/')
    robot_segmentation.train()
