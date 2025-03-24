
Author: Zakir Ullah  
Email: zakirullahmech@outlook.com  

## Project Overview

This project uses a U-Net model for segmenting robotic parts in images. Key files and their roles:

- `dataset.py`: Loads and processes images and annotations using the `RobotDataset` class with data augmentations.
- `main.py`: Entry point to initialize and train the model using `RobotSegmentation`.
- `model.py`: Defines the U-Net model architecture for segmentation.
- `train.py`: Manages the training loop, validation, and metrics. Includes early stopping.
- `util.py`: Provides helper functions for plotting, early stopping, and saving data.

## Usage

1. **Prepare Dataset**: Organize your dataset with an `images` folder and `annotations.json` file.
2. **Run Training**: Execute `main.py` to start training. Checkpoints and metrics are saved in the `results` folder.
3. **Evaluate**: Training script saves IoU, precision, recall, and F1-score metrics for analysis.

## File Modification Hints

- `dataset.py`:
  - Adjust `img_folder` and `annotation_file` paths in the `RobotDataset` class if your dataset location changes.
  - Modify `img_size` for resizing images if higher or lower resolution is needed.

- `train.py`:
  - Change `batch_size`, `epochs`, and `lr` (learning rate) in `RobotSegmentation` to tune training performance.
  - Modify augmentations in `load_data()` for different data transformations.

- `model.py`:
  - Adjust `n_class` in `UNet` if you change the number of segmentation classes.

- `util.py`:
  - `plot_training_curves` and `plot_confusion_matrix` can be modified to adjust plot size or save paths.

## Contact

For questions, contact Zakir Ullah at zakirullahmech@outlook.com.
