import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from pycocotools import mask as coco_mask
from albumentations.pytorch import ToTensorV2
from skimage.transform import resize
import matplotlib.pyplot as plt


class RobotDataset(Dataset):
    def __init__(self, img_folder, annotation_file, img_size=512, augmentations=None):
        self.img_folder = img_folder
        self.img_size = img_size
        self.augmentations = augmentations
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_data = self.coco.loadImgs(image_id)[0]
        image = Image.open(os.path.join(self.img_folder, image_data['file_name'])).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        image = np.array(image, dtype=np.float32) / 255

        ann_ids = self.coco.getAnnIds(imgIds=image_data['id'])
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        for ann in anns:
            rles = self.coco.annToRLE(ann)
            decoded_mask = coco_mask.decode(rles)
            resized_mask = resize(decoded_mask, (self.img_size, self.img_size), order=0, mode="constant",
                                  anti_aliasing=False)

            # Here we set the class value for the mask
            mask[resized_mask > 0.5] = ann['category_id']

        # # Check the number of unique classes in the mask
        # unique_classes = np.unique(mask)
        # if len(unique_classes) == 3:
        #     print(f"Image {image_data['file_name']} has only two classes")

        # Apply augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


# def main():
#     img_folder = 'multilabel_dataset1_2/images'  # Replace with your image folder path
#     annotation_file = 'multilabel_dataset1_2/annotations.json'  # Replace with your annotations file path
#
#     # Define a color map for different classes; each class ID should map to an RGB color
#     class_colors = {
#         0: [0, 0, 0],  # Background
#         1: [0, 255, 0],  # Class 1
#         2: [255, 0, 0],  # Class 2
#     }
#
#     # Define augmentations (you can customize this)
#     augmentations = A.Compose([
#         A.HorizontalFlip(p=0.5),
#         ToTensorV2()
#     ])
#
#     # Initialize dataset
#     dataset = RobotDataset(img_folder, annotation_file, img_size=512, augmentations=augmentations)
#
#     # Loop to visualize first few images and masks
#     for i in range(102):  # You can change the range to see more or fewer images
#         image, mask = dataset[i]
#
#         # If you applied ToTensorV2(), the image would be in CxHxW format. Change it back to HxWxC format for visualization
#         image = image.permute(1, 2, 0).numpy()
#
#         # Create an empty mask with 3 channels (to make an RGB image)
#         color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
#
#         # Populate the empty mask with colors based on class IDs
#         for class_id, color in class_colors.items():
#             color_mask[mask == class_id] = color
#
#         plt.figure(figsize=(10, 5))
#
#         plt.subplot(1, 2, 1)
#         plt.imshow(image)
#         plt.title('Image')
#
#         plt.subplot(1, 2, 2)
#         plt.imshow(color_mask)
#         plt.title('Colored Mask')
#
#         plt.show()
#
#
# if __name__ == "__main__":
#     main()

