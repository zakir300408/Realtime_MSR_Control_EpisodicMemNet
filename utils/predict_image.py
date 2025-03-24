import numpy as np
import torch
import cv2
import math
import time

class ResultVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.class_names = ["Background", "B", "F"]
        self.color_map = np.array([
            [0, 0, 0], [255, 255, 0], [255, 0, 255]
        ])

    @staticmethod
    def label_to_rgb(label, color_map):
        rgb_image = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for idx, color in enumerate(color_map):
            rgb_image[label == idx] = color
        return rgb_image

    def find_class_centers(self, predicted_classes):
        centers = {}
        for class_idx in [1, 2]:
            mask = predicted_classes == class_idx
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circular_contours = [cnt for cnt in contours if self.is_circular(cnt)]
            if circular_contours:
                largest_contour = max(circular_contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centers[class_idx] = (cX, cY)
        return centers

    @staticmethod
    def is_circular(contour, circularity_threshold=0.4):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        circularity = 4 * math.pi * area / (perimeter ** 2)
        return circularity > circularity_threshold

    def process_frame(self, frame, device, color_map):
        start_time = time.time()

        self.model.eval()
        frame_resized = cv2.resize(frame, (512, 512))
        frame_blurred = cv2.GaussianBlur(frame_resized, (5, 5), 0)  # Apply Gaussian blur
        frame_rgb = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2RGB)
        tensor_frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

        part1_time = time.time() - start_time
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(tensor_frame)

        part2_time = time.time() - start_time
        start_time = time.time()

        max_values, predicted_classes = torch.max(outputs, dim=1)
        confidence_mask = max_values > 60
        predicted_classes = predicted_classes * confidence_mask
        predicted_classes = predicted_classes.cpu().numpy()

        part3_time = time.time() - start_time
        start_time = time.time()

        centers = self.find_class_centers(predicted_classes[0])

        part4_time = time.time() - start_time
        start_time = time.time()

        angle = None
        orientation_labels = None
        if 1 in centers and 2 in centers:
            pt1, pt2 = tuple(map(int, centers[1])), tuple(map(int, centers[2]))
            angle = math.degrees(math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
            angle = (angle + 360) % 360
            orientation_labels = (0, 1)

        part5_time = time.time() - start_time
        start_time = time.time()

        pred_rgb = self.label_to_rgb(predicted_classes[0], color_map)
        pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

        part6_time = time.time() - start_time

        # print(
        #     f"Part1: {part1_time:.4f}, Part2: {part2_time:.4f}, Part3: {part3_time:.4f}, Part4: {part4_time:.4f}, Part5: {part5_time:.4f}, Part6: {part6_time:.4f}")

        return frame_resized, centers, angle, pred_bgr, orientation_labels  # Return the resized frame, centers, angle, visual representation, and orientation labels

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (1024, 512))

        with torch.no_grad():
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    start_time = time.time()
                    frame_resized, centers, angle, pred_bgr, orientation_labels = self.process_frame(
                         frame, self.device, self.color_map)

                    for class_idx, center in centers.items():
                        cv2.circle(pred_bgr, tuple(map(int, center)), 5, (0, 255, 0), -1)  # Draw circles on pred_bgr
                        label_text = f'Label: {orientation_labels[class_idx - 1]}'  # Get the label text
                        label_position = (int(center[0]), int(center[1] - 10))  # Position the label text above the center
                        cv2.putText(pred_bgr, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 2)  # Draw label text on pred_bgr

                    if 1 in centers and 2 in centers:
                        pt1, pt2 = tuple(map(int, centers[1])), tuple(map(int, centers[2]))
                        cv2.arrowedLine(pred_bgr, pt1, pt2, (0, 0, 255), 2)  # Draw arrow on pred_bgr
                        angle_text = f'Orientation angle: {angle:.2f}°'
                        text_position = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                        cv2.putText(pred_bgr, angle_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 2)  # Draw angle text on pred_bgr

                    # Concatenating the original frame and prediction side by side
                    concat_frame = cv2.hconcat([frame_resized, pred_bgr])
                    cv2.imshow('Frame and Prediction', concat_frame)  # Displaying the concatenated frames
                    out.write(concat_frame)  # Writing the concatenated frame to the output video
                    end_time = time.time()  # Capture end time
                    duration = end_time - start_time  # Compute duration

                    print(f"Time taken to process this frame: {duration:.4f} seconds")

                    # Wait for the user to press 'q' key to exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    model = torch.load('Multiclass_2_model_12_13_23.pth')
    end_time = time.time()

    print(f"Time taken to load the model: {end_time - start_time:.4f} seconds")

    result_visualizer = ResultVisualizer(model, device)
    result_visualizer.process_video('camera/output_20231011-234726.avi', 'output_video.avi')  # Updated method call

def predict_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    model = torch.load('Multiclass_2_model_12_13_23.pth')
    end_time = time.time()

    print(f"Time taken to load the model: {end_time - start_time:.4f} seconds")

    result_visualizer = ResultVisualizer(model, device)

    frame = cv2.imread(image_path)
    frame_resized, centers, angle, pred_bgr, orientation_labels = result_visualizer.process_frame(frame, device, result_visualizer.color_map)

    for class_idx, center in centers.items():
        cv2.circle(pred_bgr, tuple(map(int, center)), 5, (0, 255, 0), -1)
        label_text = f"Label: {orientation_labels[class_idx - 1]}"
        label_position = (int(center[0]), int(center[1] - 10))
        cv2.putText(pred_bgr, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if 1 in centers and 2 in centers:
        pt1, pt2 = tuple(map(int, centers[1])), tuple(map(int, centers[2]))
        cv2.arrowedLine(pred_bgr, pt1, pt2, (0, 0, 255), 2)
        angle_text = f"Orientation angle: {angle:.2f}"
        text_position = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(pred_bgr, angle_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    print("Orientation angle:", angle)
    print("Centers:", centers)

    cv2.imshow('Frame and Prediction', pred_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #main()
    predict_image('E:/WD_RL_Control/test_image_pred.png')






