import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from train_unet_model.model import UNet
from sklearn.decomposition import PCA
import cv2
import datetime
import threading
from gxipy import DeviceManager, GxSwitchEntry
from queue import Queue, Empty
import csv
import torch.quantization


class RobotOrientationDetector:
    def __init__(self, model_path, device, save_fps=30):
        self.device = device
        self.model = self._load_model(model_path)
        # Generate a unique filename using the current timestamp
        current_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_file_path = f'Training_data/camera_training_data_{current_timestamp}.csv'
        self.initialize_csv()
        self.prev_orientation = None
        self.orientations_buffer = []
        self.N = 2
        self.robot_centers = []
        self.out = None  # Initialize the video writer later
        self.save_fps = save_fps
        self.last_saved_time = None  # Track the last time a frame was saved
        self.raw_out = None  # Initialize the video writer for raw video later
        self.predicted_out = None  # Initialize the video writer for predicted video later

    def _load_model(self, model_path):
        """Load and trace the model."""
        model = UNet(1).to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8,  # the target dtype for quantized weights
        )

        traced_model = torch.jit.trace(
            quantized_model, torch.randn(1, 3, 256, 256).to(self.device)
        )
        return traced_model

    def get_robot_orientation(self, binary_mask, angular_velocity=None):
        cleaned_mask = cv2.morphologyEx(
            binary_mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            return None, None, None, None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 100:
            return None, None, None, None

        hull = cv2.convexHull(largest_contour)

        coords = np.array(hull.reshape(-1, 2))
        pca = PCA(n_components=2)
        pca.fit(coords)
        orientation = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])

        if orientation < 0:
            orientation += 2 * np.pi

        if angular_velocity is not None:
            orientation += angular_velocity
            orientation %= 2 * np.pi  # Normalize back to [0, 2π]

        if self.prev_orientation is not None:
            potential_ori1 = orientation
            potential_ori2 = (orientation + np.pi) % (2 * np.pi)

            diff1 = abs(potential_ori1 - self.prev_orientation)
            diff2 = abs(potential_ori2 - self.prev_orientation)
            orientation = potential_ori1 if diff1 < diff2 else potential_ori2

            if min(diff1, diff2) > np.pi / 4:
                orientation = self.prev_orientation

        self.orientations_buffer.append(orientation)
        if len(self.orientations_buffer) > self.N:
            self.orientations_buffer.pop(0)
        average_orientation = np.mean(np.cos(self.orientations_buffer)) + 1j * np.mean(np.sin(self.orientations_buffer))
        average_orientation = np.angle(average_orientation)

        self.prev_orientation = orientation

        moments = cv2.moments(hull)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            center = np.array([cy, cx])
        else:
            center = None

        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        rotating_bounding_box = {
            "top_left": box[0],
            "top_right": box[1],
            "bottom_right": box[2],
            "bottom_left": box[3],
        }

        perimeter = cv2.arcLength(hull, True)

        return  center, rotating_bounding_box



    def process_frame(self, frame):
        frame_processed = cv2.resize(frame, (256, 256))
        frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)  # Ensure it's RGB
        frame_processed = np.array(frame_processed, dtype=np.float32) / 255.0
        frame_processed = torch.tensor(frame_processed).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(frame_processed)
            width, height = frame.shape[1], frame.shape[0]
            mask = cv2.resize(outputs[0].squeeze().detach().cpu().numpy(), (width, height)) > 0.5

        return mask

    def process_video(self, video_path, output_path='demovideo.avi'):
        cap = cv2.VideoCapture(video_path)
        self._process_cap(cap, output_path)

    def process_frame_live(self, frame):
        mask = self.process_frame(frame)
        if mask is None or mask.shape[0] == 0 or mask.shape[1] == 0:
            print("Invalid mask generated.")
            return None

        orientation, center, rotating_bounding_box, perimeter = self.get_robot_orientation(
            mask
        )  # Modified line
        mask_colored, _ = self._visualize(
            mask, orientation, center, rotating_bounding_box, perimeter
        )  # Update to pass all new values

        if mask_colored is not None:
            combined = np.hstack((frame, mask_colored))
            cv2.imshow("Robot Orientation Detection Live", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.close()
        else:
            print("Error: mask_colored is None.")

        # Convert the mask to uint8 type
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Create a visualization of the detected mask
        mask_colored = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            mask_colored, contours, -1, (0, 0, 255), 2
        )  # Draw red contours on the mask

        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        combined = np.hstack((frame, mask_colored))



        # Overlay timestamp function
        def overlay_timestamp(image):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            color = (255, 255, 255)  # White color
            position = (10, image.shape[0] - 10)  # Bottom left corner with a small margin
            cv2.putText(image, current_timestamp, position, font, font_scale, color, font_thickness, cv2.LINE_AA)

        # Overlay timestamp on raw frame and predicted mask
        overlay_timestamp(frame)
        overlay_timestamp(mask_colored)

        # Overlay timestamp on combined video
        overlay_timestamp(combined)

        # Save raw frame
        if self.raw_out is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.raw_out = cv2.VideoWriter(f'Training_data/raw_video_{current_timestamp}.avi', fourcc, self.save_fps,
                                           (frame.shape[1], frame.shape[0]))
        self.raw_out.write(frame)

        # Save predicted frame (or mask)
        if self.predicted_out is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.predicted_out = cv2.VideoWriter(f'Training_data/predicted_video_{current_timestamp}.avi', fourcc,
                                                 self.save_fps,
                                                 (mask_colored.shape[1], mask_colored.shape[0]))
        self.predicted_out.write(mask_colored)

        cv2.imshow('Robot Orientation Detection Live', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        if self.raw_out is not None:
            self.raw_out.release()
        if self.predicted_out is not None:
            self.predicted_out.release()
        cv2.destroyAllWindows()

    def process_camera(self, camera_index=0, output_path='live_demo.avi'):
        cap = cv2.VideoCapture(camera_index)
        self._process_cap(cap, output_path)

    def _process_cap(self, cap, output_path):
        if not cap.isOpened():
            print("Error: Couldn't open the video or camera.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mask = self.process_frame(frame)
            (
                orientation,
                center,
                rotating_bounding_box,
                perimeter,
            ) = RobotOrientationDetector.get_robot_orientation(
                mask
            )  # Unpack the return values
            mask_colored, _ = self._visualize(
                mask, orientation, center, rotating_bounding_box, perimeter
            )  # Update to pass all new values

            if mask_colored is not None:
                combined = np.hstack((frame, mask_colored))
                out.write(combined)
                cv2.imshow("Robot Orientation Detection", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()  # Release the output video writer.
        cv2.destroyAllWindows()

    def initialize_csv(self):
        """
        Initialize the CSV file with headers if it doesn't exist.
        """
        if not os.path.exists(self.data_file_path):
            with open(self.data_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['timestamp', 'center_x', 'center_y', 'orientation'])

    def save_training_data(self, center, average_orientation):
        """
        Save center_x, center_y, orientation, and real timestamp in a CSV file.
        """
        current_timestamp = datetime.datetime.now()
        with open(self.data_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [current_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], center[1], center[0], average_orientation])

    def _visualize(self, mask, orientation, center, rotating_bounding_box, perimeter):
        mask_colored = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

        # Add timestamp to the mask_colored
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Includes milliseconds
        cv2.putText(
            mask_colored,
            timestamp,
            (10, mask_colored.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Draw robot centers if available
        if center is not None:
            self.robot_centers.append((center[1], center[0]))
            for i in range(1, len(self.robot_centers)):
                cv2.line(
                    mask_colored,
                    self.robot_centers[i - 1],
                    self.robot_centers[i],
                    (0, 255, 0),
                    2,
                )

        # Draw orientation if available
        if orientation is not None and center is not None:
            cv2.arrowedLine(
                mask_colored,
                (center[1], center[0]),
                (
                    int(center[1] + 50 * np.cos(orientation)),
                    int(center[0] + 50 * np.sin(orientation)),
                ),
                (0, 0, 255),
                2,
            )
            angle_text = f"{np.degrees(orientation):.2f} deg"
            cv2.putText(
                mask_colored,
                angle_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            center_text = f"Center: ({center[1]}, {center[0]})"
            cv2.putText(
                mask_colored,
                center_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        # Draw rotating bounding box if available
        if rotating_bounding_box is not None:
            for start, end in zip(
                rotating_bounding_box.values(),
                list(rotating_bounding_box.values())[1:]
                + [list(rotating_bounding_box.values())[0]],
            ):
                cv2.line(mask_colored, tuple(start), tuple(end), (255, 0, 0), 2)

        # Draw perimeter if available
        if perimeter is not None:
            perimeter_text = f"Perimeter: {perimeter:.2f}"
            cv2.putText(
                mask_colored,
                perimeter_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        return mask_colored, orientation





class Camera:
    def __init__(self):
        self.frame_queue = Queue(maxsize=10)

    def put_frame(self, frame):
        if self.frame_queue.full():
            _ = self.frame_queue.get()
        self.frame_queue.put(frame)

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=1)
        except Empty:
            return None

    def capture_video(self):
        device_manager = DeviceManager()
        if not device_manager.update_device_list()[0]:
            print("No devices found.")
            return

        cam = device_manager.open_device_by_index(1)
        if not cam.PixelColorFilter.is_implemented():
            print("This sample does not support mono camera.")
            cam.close_device()
            return

        cam.TriggerMode.set(GxSwitchEntry.OFF)
        cam.ExposureTime.set(20000.0)
        cam.Gain.set(5.0)
        cam.stream_on()

        ratio = None
        try:
            while True:
                raw_image = cam.data_stream[0].get_image()
                rgb_image = raw_image.convert("RGB")
                numpy_img = rgb_image.get_numpy_array()
                if numpy_img is not None:
                    # Resize the frame to the desired dimensions
                    desired_size = (850, 710)  # Change this to the desired size
                    numpy_img = cv2.resize(numpy_img, desired_size)
                    self.put_frame(cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error while capturing video: {e}")
        finally:
            cam.stream_off()
            cam.close_device()

    def feed_to_detector(self, detector):
        try:
            while True:
                frame = self.get_frame()
                if frame is not None:
                    detector.process_frame_live(frame)
        except Exception as e:
            print(f"Error while feeding frames to detector: {e}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = RobotOrientationDetector(model_path='Multiclass_2_model_12_13_23.pth', device=device)

    cam = Camera()
    capture_thread = threading.Thread(target=cam.capture_video)
    detector_thread = threading.Thread(target=cam.feed_to_detector, args=(detector,))

    capture_thread.start()
    detector_thread.start()

    try:
        capture_thread.join()
        detector_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user, closing all threads.")
        detector.close()  # Ensure the video writer is closed
