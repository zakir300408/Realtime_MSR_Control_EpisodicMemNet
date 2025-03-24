import cv2
import datetime
import threading
from gxipy import DeviceManager, GxSwitchEntry
from queue import Queue
import numpy as np

# Define constants
MAX_QUEUE_SIZE = 1
VIDEO_FOURCC = "FFV1"
VIDEO_FPS = 25
IMG_WIDTH = 1080
EXPOSURE_TIME = 40000.0
GAIN_VALUE = 8.0
CROP_FACTOR = 0.7
TOP_CROP_FACTOR = 0.1
BOTTOM_CROP_FACTOR = 0.23
VIDEO_FEED_WINDOW_NAME = "Video Feed"
CIRCLE_SCALE = 0.92


class Camera:
    def __init__(self):
        self.frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.out, self.ratio = None, None
        self.capture_enabled = False

    def put_frame(self, frame):
        if self.frame_queue.full():
            self.frame_queue.get()
        self.frame_queue.put(frame)

    def get_frame(self):
        return self.frame_queue.get(timeout=1)
    def get_latest_frame(self):

            # Using 'timeout' to avoid blocking indefinitely if there's no frame
            frame = self.frame_queue.get(timeout=1)
            self.frame_queue.put(frame)  # Put the frame back for other potential uses
            return frame

    def process_frame(self, numpy_img):
        self.ratio = self.ratio or IMG_WIDTH / numpy_img.shape[1]
        numpy_img_resized = cv2.resize(
            numpy_img, (IMG_WIDTH, int(numpy_img.shape[0] * self.ratio))
        )
        numpy_img_resized = cv2.cvtColor(numpy_img_resized, cv2.COLOR_RGB2BGR)
        sharpened_img = self.sharpen_image(numpy_img_resized)

        # Fine-tuning offsets (you can adjust these)
        x_offset = 7  # Pixels to crop from each side
        y_offset_bottom = 15  # Pixels to crop from the bottom
        y_offset_top = 0  # Pixels to crop from the top

        # Crop and resize
        center_x, center_y = sharpened_img.shape[1] // 2, sharpened_img.shape[0] // 2
        width_to_retain = int(IMG_WIDTH * CROP_FACTOR)
        x1, x2 = (center_x - width_to_retain // 2 + x_offset, center_x + width_to_retain // 2 - x_offset)

        height = sharpened_img.shape[0]
        y1, y2 = int(height * TOP_CROP_FACTOR) + y_offset_top, int(height * (1 - BOTTOM_CROP_FACTOR)) - y_offset_bottom
        cropped_frame = sharpened_img[y1:y2, x1:x2]

        # Determine the circle's center and radius
        circle_center_x, circle_center_y = cropped_frame.shape[1] // 2, cropped_frame.shape[0] // 2
        circle_radius = int(min(cropped_frame.shape[0] // 2, cropped_frame.shape[1] // 2) * CIRCLE_SCALE)
        # Crop the square region that includes the circle
        x1_rect = max(circle_center_x - circle_radius, 0)
        x2_rect = min(circle_center_x + circle_radius, cropped_frame.shape[1])
        y1_rect = max(circle_center_y - circle_radius, 0)
        y2_rect = min(circle_center_y + circle_radius, cropped_frame.shape[0])

        # Crop the circular region
        square_region = cropped_frame[y1_rect:y2_rect, x1_rect:x2_rect]

        # Resize the square region to 512x512
        resized_square = cv2.resize(square_region, (512, 512))

        # Create a circular mask and apply to the resized square
        mask = np.zeros((512, 512), np.uint8)
        cv2.circle(mask, (256, 256), 256, 1, -1)  # Create a white circle in the black mask
        masked_resized_square = cv2.bitwise_and(resized_square, resized_square, mask=mask)

        return masked_resized_square

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
        cam.ExposureTime.set(EXPOSURE_TIME)
        cam.Gain.set(GAIN_VALUE)
        cam.stream_on()

        while True:
            raw_image = cam.data_stream[0].get_image()
            rgb_image = raw_image.convert("RGB")
            numpy_img = rgb_image.get_numpy_array()
            if numpy_img is not None:
                processed_frame = self.process_frame(numpy_img)
                self.put_frame(processed_frame)

        cam.stream_off()
        cam.close_device()

    def sharpen_image(self, img):
        blur = cv2.GaussianBlur(img, (0, 0), 1)
        sharpened = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        return sharpened

    def display_video(self, shared_data=None):
        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    continue

                if shared_data is not None:
                    processed_data = shared_data.get("processed_data", None)
                    if processed_data is not None:
                        (
                            frame_resized,
                            centers,
                            angle,
                            pred_bgr,
                            orientation_labels,
                        ) = processed_data
                        # Code to overlay centers, angle, and orientation labels on the frame
                        for class_idx, center in centers.items():
                            cv2.circle(
                                frame, tuple(map(int, center)), 5, (0, 255, 0), -1
                            )  # Draw circles on frame
                            label_text = f"Label: {orientation_labels[class_idx - 1]}"  # Get the label text
                            label_position = (
                                int(center[0]),
                                int(center[1] - 10),
                            )  # Position the label text above the center
                            cv2.putText(
                                frame,
                                label_text,
                                label_position,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                2,
                            )  # Draw label text on frame

                        if 1 in centers and 2 in centers:
                            pt1, pt2 = tuple(map(int, centers[1])), tuple(map(int, centers[2]))
                            cv2.arrowedLine(
                                frame, pt1, pt2, (0, 0, 255), 2
                            )  # Draw arrow on frame
                            angle_text = f"Orientation angle: {angle:.2f}°"
                            text_position = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                            cv2.putText(
                                frame,
                                angle_text,
                                text_position,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                2,
                            )  # Draw angle text on frame

                self.out = self.out or cv2.VideoWriter(
                    f"output_{datetime.datetime.now():%Y%m%d-%H%M%S}.avi",
                    cv2.VideoWriter_fourcc(*VIDEO_FOURCC),
                    VIDEO_FPS,
                    (frame.shape[1], frame.shape[0]),
                )
                cv2.imshow(VIDEO_FEED_WINDOW_NAME, frame)
                self.out.write(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            cv2.destroyAllWindows()
            if self.out:
                self.out.release()




if __name__ == "__main__":
    cam = Camera()
    capture_thread, display_thread = threading.Thread(
        target=cam.capture_video
    ), threading.Thread(target=cam.display_video)

    capture_thread.start()
    display_thread.start()

    capture_thread.join()
    display_thread.join()
