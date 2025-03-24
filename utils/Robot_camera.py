import threading
from camera.camera import Camera
from utils.predict_image import ResultVisualizer
import torch


class FrameProcessor:
    def __init__(self, camera, result_visualizer, shared_data):
        self.camera = camera
        self.result_visualizer = result_visualizer
        self.shared_data = shared_data
        self.running = True

    def process_frames(self):
        while self.running:
            frame = self.camera.get_frame()
            if frame is not None:
                processed_data = self.result_visualizer.process_frame( frame, self.result_visualizer.device, self.result_visualizer.color_map)
                self.shared_data['processed_data'] = processed_data

    def stop(self):
        self.running = False

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('Multiclass_2_model_12_13_23.pth')

    camera = Camera()
    result_visualizer = ResultVisualizer(model, device)

    shared_data = {}
    frame_processor = FrameProcessor(camera, result_visualizer, shared_data)

    # Start the camera threads
    capture_thread, display_thread = threading.Thread(target=camera.capture_video), threading.Thread(target=lambda: camera.display_video(shared_data))
    capture_thread.start()
    display_thread.start()

    # Start the frame processor thread
    frame_processor_thread = threading.Thread(target=frame_processor.process_frames)
    frame_processor_thread.start()

    try:
        while True:
            pass  # Keep the program running
    except KeyboardInterrupt:
        frame_processor.stop()  # Stop the frame processor when the user interrupts the program

    # Wait for all threads to finish
    capture_thread.join()
    display_thread.join()
    frame_processor_thread.join()

if __name__ == "__main__":
    main()