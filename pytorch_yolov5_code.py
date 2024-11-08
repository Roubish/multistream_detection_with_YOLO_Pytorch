# import cv2
# import torch
# import multiprocessing as mp
# from PIL import Image
# import numpy as np
# import time
# import logging

# # Set up logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("detection_log.txt"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Define video file paths
# video_files = [
#     "/home/ghost/Documents/yolov5_training/yolo_pytorch_tensorflow/test.mp4",
#     "/home/ghost/Documents/yolov5_training/yolo_pytorch_tensorflow/video_output.mp4",
#     "/home/ghost/Documents/yolov5_training/yolo_pytorch_tensorflow/A04_test.mp4",
# ]

# def load_model():
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#     model.eval()
#     return model

# def inference_worker(frame_queue, results_queue):
#     model = load_model()
#     confidence_threshold = 0.45  # Set confidence threshold

#     while True:
#         video_id, frame = frame_queue.get()
#         if frame is None:  # Signal to end process
#             break

#         pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         results = model(pil_image)

#         detections = results.pandas().xyxy[0]
#         filtered_detections = detections[detections['confidence'] >= confidence_threshold]

#         # Log detection information
#         for _, row in filtered_detections.iterrows():
#             logger.info(f"Video {video_id} - Label: {row['name']}, Confidence: {row['confidence']:.2f}, BBox: ({row['xmin']:.0f}, {row['ymin']:.0f}), ({row['xmax']:.0f}, {row['ymax']:.0f})")

#         filtered_frame = frame.copy()
#         for _, row in filtered_detections.iterrows():
#             x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#             label = f"{row['name']} {row['confidence']:.2f}"
#             cv2.rectangle(filtered_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#             cv2.putText(filtered_frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         results_queue.put((video_id, filtered_frame))

# def process_video(video_file, video_id, frame_queue, fps_queue):
#     cap = cv2.VideoCapture(video_file)
#     if not cap.isOpened():
#         logger.error(f"Error: Video {video_id} failed to open. Path: {video_file}")
#         return

#     logger.info(f"Processing video {video_id}: {video_file}")
#     prev_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.info(f"Video {video_id} ended.")
#             break

#         frame = cv2.resize(frame, (640, 360))
#         curr_time = time.time()
#         fps = 1 / (curr_time - prev_time)
#         prev_time = curr_time

#         frame_queue.put((video_id, frame))
#         fps_queue.put((video_id, fps))

#     cap.release()
#     frame_queue.put((video_id, None))

# def display_frames(results_queue, fps_queue, num_videos):
#     frames = [None] * num_videos
#     fps_data = [None] * num_videos

#     while True:
#         if not results_queue.empty():
#             video_id, frame = results_queue.get()
#             frames[video_id - 1] = frame

#         if not fps_queue.empty():
#             video_id, fps = fps_queue.get()
#             fps_data[video_id - 1] = fps

#         if all(f is not None for f in frames) and all(fps is not None for fps in fps_data):
#             min_height = min([f.shape[0] for f in frames])
#             min_width = min([f.shape[1] for f in frames])
#             resized_frames = [cv2.resize(f, (min_width, min_height)) for f in frames]
#             combined_frame = np.hstack(resized_frames)

#             window_size = (1280, 720)
#             combined_frame_resized = cv2.resize(combined_frame, window_size)

#             fps_text = "FPS: " + " | ".join([f"Video {i+1}: {fps_data[i]:.2f}" for i in range(num_videos)])
#             cv2.putText(combined_frame_resized, fps_text, (10, window_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             cv2.imshow("Combined Video Stream", combined_frame_resized)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     frame_queue = mp.Queue(maxsize=10)
#     results_queue = mp.Queue(maxsize=10)
#     fps_queue = mp.Queue(maxsize=10)

#     inference_process = mp.Process(target=inference_worker, args=(frame_queue, results_queue))
#     inference_process.start()

#     processes = []
#     for idx, video_file in enumerate(video_files, start=1):
#         p = mp.Process(target=process_video, args=(video_file, idx, frame_queue, fps_queue))
#         processes.append(p)
#         p.start()

#     display_process = mp.Process(target=display_frames, args=(results_queue, fps_queue, len(video_files)))
#     display_process.start()

#     for p in processes:
#         p.join()

#     for _ in range(len(video_files)):
#         frame_queue.put((None, None))
#     inference_process.join()

#     display_process.join()
#     cv2.destroyAllWindows()



import cv2
import torch
import multiprocessing as mp
from PIL import Image
import numpy as np
import time
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define video file paths
video_files = [
    "/home/ghost/Documents/yolov5_training/yolo_pytorch_tensorflow/test.mp4",
#     "/home/ghost/Documents/yolov5_training/yolo_pytorch_tensorflow/video_output.mp4",
    "/home/ghost/Documents/yolov5_training/yolo_pytorch_tensorflow/A04_test.mp4",
]

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.eval()
    return model

def inference_worker(frame_queue, results_queue):
    model = load_model()
    confidence_threshold = 0.45  # Set confidence threshold

    while True:
        video_id, frame = frame_queue.get()
        if frame is None:  # Signal to end process
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(pil_image)

        detections = results.pandas().xyxy[0]
        filtered_detections = detections[detections['confidence'] >= confidence_threshold]

        # Log detection information
        for _, row in filtered_detections.iterrows():
            logger.info(f"Video {video_id} - Label: {row['name']}, Confidence: {row['confidence']:.2f}, BBox: ({row['xmin']:.0f}, {row['ymin']:.0f}), ({row['xmax']:.0f}, {row['ymax']:.0f})")

        filtered_frame = frame.copy()
        for _, row in filtered_detections.iterrows():
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(filtered_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(filtered_frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        results_queue.put((video_id, filtered_frame))

def process_video(video_file, video_id, frame_queue, fps_queue):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logger.error(f"Skipping video {video_id}: Unable to open video file {video_file}")
        return

    logger.info(f"Processing video {video_id}: {video_file}")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info(f"Video {video_id} ended.")
            break

        frame = cv2.resize(frame, (640, 360))
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        frame_queue.put((video_id, frame))
        fps_queue.put((video_id, fps))

    cap.release()
    frame_queue.put((video_id, None))

def display_frames(results_queue, fps_queue, num_videos):
    frames = [None] * num_videos
    fps_data = [None] * num_videos

    while True:
        if not results_queue.empty():
            video_id, frame = results_queue.get()
            frames[video_id - 1] = frame

        if not fps_queue.empty():
            video_id, fps = fps_queue.get()
            fps_data[video_id - 1] = fps

        if all(f is not None for f in frames) and all(fps is not None for fps in fps_data):
            min_height = min([f.shape[0] for f in frames])
            min_width = min([f.shape[1] for f in frames])
            resized_frames = [cv2.resize(f, (min_width, min_height)) for f in frames]
            combined_frame = np.hstack(resized_frames)

            window_size = (1280, 720)
            combined_frame_resized = cv2.resize(combined_frame, window_size)

            fps_text = "FPS: " + " | ".join([f"Video {i+1}: {fps_data[i]:.2f}" for i in range(num_videos)])
            cv2.putText(combined_frame_resized, fps_text, (10, window_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Combined Video Stream", combined_frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_queue = mp.Queue(maxsize=10)
    results_queue = mp.Queue(maxsize=10)
    fps_queue = mp.Queue(maxsize=10)

    inference_process = mp.Process(target=inference_worker, args=(frame_queue, results_queue))
    inference_process.start()

    processes = []
    for idx, video_file in enumerate(video_files, start=1):
        p = mp.Process(target=process_video, args=(video_file, idx, frame_queue, fps_queue))
        processes.append(p)
        p.start()

    display_process = mp.Process(target=display_frames, args=(results_queue, fps_queue, len(video_files)))
    display_process.start()

    for p in processes:
        p.join()

    for _ in range(len(video_files)):
        frame_queue.put((None, None))
    inference_process.join()

    display_process.join()
    cv2.destroyAllWindows()
