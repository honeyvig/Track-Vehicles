# Track-Vehicles
To track the speed of vehicles on highways using Python, the system will need a combination of hardware (camera, radar, or lidar sensor), software (image processing, possibly deep learning models for vehicle detection), and algorithms for vehicle speed estimation.

Here’s a breakdown of the process, including the hardware setup, Python libraries, and code to implement a basic vehicle speed tracking system.
Hardware Setup:

    Camera: A high-quality camera (e.g., a webcam or IP camera) mounted along the highway.
        Recommended Cameras: A high-resolution camera with at least 30 FPS (frames per second) is essential to capture moving vehicles clearly.
        Example: Logitech Brio, or any IP camera that supports high resolution and frame rate.

    Radar/Lidar (Optional): For more accurate speed detection, you can integrate radar or lidar sensors. However, this setup involves more complex hardware and algorithms.
        Example: RADAR sensors like the Navtech Radar or RoboSense Lidar.

    Computer: A computer (could be a server or a Raspberry Pi) where the image processing happens in real-time. This device will run the Python scripts and capture data from the camera.

Libraries & Tools:

    OpenCV: For image processing, vehicle detection, and frame extraction.
    YOLO or MobileNet: For vehicle detection using deep learning.
    NumPy: For numerical operations.
    Time: For measuring time intervals between frames.
    Matplotlib: For visualization (optional).
    TensorFlow/PyTorch: For running pre-trained object detection models (e.g., YOLO).
    Haar Cascades (OpenCV): A simpler method to detect vehicles, though not as accurate as deep learning models.

GitHub Repositories:

    OpenCV Vehicle Tracking: You can use this repository as an example of basic vehicle tracking using OpenCV: Vehicle Detection and Tracking.
    YOLOv4 for Vehicle Detection: An advanced detection model repository for vehicle tracking: YOLOv4 Vehicle Detection.

Python Script to Track Vehicle Speed:

This script uses OpenCV for vehicle detection and calculates the speed based on frame differences.

import cv2
import numpy as np
import time

# Parameters (adjust as necessary)
frame_rate = 30  # Frames per second of the camera
frame_width = 640  # Frame width
frame_height = 480  # Frame height
distance_between_points = 10  # Distance (in meters) between the two detection points on the road

# Load pre-trained vehicle detection model (YOLO, Haar Cascade, etc.)
# For example, we will use YOLO here:
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getLayers() if net.getLayer(i).type == "Detection"]

# Initialize the camera
cap = cv2.VideoCapture(0)  # You can change '0' to the IP camera URL for remote camera

# Initialize the time variable for speed calculation
previous_time = 0
vehicle_speed = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistency (optional)
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Prepare the image for YOLO detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Loop through the detections and draw bounding boxes around vehicles
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Rectangle coordinates
                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maxima suppression to remove duplicate boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Vehicle detection, calculate speed
    if len(indexes) > 0:
        current_time = time.time()
        time_diff = current_time - previous_time  # Time difference between frames

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Speed estimation (simplified based on time difference)
                vehicle_speed = (distance_between_points / time_diff) * frame_rate  # Speed in m/s
                previous_time = current_time

        # Display speed on the image
        cv2.putText(frame, f"Speed: {vehicle_speed:.2f} m/s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the result
    cv2.imshow('Vehicle Speed Tracker', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

Explanation of the Code:

    Camera Input: The script starts by capturing video from the camera using OpenCV's cv2.VideoCapture(). You can replace it with the URL of an IP camera if needed.

    Vehicle Detection: The script uses a pre-trained YOLO model to detect vehicles in each frame. You can also use other methods such as Haar Cascades or MobileNet SSD if desired.

    Speed Calculation: The code calculates the vehicle speed based on the time it takes for the vehicle to travel across a known distance (set by distance_between_points). The formula assumes the vehicle moves across two points in the image, and speed is estimated as distance/time.

    Display: The speed of each vehicle is shown on the frame, and the bounding box for each detected vehicle is drawn.

    Time Calculation: The time.time() method is used to calculate the time difference between successive frames. This time difference is used to estimate the speed of the vehicle based on the distance between two points in the frame.

    Exit: Pressing the q key exits the loop and closes the window.

Next Steps:

    Calibration: You should calibrate the system by determining the real-world distance represented by the pixels in your camera. This will allow you to convert the pixel speed into real-world units like km/h or mph.

    Accuracy: For more precise speed detection, integrating radar or lidar sensors would give more accurate results. These sensors are specifically designed for speed detection, unlike camera-based systems, which might be affected by traffic flow, lighting, or road conditions.

    Vehicle Tracking & Multiple Lane Detection: If tracking multiple vehicles on different lanes is necessary, you may need to use object tracking algorithms such as SORT (Simple Online and Realtime Tracking) or Deep SORT.

Conclusion:

This basic Python code uses a camera to track vehicle speed on highways. With improvements, such as integrating a radar or lidar sensor and calibrating the system properly, the vehicle speed detection system can be enhanced for more reliable and accurate results.
