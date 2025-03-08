import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
# Load the pre-trained model from TensorFlow Hub (cat model)
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2_fpnlite_320x320/1"
model = hub.load(model_url)


def run_inference_for_single_image(model, image):
    image_np = np.asarray(image)  # Convert to numpy array
    # Convert the image to tensor and add batch dimension
    input_tensor = tf.convert_to_tensor(image_np) 
    # Models typically expect input in a batch, even if there's only a single image. This makes the input shape compatible with the model (from (height, width, channels) to (1, height, width, channels)).

    input_tensor = input_tensor[tf.newaxis,...]  # Add batch dimension

    # Run inference
    model_fn = model.signatures['default'] # provides the inference function of the model, which can be used to run the model on input data.

    output_dict = model_fn(input_tensor) # Passes the input tensor through the model's inference function (model_fn). The output is stored in output_dict, which is a dictionary containing the model's predictions.


    # Extract relevant output data
    num_detections = int(output_dict['num_detections'][0])  # Extracts the number of detections from the output dictionary. This value indicates how many objects were detected in the image
    detection_boxes = output_dict['detection_boxes'][0].numpy() 
    # Extracts the number of bounding boxes These boxes are stored as a tensor. 
    # Each box is a list of coordinates representing the top-left and bottom-right corners (normalized between 0 and 1).
    detection_scores = output_dict['detection_scores'][0].numpy().astype(np.float32) # Extracts the confidence scores for each detection. These scores are converted to a NumPy array and cast to float32 for consistency.
    detection_classes = output_dict['detection_classes'][0].numpy().astype(np.int32) 
    # Extracts the class labels (IDs) of the detected objects. These are the model's predictions for what each object is (e.g., "person", "car"). 
    # The class labels are then converted to integers.

    return num_detections, detection_boxes, detection_scores, detection_classes

# Visualize results
def show_inference(image, boxes, classes, scores, threshold=0.5):
    im_width, im_height = image.shape[1], image.shape[0]
    for i in range(len(boxes)):
        # Checks if the confidence score for the current object is greater than the threshold (default is 0.5). This helps filter out weak detections that are not reliable.
        if scores[i] > threshold:
            box = tuple(boxes[i].tolist())
            # Converts the bounding box coordinates into a tuple. 
            # The bounding box is in the form [ymin, xmin, ymax, xmax] (normalized), and it’s converted into a tuple for easy access.
            (left, right, top, bottom) = (box[1] * im_width, box[3] * im_width, box[0] * im_height, box[2] * im_height)
            # Converts the normalized bounding box coordinates back to pixel coordinates by multiplying with the image width and height.
            
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2) 
            # Draws a green rectangle around the detected object on the image using OpenCV. The rectangle is drawn from (left, top) to (right, bottom).
           
            # Line 8: Constructs a label with the class label (e.g., "person", "car") and confidence score for the detected object. The score is formatted to 2 decimal places.

            label = f"Class: {classes[i]}, Score: {scores[i]:.2f}"
            cv2.putText(image, label, (int(left), int(top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #  Displays the label text (class and score) near the top-left corner of the bounding box, slightly above the box. It uses OpenCV's putText function to add the text on the image.

    # Display output image
    plt.figure(figsize=(12,8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# Load webcam feed
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the current frame to RGB for TensorFlow processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run object detection on the frame
    num_detections, detection_boxes, detection_scores, detection_classes = run_inference_for_single_image(model, frame_rgb)

    # Visualize the detections
    show_inference(frame, detection_boxes, detection_classes, detection_scores, threshold=0.5)

    # Display the result
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

video.release()
cv2.destroyAllWindows()
