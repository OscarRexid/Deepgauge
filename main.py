import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def draw_boxes(image, boxes, class_names, confidence_threshold=0.5):
    im_height, im_width = image.shape[:2]
    for box in boxes:
        ymin, xmin, ymax, xmax = box[:4]
        class_score = box[4]
        if class_score > confidence_threshold:
            (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                          int(ymin * im_height), int(ymax * im_height))
            class_id = int(box[5])
            class_name = class_names[class_id]
            color = (0, 255, 0)  # Green color for the bounding box
            thickness = 2
            cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
            label = f"{class_name}: {class_score:.2f}"
            cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return image

def crop_object_class(image, boxes, class_id, class_names, confidence_threshold=0.5):
    for i in range(len(boxes)):
        box = boxes[i]
        class_score = box[4]
        if class_score > confidence_threshold and int(box[5]) == class_id:
            ymin, xmin, ymax, xmax = box[:4]
            im_height, im_width = image.shape[:2]
            (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                          int(ymin * im_height), int(ymax * im_height))
            cropped_image = image[top:bottom, left:right]
            class_name = class_names[int(box[5])]
            return cropped_image, class_name
    return None, None¨
    
#Find the cordinates of boxes(dial,min,max,center)
def find_cordinates(image,boxes,class_id, class_names,confidence_threshold=0.5)
    for i in range(len(boxes)):
        box = boxes[i]
        class_score = box[4]
        if class_score > confidence_threshold and int(box[5]) == class_id:
            ymin, xmin, ymax, xmax = box[:4]
            im_height, im_width = image.shape[:2]
            (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                          int(ymin * im_height), int(ymax * im_height))
            cropped_image = image[top:bottom, left:right]
            class_name = class_names[int(box[5])]
            mid_x = left + (left-right)/2
            mid_y = bottom + (top-bottom)/2
            return (mid_x,mid_y), class_name
    return None, None

#Find angle between vectors
def vector_angle(vector_1,vector_2)
    return np.arccos((np.dot(vector_1,vector_2))/(abs(vector_1)*abs(vector_2)))
    
def find_all_parts(image,model)
    detections = model.predict(image)
    
    #Ids to the different parts we are looking for
    center_id = 0
    dial_id = 0
    min_id = 0
    max_id = 0
    
    image_with_boxes = draw_boxes(image.copy(), detections[0],class_names)
    center_cordinates, class_center = find_cordinates(image,detections[0],center_id,class_names)
    dial_cordinates, class_dial = find_cordinates(image,detections[0],dial_id,class_names)
    min_cordinates,class_min = find_cordinates(image,detections[0],min_id,class_names)
    max_cordinates,class_max = find_cordinates(image,detections[0],max_id,class_names)
    
    if class_center is not None and class_center is not None and class_min is not None and class_max is not None
        center_to_dial = (dial_cordinates(0) - center_cordinates(0), dial_cordinates(1) - center_cordinates(1))
        center_to_min = (min_cordinates(0) - center_cordinates(0), min_cordinates(1) - center_cordinates(1))
        center_to_max = (max_cordinates(0) - center_cordinates(0), max_cordinates(1) - center_cordinates(1))
        
    # Display the image with all the detected boxes
    cv2.imshow('Detected Objects', cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def main():
    model_path = 'path_to_your_saved_model.h5'
    test_image_path = 'path_to_your_test_image.jpg'
    class_names = ['class_0', 'class_1', 'class_2', 'class_3']  # Add your class names here

    # Load the model
    model = load_model(model_path)

    # Load and preprocess the test image
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image_tensor = tf.expand_dims(image, axis=0)

    # Perform object detection
    detections = model.predict(image_tensor)

    # Draw bounding boxes on the test image
    image_with_boxes = draw_boxes(image.copy(), detections[0], class_names)

    # Display the image with all the detected boxes
    cv2.imshow('Detected Objects', cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    # Extract the detected class and crop the image
    class_id_to_find = 2  # Replace this with the specific class ID you want to find
    cropped_image, class_name = crop_object_class(image, detections[0], class_id_to_find, class_names)

    if cropped_image is not None:
        # Display the cropped image and class name
        cv2.imshow('Cropped Image', cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        print(f"Detected Class: {class_name}")
        print(f"Middle Point Coordinates: {middle_point}")
        print()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Specified class not found in the image.")

if __name__ == '__main__':
    main()
