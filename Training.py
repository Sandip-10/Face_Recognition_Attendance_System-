import cv2
import numpy as np
import os

# Set the path to the dataset containing facial images for training
data_path = 'face_data_dir/'

# Get all file names in the directory and filter out the ones that are not image files
onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

# Initialize empty lists for training data and labels
Training_Data, Labels = [], []

# Loop through each image file, read the image in grayscale, and append it to the Training_Data list
# Append the index of the image file to the Labels list
for i, files in enumerate(onlyfiles):  
    image_path = os.path.join(data_path, files)
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is not None:
        # Check if image reading is successful
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(int(files.split('_')[0]))
    else:
        print(f"Failed to read image: {image_path}")

# Check if any images were read successfully
if len(Training_Data) == 0:
    print("No training data available. Please check the dataset directory.")
    exit()

# Convert the Labels list to a numpy array of type int32
Labels = np.asarray(Labels, dtype=np.int32)

# Create a new LBPH face recognition model
model = cv2.face.LBPHFaceRecognizer_create()

# Train the model using the training data and labels
model.train(np.asarray(Training_Data), np.asarray(Labels))

# Save the trained model
model.save('face_recognizer.yml')

print("Dataset Model Training Completed")
