import csv
import cv2
import os
from datetime import datetime

# Define directories for CSV files and face data
csv_file_dir = 'csv_files_dir'
face_data_dir = 'face_data_dir'

# Create directories if they do not exist
os.makedirs(csv_file_dir, exist_ok=True)
os.makedirs(face_data_dir, exist_ok=True)

# Define the path for the CSV file
csv_file_path = os.path.join(csv_file_dir, 'reg.csv')

# Check if the CSV file exists, if not, create it with headers
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Student Name", "Department", "Year of Admission", "Father Name", "Address", "Phone Number", "Email ID"])

# Input student details
student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")
department = input("Enter Department: ").upper()
year_of_admission = input("Enter Year of Admission (4 digit integer): ")
father_name = input("Enter Father's Name: ")
address = input("Enter Address: ")
phone_number = input("Enter Phone Number (10 digit integer): ")
email_id = input("Enter Email ID: ").lower()

# Append student details to the CSV file
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([student_id, student_name, department, year_of_admission, father_name, address, phone_number, email_id])

# Display student information
print("\n----------------------\n")
print("Student Information:")
print("ID:", student_id)
print("Student Name:", student_name)
print("Department:", department)
print("Year of Admission:", year_of_admission)
print("Father's Name:", father_name)
print("Address:", address)
print("Phone Number:", phone_number)
print("Email ID:", email_id)
print("\nStudent registered successfully.\n")
print("\n----------------------\n")

# Initialize a loop to capture face
while True:
    digit = int(input("Enter 1 to capture face: "))
    if digit == 1:
        print("Face recognition started.")
        break
    else:
        continue

# Initialize the face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to extract faces from an image
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face, (x, y, w, h)
    return None, None

# Start video capture
cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    face, rect = face_extractor(frame)
    if face is not None:
        count += 1
        face = cv2.resize(face, (350, 350))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Define the file name and path for the face image
        file_name_path = os.path.join(face_data_dir, f'{student_id}_{count}.jpg')
        cv2.imwrite(file_name_path, face)

        # Draw a rectangle around the face and display the count
        if rect:
            x, y, w, h = rect
            cv2.putText(frame, f'Count: {count}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Camera Feed', frame)
    else:
        print("Face not found.")
        pass

    # Exit the loop when the 'Enter' key is pressed or the count reaches 100
    if cv2.waitKey(1) == 13 or count == 100:
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
print('Face data collection completed.')