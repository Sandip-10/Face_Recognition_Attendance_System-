import cv2
import csv
import os
import numpy as np
from datetime import datetime

# Load the pre-trained face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognizer.yml')

# Load CSV file containing student information
csv_dir = 'csv_files_dir'
csv_file = os.path.join(csv_dir, 'reg.csv')
attendance_dir = csv_dir
student_data = {}

if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        student_id = row[0]
        student_name = row[1]
        department = row[2]
        student_data[student_id] = {"name": student_name, "department": department}

def mark_attendance(student_id, date, time):
    attendance_csv = os.path.join(attendance_dir, f'attendance_{date}.csv')
    headers = ["ID", "Student Name", "Department", "Attendance", "Date", "Time"]

    if not os.path.exists(attendance_csv):
        with open(attendance_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    
    with open(attendance_csv, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            if row[0] == student_id:
                return "Attendance Marked Already"
    
    with open(attendance_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([student_id, student_data[student_id]["name"], student_data[student_id]["department"], 'Present', date, time])
    return "Attendance Marked"

def recognize_faces():
    cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face_roi)

            if confidence < 50:
                student_id = str(label)
                student_name = student_data[student_id]["name"]
                department = student_data[student_id]["department"]
                date = datetime.now().strftime("%Y-%m-%d")
                time = datetime.now().strftime("%H:%M:%S")
                
                attendance_status = mark_attendance(student_id, date, time)
                
                if attendance_status == "Attendance Marked Already":
                    status_text = attendance_status
                    status_color = (0, 0, 255)
                else:
                    status_text = "Attendance Marked"
                    status_color = (0, 255, 0)
                
                # Draw rectangle around face and display student info
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {int(confidence)}%", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Create a white space beside the frame to display student details and attendance status
        height, width, _ = frame.shape
        info_frame = 255 * np.ones(shape=[height, 600, 3], dtype=np.uint8)
        
        if 'student_id' in locals():
            # Display student details on the info frame
            cv2.putText(info_frame, "Student Details", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(info_frame, f"ID: {student_id}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            cv2.putText(info_frame, f"Name: {student_name}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            cv2.putText(info_frame, f"Department: {department}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            cv2.putText(info_frame, f"Status: {status_text}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 1)

        combined_frame = np.hstack((frame, info_frame))
        cv2.imshow('Attendance System', combined_frame)

        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
