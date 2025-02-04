
# Automated Attendance System using Face Recognition

This project automates attendance management using face recognition technology, replacing traditional manual methods. It leverages computer vision techniques for real-time face detection and recognition, ensuring a seamless and contactless experience.

## Features
- **Face Detection & Recognition**: Implements Haar Cascade Classifier and LBPH Algorithm for accurate facial recognition.
- **Real-time Attendance**: Captures attendance in real-time through webcam inputs.
- **CSV Data Management**: Stores attendance and registration details in `.csv` files.
- **User-Friendly Interface**: Simple Python-based GUI for registration and attendance marking.

## Tech Stack
- **Programming Language**: Python
- **Libraries**: OpenCV, NumPy, Pandas
- **Tools**: Visual Studio Code, Python 3.10
- **Hardware**: Logitech C270 webcam, Windows/Linux PC

## Installation
1. Install Python 3.10 or later.
2. Install the required libraries using `pip install opencv-python numpy pandas`.
3. Clone this repository:
4. Run the `main_registration.py` for user registration and `main_attendance.py` for marking attendance.

## How It Works
1. **Registration**: Captures images of students and saves them for training.
2. **Training**: Generates the trained model (`face_recognizer.yml`) using LBPH algorithm.
3. **Attendance**: Matches live face input with the trained dataset and logs attendance.

## Project Structure
- `main_registration.py`: Registers student details and captures face data.
- `training.py`: Trains the captured face data.
- `main_attendance.py`: Recognizes faces and logs attendance.
- `face_data_dir/`: Stores captured face images.
- `CSV_files_dir/`: Stores attendance and registration records.

## Future Enhancements
- Mobile application integration.
- Improved accuracy with advanced facial recognition techniques.
- Support for cloud-based data storage.

## Acknowledgment
This project was developed as a prototype for automating attendance systems in schools and organizations.



