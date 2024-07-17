# Face Recognition and Blink Detection System
## Overview 🌐
This project implements a face recognition system with different features ways 
(ear, blink, haarcascade, mtcnn) detection using Python. It utilizes OpenCV, dlib, 
and scikit-learn to create a robust facial recognition model.

## Features 🌟

- Face detection and recognition using multiple methods:
  - Haarcascade for efficient processing
  - MTCNN for higher accuracy
- Liveness detection to prevent photo spoofing:
  - Blink detection using custom algorithms
  - Eye Aspect Ratio (EAR) method for precise eye state analysis
- Real-time processing of video feed
- Support for handling multiple faces (in advanced versions)
- Model training and saving capabilities for personalized recognition
- User-friendly output with visual indicators:
  - Color-coded bounding boxes
  - On-screen labels and confidence scores
- Modular code structure for easy customization and expansion
- Ability to handle various lighting conditions and face orientations 
(especially with MTCNN)

## Requirements 🛠️

- Python 3.12
- OpenCV (opencv-python) 4.10.0
- dlib 19.24.4
- NumPy 1.26.4
- scikit-learn 1.5.1
- TensorFlow 2.17.0
- Keras 3.4.1
- MTCNN 0.1.1
- deepface 0.0.92
- Pillow 10.4.0
- pandas 2.2.2
- scipy 1.14.0

Optional but recommended:
- CUDA-capable GPU for faster processing

Note: Some of these libraries have their own dependencies which will be automatically 
installed when using pip.

## Installation 🚀

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/face-recognition-project.git
   cd face-recognition-project
   ```
   
2. Create and activate a virtual environment:
   ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS and Linux
    source venv/bin/activate
    ```

3. Install the required dependencies:

   ```bash
    pip install -r requirements.txt
    ```
    Note: If you encounter issues installing dlib, you may need to install additional system dependencies. Refer to the dlib documentation for platform-specific instructions.
    Download the shape predictor file:


4. (Optional) If you plan to use MTCNN or DeepFace models, they will be downloaded automatically on first use.

Note: While Poetry is` a great tool for dependency management, it may have issues with dlib installation on some systems. Using pip with a virtual environment as described above is recommended for this project.

## Usage 💡
Before using the face recognition system, it's crucial to set up your data correctly:

1. Create a `data/faces` directory in your project root.
2. Inside `data/faces`, create subdirectories for each person you want to recognize. The name of each subdirectory should be the person's name or identifier.
3. Place multiple high-quality photos of each person in their respective subdirectories:
   - Include at least 5-10 different photos per person for better model training.
   - Use clear, well-lit images with different facial expressions and angles.
   - Ensure the face is clearly visible and not obscured in the photos.
   - Vary the background and lighting conditions for improved model robustness.
   - Photos should be in common formats like .jpg or .png.
Example structure:
```bash
data/
└── faces/
    ├── john_doe/
    │   ├── photo1.jpg
    │   ├── photo2.jpg
    │   └── photo3.jpg
    └── jane_smith/
        ├── image1.jpg
        ├── image2.jpg
        └── image3.jpg
   ```
        
To run the system:

### Running the System

1. Ensure your data is organized as described above.
2. Choose the appropriate version based on your needs:

   a. `face_recognition_haarcascade_basic.py`: Basic implementation using Haarcascade.
   b. `face_recognition_haarcascade_ear.py`: Uses Haarcascade with Eye Aspect Ratio (EAR) for blink detection.
   c. `face_recognition_haarcascade_blink_advanced.py`: Advanced version with improved blink detection and multi-face handling.
   d. `face_recognition_mtcnn.py`: Implements facial recognition using MTCNN for higher accuracy.

3. Run the chosen script:
   ```bash
   python <chosen_script_name>.py

4. The system will either load an existing model or train a new one if no model exists.
5. Once running, the system will use your webcam to detect and recognize faces in real-time.

**Note:** We recommend starting with face_recognition_haarcascade_blink_advanced.py for a good balance of features, accuracy, and performance. It offers advanced capabilities like improved blink detection and multi-face handling, making it suitable for a wide range of applications.
### Version Selection Guide

- Use face_recognition_haarcascade_basic.py for simple projects or systems with limited resources.
- Choose face_recognition_haarcascade_ear.py when basic liveness detection is needed.
- Opt for face_recognition_haarcascade_blink_advanced.py for high security and accuracy in liveness detection, especially with multiple faces.
- Select face_recognition_mtcnn.py when the highest accuracy in face detection is required, particularly in challenging conditions.

The choice depends on your project's specific requirements, available computational resources, and required level of security.

## How it Works 🧠

The face recognition system employs various techniques depending on the version used:

1. **Face Detection**:
   - Haarcascade: Fast and lightweight, suitable for basic scenarios.
   - MTCNN: More accurate, especially in challenging conditions.

2. **Feature Extraction**:
   - For Haarcascade versions: Uses facial landmarks to extract key features.
   - For MTCNN: Utilizes deep learning to extract facial features.

3. **Face Recognition**:
   - All versions use a Support Vector Machine (SVM) classifier trained on the extracted features from the user-provided images.

4. **Liveness Detection**:
   - Basic version: Does not include liveness detection.
   - EAR version: Uses Eye Aspect Ratio to detect blinks.
   - Advanced Blink version: Implements a more sophisticated blink detection algorithm.

5. **Real-time Processing**:
   - All versions process video feed in real-time, providing immediate feedback.
   - Recognized individuals are labeled with names and confidence scores.
   - Blink counts (where applicable) are displayed to indicate liveness.

6. **Model Training**:
   - The system automatically trains on the provided image dataset.
   - If a pre-trained model exists, it loads it for faster startup.

The choice of version affects the balance between speed, accuracy, and additional features like liveness detection. The advanced versions offer better security against spoofing attempts, while the basic version prioritizes speed and simplicity.

## Customization 🔧

Adjust the blink_threshold in the code to fine-tune blink detection sensitivity.
Modify the grace_period to change how long the system waits before making predictions.

## Contributing 🤝
Contributions to improve the system are welcome. Please follow the standard fork-and-pull request workflow.

## License 📄
This project is open-source and available under the MIT License.

