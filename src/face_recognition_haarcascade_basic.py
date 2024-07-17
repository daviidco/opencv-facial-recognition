import cv2
import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Definición de colores
GREEN_COLOR = (144, 238, 144)
RED_COLOR = (0, 0, 255)  # En formato BGR


class FaceRecognizer:
    def __init__(self, data_folder, cascade_path, model_path):
        self.data_folder = data_folder
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.model_path = model_path
        self.model = None
        self.label_encoder = LabelEncoder()

    def load_images_from_folder(self, folder):
        images = []
        labels = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.equalizeHist(img)  # Añadir ecualización del histograma
                resized = cv2.resize(img, (100, 100))
                images.append(resized.flatten())
                labels.append(os.path.basename(folder))
            else:
                print(f"No se pudo cargar la imagen: {img_path}")
        return images, labels

    def prepare_data(self):
        X = []
        y = []
        for person_folder in os.listdir(self.data_folder):
            folder_path = os.path.join(self.data_folder, person_folder)
            if os.path.isdir(folder_path):
                images, labels = self.load_images_from_folder(folder_path)
                X.extend(images)
                y.extend(labels)
        return np.array(X), np.array(y)

    def train_model(self):
        X, y = self.prepare_data()
        if len(X) == 0:
            raise ValueError("No se cargaron imágenes. Verifica las rutas y el contenido de las carpetas.")

        y = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo: {accuracy:.2f}")

        if self.model_path:
            self.save_model()

    def save_model(self):
        with open(self.model_path, 'wb') as f:
            joblib.dump((self.model, self.label_encoder), f)
        print(f"Modelo guardado en {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model, self.label_encoder = joblib.load(f)
            print(f"Modelo cargado desde {self.model_path}")
        else:
            raise FileNotFoundError(f"No se encontró el archivo del modelo: {self.model_path}")

    def predict(self, face_encoding):
        probabilities = self.model.predict_proba([face_encoding])[0]
        max_prob = np.max(probabilities)
        if max_prob > 0.68:
            label = self.model.predict([face_encoding])[0]
            label_name = self.label_encoder.inverse_transform([label])[0]
            return label_name, max_prob, GREEN_COLOR
        return "Desconocido", max_prob, RED_COLOR

    def recognize_faces(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_roi = cv2.resize(gray[y:y + h, x:x + w], (100, 100)).flatten()
                label_name, max_prob, color = self.predict(face_roi)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label_name} ({max_prob:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,
                            2)
            cv2.imshow('Reconocimiento Facial', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    data_folder = '../data/faces'
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    model_generated_path = 'models/last_model_generated.pkl'
    face_recognizer = FaceRecognizer(data_folder, cascade_path, model_generated_path)
    try:
        face_recognizer.load_model()
    except FileNotFoundError:
        face_recognizer.train_model()
    face_recognizer.recognize_faces()
