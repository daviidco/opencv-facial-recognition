import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import threading
import pickle
from mtcnn import MTCNN

GREEN_COLOR = (144, 238, 144)
RED_COLOR = (0, 0, 255)


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (100, 100))  # Redimensionar a un tamaño fijo
            flattened = resized.flatten()  # Aplanar la imagen
            images.append(flattened)
            labels.append(os.path.basename(folder))
        else:
            print(f"No se pudo cargar la imagen: {img_path}")
    return images, labels

def train_or_load_model(X, y):
    model_path = 'models/last_model_generated_mtcnn.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Modelo cargado desde archivo.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo: {accuracy:.2f}")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print("Modelo entrenado y guardado.")
    return model


def process_frame(frame, face_detector, model, le):
    faces = face_detector.detect_faces(frame)
    for face in faces:
        x, y, w, h = face['box']
        face_roi = frame[y:y + h, x:x + w]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(face_roi, (100, 100))
        face_encoding = face_roi.flatten()
        probabilities = model.predict_proba([face_encoding])[0]
        max_prob = np.max(probabilities)
        if max_prob > 0.7:
            label = model.predict([face_encoding])[0]
            label_name = le.inverse_transform([label])[0]
            color = GREEN_COLOR
        else:
            label_name = "Desconocido"
            color = RED_COLOR
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label_name} ({max_prob:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


# Carga de datos y entrenamiento del modelo
data_folder = '../data/faces'
X, y = [], []
print(f"Buscando carpetas en: {os.path.abspath(data_folder)}")
for person_folder in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, person_folder)
    if os.path.isdir(folder_path):
        images, labels = load_images_from_folder(folder_path)
        X.extend(images)
        y.extend(labels)

if len(X) == 0:
    print("No se cargaron imágenes. Verifica las rutas y el contenido de las carpetas.")
    exit()

X = np.array(X)
le = LabelEncoder()
y = le.fit_transform(y)

model = train_or_load_model(X, y)

# Inicialización de la cámara y el detector facial
cap = cv2.VideoCapture(0)
face_detector = MTCNN()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar el frame en un hilo separado
    thread = threading.Thread(target=process_frame, args=(frame, face_detector, model, le))
    thread.start()
    thread.join()

    cv2.imshow('Reconocimiento Facial', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()