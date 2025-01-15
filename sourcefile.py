import os
import cv2
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Activation, Dropout

# Constants
IMG_SIZE = 100
DATADIR = r'Data'  # Root data directory
CATEGORIES = os.listdir(DATADIR)

# Initialize data lists
x, y = [], []

# Preprocessing function
def preprocess():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_index = CATEGORIES.index(category)
        print(f"Processing {path}")
        for imgs in tqdm(os.listdir(path)):
            img_arr = cv2.imread(os.path.join(path, imgs))
            resized_array = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
            resized_array = resized_array / 255.0
            x.append(resized_array)
            y.append(class_index)

# Call the preprocessing function
preprocess()

# Destroy OpenCV windows
cv2.destroyAllWindows()

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 3),
    np.array(y),
    test_size=0.20,
    random_state=42
)

# Model configuration
BATCH_SIZE = 32
EPOCHS = 15

# Model creation
model = Sequential([
    Conv2D(64, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3)),
    Activation('relu'),
    Dropout(0.25),

    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.25),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(CATEGORIES)),
    Activation('softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
t1 = time.time()
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.3, verbose=1)
t2 = time.time()
print(f"Training completed in {t2 - t1:.2f} seconds")

# Save the trained model
model.save("face_mask_detector.h5")

# Evaluate the model
print("Model evaluation:")
validation_loss, validation_accuracy = model.evaluate(X_test, y_test)
print(f"Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}")

# Detection function
def get_detection(frame):
    try:
        height, width, channel = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detection.process(img_rgb)
        for detection in result.detections:
            box = detection.location_data.relative_bounding_box
            x, y, w, h = int(box.xmin * width), int(box.ymin * height), int(box.width * width), int(box.height * height)
            return x, y, w, h
    except:
        pass
    return None, None, None, None

# Categories
CATEGORIES = ['no_mask', 'mask']

# Real-time detection
cap = cv2.VideoCapture(0)
c = 0

while True:
    _, frame = cap.read()
    img = frame.copy()
    try:
        x, y, w, h = get_detection(frame)
        if x is not None:
            crop_img = img[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
            crop_img = np.expand_dims(crop_img, axis=0)

            # Get prediction from the model
            prediction = model.predict(crop_img)
            index = np.argmax(prediction)
            res = CATEGORIES[index]

            color = (0, 0, 255) if res == 'no_mask' else (0, 255, 0)
            c = c + 1 if res == 'no_mask' else 0
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, res, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            if c > 50:
                os.startfile("F:\\m1.ogg")
                c = 0
    except:
        pass

    cv2.imshow("Face Mask Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
