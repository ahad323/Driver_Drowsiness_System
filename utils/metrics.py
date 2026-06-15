import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

MODEL_PATH = "models/eye_model.h5"
TEST_DIR = "data/eyes/test"

model = load_model(MODEL_PATH)

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

y_prob = model.predict(test_data)
y_pred = (y_prob > 0.5).astype(int).reshape(-1)
y_true = test_data.classes

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("AUC:", roc_auc_score(y_true, y_prob))

print(classification_report(y_true, y_pred))