import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# ======================
MODEL_PATH = "models/eye_model.h5"
TEST_DIR = "data/eyes/test"
SAVE_DIR = "results/eyes/"
# ======================

os.makedirs(SAVE_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ======================
# PREDICTIONS
# ======================
y_true = test_data.classes
y_prob = model.predict(test_data)
y_pred = (y_prob > 0.5).astype(int).reshape(-1)

# ======================
# METRICS
# ======================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

print("\n===== METRICS =====")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("AUC      :", auc)

# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_true, y_pred)

class_names = list(test_data.class_indices.keys())

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(SAVE_DIR + "confusion_matrix.png")
plt.close()
# ======================
# ROC CURVE
# ======================
fpr, tpr, _ = roc_curve(y_true, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1],[0,1],"--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(SAVE_DIR + "roc_curve.png")
plt.close()

print("Evaluation completed ✅ All metrics + graphs saved.")