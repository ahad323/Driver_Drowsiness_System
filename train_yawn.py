import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 📁 Paths
train_dir = "data/yawn/train"
val_dir   = "data/yawn/val"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 🔄 Data preprocessing
train_gen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# 🧠 Model
base_model = MobileNetV2(
    input_shape=(64,64,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# ⚙️ Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 🏋️ Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# 💾 Save model
model.save("models/yawn_model.h5")

# 📊 SAVE CURVES
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Yawn Loss Curve')
plt.legend()
plt.savefig("results/yawn_loss_curve.png")
plt.close()

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Yawn Accuracy Curve')
plt.legend()
plt.savefig("results/yawn_accuracy_curve.png")
plt.close()

print("Yawn model trained + graphs saved ✅")