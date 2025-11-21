import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# ปิด Warning ที่รกหน้าจอ (OneDNN และ Deprecated warning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ฟังก์ชันสำหรับสร้าง Dataset
def build_dataset(files, labels, img_size, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))

    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (img_size, img_size))
        img = img / 255.0
        return img, label

    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# MAIN BLOCK
# ============================================================
if __name__ == "__main__":

    # ตั้งค่า GPU (ถ้ามี)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # ============================================================
    # 1) COPY DATASET (แก้ปัญหาโฟลเดอร์ว่าง)
    # ============================================================
    original_path = r"C:\Users\rutsa\PycharmProjects\MLproject\dataset"
    new_path = r"C:\Users\rutsa\PycharmProjects\MLproject\new dataset"

    print(f"Checking dataset at: {original_path}")

    # เช็คว่าต้นฉบับมีอยู่จริงไหม
    if not os.path.exists(original_path):
        print(f"\n!!! ERROR: ไม่เจอโฟลเดอร์ต้นฉบับที่ {original_path}")
        print("กรุณาเช็คว่าชื่อโฟลเดอร์ 'dataset' เขียนถูกไหม หรือเอาไฟล์ไปวางไว้ถูกที่ไหม")
        exit()

    # เช็คว่าต้นฉบับมีรูปไหม
    if len(os.listdir(original_path)) == 0:
        print("\n!!! ERROR: โฟลเดอร์ต้นฉบับว่างเปล่า ไม่มีรูปภาพข้างใน")
        exit()

    # เช็คโฟลเดอร์ปลายทาง (new dataset)
    if os.path.exists(new_path):
        # ถ้ามีโฟลเดอร์อยู่แล้ว แต่ข้างในว่างเปล่า (จากการรันผิดพลาดครั้งก่อน) ให้ลบทิ้งก๊อปใหม่
        if len(os.listdir(new_path)) == 0:
            print("Found empty/broken 'new dataset'. Deleting and re-copying...")
            shutil.rmtree(new_path)  # ลบทิ้ง
            shutil.copytree(original_path, new_path)  # ก๊อปใหม่
            print("Recopied dataset successfully.")
        else:
            print("Using existing 'new dataset'.")
    else:
        # ถ้ายังไม่มี ก็ก๊อปใหม่เลย
        shutil.copytree(original_path, new_path)
        print(f"Copied dataset to: {new_path}")

    dataset_path = new_path

    # ============================================================
    # 2) LOAD FILES
    # ============================================================
    # ดึงชื่อโฟลเดอร์คลาส (เช่น dog, cat)
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print("Classes found:", classes)

    if len(classes) == 0:
        print("\n!!! ERROR: ไม่เจอโฟลเดอร์ Class ใน Dataset")
        print(f"สิ่งที่เจอใน {dataset_path} คือ: {os.listdir(dataset_path)}")
        print("โครงสร้างที่ถูกต้องคือ: dataset -> folder_class (เช่น dog) -> image.jpg")
        # ถ้าโครงสร้างผิด (รูปกองรวมกัน) ให้ลองอ่านไฟล์โดยตรง (Optional Logic)
        exit()

    file_paths = []
    labels = []

    print("Loading images...")
    for label, cls in enumerate(classes):
        cls_path = os.path.join(dataset_path, cls)
        # อ่านไฟล์รูปภาพทุกนามสกุล
        images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if len(images) == 0:
            print(f"Warning: Class '{cls}' has no images.")

        file_paths.extend(images)
        labels.extend([label] * len(images))

    file_paths = np.array(file_paths)
    labels = np.array(labels)

    print(f"Total images loaded: {len(file_paths)}")

    if len(file_paths) == 0:
        print("!!! ERROR: ไม่เจอรูปภาพเลย กรุณาเช็คนามสกุลไฟล์")
        exit()

    # ============================================================
    # 3) SPLIT DATA
    # ============================================================
    # ถ้าข้อมูลน้อยเกินไปสำหรับการ Split
    if len(file_paths) < 10:
        print("Warning: ข้อมูลน้อยมาก อาจ Error ตอน Split (ควรมีอย่างน้อย 10 รูป)")

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.30, stratify=labels, random_state=42
    )

    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # ============================================================
    # 4) BUILD DATASET & MODEL
    # ============================================================
    IMG_SIZE = 224
    BATCH_SIZE = 32

    train_ds = build_dataset(train_files, train_labels, IMG_SIZE, BATCH_SIZE, shuffle=True)
    val_ds = build_dataset(val_files, val_labels, IMG_SIZE, BATCH_SIZE)
    test_ds = build_dataset(test_files, test_labels, IMG_SIZE, BATCH_SIZE)

    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ])

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_aug(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # ปรับ Output layer ให้เหมาะกับจำนวนคลาส
    if len(classes) == 2:
        # Binary Classification
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        loss_fn = "binary_crossentropy"
    else:
        # Multi-class Classification
        outputs = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
        loss_fn = "sparse_categorical_crossentropy"

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    print("\nStarting Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15
    )

    # ============================================================
    # 5) EVALUATION
    # ============================================================
    print("\nGenerating Results...")

    # Confusion Matrix
    y_true = []
    y_pred = []

    for images, batch_labels in test_ds:
        preds = model.predict(images, verbose=0)
        if len(classes) == 2:
            preds = (preds > 0.5).astype(int).flatten()
        else:
            preds = np.argmax(preds, axis=1)

        y_true.extend(batch_labels.numpy())
        y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.show()

    # Learning Curve
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.legend()
    plt.show()