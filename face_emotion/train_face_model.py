import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "archive", "train")
TEST_DIR  = os.path.join(BASE_DIR, "archive", "test")  # 最後評估用

IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 80
SEED = 42

MODEL_OUT = os.path.join(BASE_DIR, "face_model_best.h5")
FINAL_OUT = os.path.join(BASE_DIR, "face_model_final.h5")

L2 = 1e-5

# =========================
# Dataset
# =========================
def build_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=True,
        label_mode="int",
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("分類種類:", class_names)

    # ✅ 保守 augmentation（太猛會讓 val/test 抖）
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.06, 0.06),
        layers.RandomContrast(0.08),
    ], name="augmentation")

    norm = layers.Rescaling(1.0 / 255.0)

    def train_map(x, y):
        x = tf.cast(x, tf.float32)
        x = aug(x, training=True)
        x = norm(x)
        return x, y

    def val_map(x, y):
        x = tf.cast(x, tf.float32)
        x = norm(x)
        return x, y

    train_ds = train_ds.map(train_map, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(val_map,   num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes

def compute_class_weights_from_dir(train_dir: str, class_names: list[str]):
    # 掃資料夾計算每類數量
    counts = []
    for c in class_names:
        p = os.path.join(train_dir, c)
        if not os.path.isdir(p):
            counts.append(0)
            continue
        cnt = 0
        for f in os.listdir(p):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                cnt += 1
        counts.append(cnt)

    # 建 y 來算 balanced class_weight
    y = []
    for idx, cnt in enumerate(counts):
        y += [idx] * cnt
    y = np.array(y, dtype=np.int32)

    classes = np.arange(len(class_names), dtype=np.int32)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)

    # ✅ 重要：你的 disgust 太少，balanced 權重會很大 → 讓訓練不穩
    # 做「降溫 + 上限」：既照顧少數類，又不會爆炸
    POWER = 0.5      # 0.5 = sqrt 降溫（推薦）
    MAX_W = 6.0      # 權重上限（推薦 5~8）
    cw = np.power(cw, POWER)
    cw = np.minimum(cw, MAX_W)

    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("class_counts:", {class_names[i]: counts[i] for i in range(len(class_names))})
    print("class_weight (tempered):", class_weight)
    return class_weight

# =========================
# Model (ResNet-like)
# =========================
def res_block(x, filters: int, downsample: bool, l2=L2):
    stride = 2 if downsample else 1

    shortcut = x
    if downsample or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same",
                                 kernel_regularizer=regularizers.l2(l2))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, strides=1, padding="same",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_model(num_classes: int):
    inputs = layers.Input(shape=IMG_SIZE + (1,))

    x = layers.Conv2D(32, 3, padding="same",
                      kernel_regularizer=regularizers.l2(L2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ResNet-ish stages
    x = res_block(x, 32,  downsample=False)
    x = res_block(x, 32,  downsample=False)

    x = res_block(x, 64,  downsample=True)
    x = res_block(x, 64,  downsample=False)

    x = res_block(x, 128, downsample=True)
    x = res_block(x, 128, downsample=False)

    x = res_block(x, 256, downsample=True)
    x = res_block(x, 256, downsample=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="face_emotion_resnet_small")

# =========================
# Train + Eval
# =========================
def main():
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)

    train_ds, val_ds, class_names, num_classes = build_datasets()
    class_weight = compute_class_weights_from_dir(TRAIN_DIR, class_names)

    model = build_model(num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    model.save(FINAL_OUT)
    print("最佳模型已儲存:", MODEL_OUT)
    print("最後模型已儲存:", FINAL_OUT)
    print("class_names（請記下順序）:", class_names)

    # ✅ 用 best model 測 test + 印出每類表現
    if os.path.isdir(TEST_DIR):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            TEST_DIR,
            image_size=IMG_SIZE,
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            shuffle=False,
            label_mode="int",
        )
        test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)

        best_model = tf.keras.models.load_model(MODEL_OUT)
        test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
        print(f"[Best Model Test] acc={test_acc:.4f}, loss={test_loss:.4f}")

        # 收集 y_true / y_pred
        y_true_all, y_pred_all = [], []
        for x, y in test_ds:
            p = best_model.predict(x, verbose=0)
            y_pred_all.extend(np.argmax(p, axis=1))
            y_true_all.extend(y.numpy())

        print("\n=== Classification Report (TEST) ===")
        print(classification_report(y_true_all, y_pred_all, target_names=class_names, digits=4))

        print("\n=== Confusion Matrix (TEST) ===")
        print(confusion_matrix(y_true_all, y_pred_all))

if __name__ == "__main__":
    main()
