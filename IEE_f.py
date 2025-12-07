import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras_tuner.tuners import RandomSearch
import gc
import tensorflow as tf

# -------- Constants --------
DATASET_PATH = "UrbanSound8K/audio/"
METADATA_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"
NUM_CLASSES = 10
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# -------- Feature extraction functions --------
def augment_audio(y, sr=22050):
    y_noisy = y + 0.005 * np.random.randn(len(y))
    return librosa.effects.pitch_shift(y_noisy, sr=sr, n_steps=2)

def extract_mfcc_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=4.0)
        y_aug = augment_audio(y, sr)
        mfcc = librosa.feature.mfcc(y=y_aug, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error in extract_mfcc_features for {file_path}: {e}")
        return None

def extract_mel_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        target_length = 4 * sr
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        y_aug = augment_audio(y, sr)
        mel = librosa.feature.melspectrogram(y=y_aug, sr=sr, n_mels=128, n_fft=1024, hop_length=678)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape != (128, 128):
            mel_db = np.resize(mel_db, (128, 128))
        return mel_db
    except Exception as e:
        print(f"Error in extract_mel_spectrogram for {file_path}: {e}")
        return None

# -------- Load or extract features --------
if os.path.exists("X_mfcc.npy") and os.path.exists("X_mel.npy") and os.path.exists("y.npy"):
    X_mfcc = np.load("X_mfcc.npy")
    X_mel = np.load("X_mel.npy")
    y = np.load("y.npy")
    print("Loaded features from disk.")
else:
    metadata = pd.read_csv(METADATA_PATH)
    mfcc_features = []
    mel_features = []
    labels = []

    print("Extracting features...")
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing files"):
        file_path = os.path.join(DATASET_PATH, f"fold{row['fold']}", row['slice_file_name'])
        mfcc = extract_mfcc_features(file_path)
        mel = extract_mel_spectrogram(file_path)
        if mfcc is not None and mel is not None:
            mfcc_features.append(mfcc)
            mel_features.append(mel)
            labels.append(row['classID'])

    X_mfcc = np.array(mfcc_features)
    X_mel = np.array(mel_features)
    y = np.array(labels)
    X_mel = np.expand_dims(X_mel, axis=-1)
    print(f"X_mfcc shape: {X_mfcc.shape}, X_mel shape: {X_mel.shape}, y shape: {y.shape}")

    np.save("X_mfcc.npy", X_mfcc)
    np.save("X_mel.npy", X_mel)
    np.save("y.npy", y)
    print("Features saved to disk.")

# -------- Train-test split --------
X_mfcc_train, X_mfcc_test, y_train, y_test = train_test_split(
    X_mfcc, y, test_size=0.2, random_state=42, stratify=y
)
X_mel_train, X_mel_test, _, _ = train_test_split(
    X_mel, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize MFCC features
scaler = StandardScaler()
X_mfcc_train = scaler.fit_transform(X_mfcc_train)
X_mfcc_test = scaler.transform(X_mfcc_test)

# Compute class weights
class_weights = {i: 1.0 / np.sum(y_train == i) for i in range(NUM_CLASSES)}

# -------- MLP Model Tuning --------
print("Starting MLP tuning...")
def build_mlp(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=128, max_value=512, step=128), activation='relu', input_shape=(40,)))
    model.add(Dropout(hp.Float('dropout', 0.2, 0.4, step=0.1)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(hp.Float('dropout', 0.2, 0.4, step=0.1)))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(build_mlp, objective='val_accuracy', max_trials=5, executions_per_trial=1,
                     directory='tuner_mlp', project_name='mlp_tuning')
tuner.search(X_mfcc_train[:1000], y_train[:1000], epochs=10, validation_data=(X_mfcc_test[:250], y_test[:250]),
             class_weight={i: 1.0 / np.sum(y_train[:1000] == i) for i in range(NUM_CLASSES)}, verbose=2)
best_mlp = tuner.get_best_models(num_models=1)[0]
history_mlp = best_mlp.fit(X_mfcc_train, y_train, epochs=30, batch_size=32,
                           validation_data=(X_mfcc_test, y_test), class_weight=class_weights, verbose=2)
print("MLP training complete.")

# -------- MLP Predictions --------
print("Making MLP predictions...")
y_pred_mlp_prob = best_mlp.predict(X_mfcc_test)
y_pred_mlp = np.argmax(y_pred_mlp_prob, axis=1)

# Clear MLP memory
best_mlp = None
tf.keras.backend.clear_session()
gc.collect()

# -------- SVM Tuning --------
print("Starting SVM tuning...")
param_grid = {'C': [1, 10], 'gamma': ['scale', 0.1], 'kernel': ['rbf']}
svm = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1)
svm.fit(X_mfcc_train[:1000], y_train[:1000])  # Tune on subset
print(f"Best SVM params: {svm.best_params_}")
svm = SVC(**svm.best_params_, probability=True, random_state=42)
svm.fit(X_mfcc_train, y_train)
y_pred_svm = svm.predict(X_mfcc_test)
y_pred_svm_prob = svm.predict_proba(X_mfcc_test)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

# -------- Random Forest Tuning --------
print("Starting Random Forest tuning...")
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
rf.fit(X_mfcc_train[:1000], y_train[:1000])  # Tune on subset
print(f"Best RF params: {rf.best_params_}")
rf = RandomForestClassifier(**rf.best_params_, random_state=42)
rf.fit(X_mfcc_train, y_train)
y_pred_rf = rf.predict(X_mfcc_test)
y_pred_rf_prob = rf.predict_proba(X_mfcc_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# -------- Clear MFCC data --------
del X_mfcc, X_mfcc_train, X_mfcc_test
gc.collect()

# -------- CNN Tuning --------
print("Starting CNN tuning...")
def build_cnn(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int('filters1', 32, 64, step=32), (3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(hp.Int('filters2', 64, 128, step=64), (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(hp.Int('filters3', 64, 128, step=64), (3, 3), activation='relu'))  # Additional layer
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout', 0.25, 0.5, step=0.25)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(hp.Float('dropout', 0.25, 0.5, step=0.25)))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(build_cnn, objective='val_accuracy', max_trials=10, executions_per_trial=1,
                     directory='tuner_cnn', project_name='cnn_tuning')
tuner.search(X_mel_train[:2000], y_train[:2000], epochs=20, validation_data=(X_mel_test[:500], y_test[:500]),
             class_weight={i: 1.0 / np.sum(y_train[:2000] == i) for i in range(NUM_CLASSES)}, verbose=2)
best_cnn = tuner.get_best_models(num_models=1)[0]
history_cnn = best_cnn.fit(X_mel_train, y_train, epochs=30, batch_size=32,
                           validation_data=(X_mel_test, y_test), class_weight=class_weights,
                           verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
print("CNN training complete.")

# -------- CNN Predictions --------
y_pred_cnn_prob = best_cnn.predict(X_mel_test)
y_pred_cnn = np.argmax(y_pred_cnn_prob, axis=1)

# Clear CNN model
best_cnn = None
tf.keras.backend.clear_session()
gc.collect()

# -------- Accuracy scores --------
mlp_acc = accuracy_score(y_test, y_pred_mlp)
cnn_acc = accuracy_score(y_test, y_pred_cnn)
svm_acc = accuracy_score(y_test, y_pred_svm)
rf_acc = accuracy_score(y_test, y_pred_rf)

print(f"\n=== Model Accuracies ===")
print(f"MLP Test Accuracy: {mlp_acc:.4f}")
print(f"SVM Test Accuracy: {svm_acc:.4f}")
print(f"Random Forest Test Accuracy: {rf_acc:.4f}")
print(f"CNN Test Accuracy: {cnn_acc:.4f}")

# -------- Ensemble Predictions --------
print("Making ensemble predictions...")
model_accs = {'MLP': mlp_acc, 'SVM': svm_acc, 'RF': rf_acc, 'CNN': cnn_acc}
total_acc = sum(model_accs.values())
weights = [acc / total_acc for acc in model_accs.values()] if total_acc > 0 else [0.25, 0.25, 0.25, 0.25]
print(f"Dynamic ensemble weights: {weights}")
votes = np.array([y_pred_mlp, y_pred_svm, y_pred_rf, y_pred_cnn])
weighted_votes = np.average(votes, axis=0, weights=weights)
y_pred_ensemble = np.round(weighted_votes).astype(int)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")

# -------- Helper functions for plotting --------
def save_plot(fig, filename):
    print(f"Saving plot: {filename}")
    fig.savefig(os.path.join(PLOT_DIR, filename))
    plt.close(fig)

def plot_accuracy_loss(history, model_name):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(history.history['accuracy'], label='train')
    axs[0].plot(history.history['val_accuracy'], label='val')
    axs[0].set_title(f'{model_name} Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(history.history['loss'], label='train')
    axs[1].plot(history.history['val_loss'], label='val')
    axs[1].set_title(f'{model_name} Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    save_plot(fig, f"{model_name}_accuracy_loss.png")

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{model_name} Confusion Matrix')
    save_plot(fig, f"{model_name}_confusion_matrix.png")

def plot_roc_curves(y_true, y_prob, model_name):
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_title(f'ROC Curve - {model_name}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    save_plot(fig, f"{model_name}_roc_curve.png")

def plot_accuracy_comparison(results_dict):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=list(results_dict.keys()), y=list(results_dict.values()), palette='mako', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylabel('Accuracy')
    save_plot(fig, "model_accuracy_comparison.png")

# -------- Plot everything --------
print("Generating plots...")
plot_accuracy_loss(history_mlp, "MLP")
plot_accuracy_loss(history_cnn, "CNN")
plot_confusion_matrix(y_test, y_pred_mlp, "MLP")
plot_confusion_matrix(y_test, y_pred_svm, "SVM")
plot_confusion_matrix(y_test, y_pred_rf, "RandomForest")
plot_confusion_matrix(y_test, y_pred_cnn, "CNN")
plot_roc_curves(y_test, y_pred_mlp_prob, "MLP")
plot_roc_curves(y_test, y_pred_svm_prob, "SVM")
plot_roc_curves(y_test, y_pred_rf_prob, "RandomForest")
plot_roc_curves(y_test, y_pred_cnn_prob, "CNN")
plot_accuracy_comparison({
    "MLP": mlp_acc,
    "SVM": svm_acc,
    "Random Forest": rf_acc,
    "CNN": cnn_acc,
    "Ensemble": ensemble_acc
})

# -------- Classification reports --------
print("\n=== Classification Reports ===")
print("MLP:\n", classification_report(y_test, y_pred_mlp))
print("SVM:\n", classification_report(y_test, y_pred_svm))
print("Random Forest:\n", classification_report(y_test, y_pred_rf))
print("CNN:\n", classification_report(y_test, y_pred_cnn))
print("Ensemble:\n", classification_report(y_test, y_pred_ensemble))

# -------- Final cleanup --------
del X_mel, X_mel_train, X_mel_test, y, y_test, y_train
gc.collect()
tf.keras.backend.clear_session()
print("Final memory cleanup completed.")