import albumentations as A
import cv2
import os
import pandas as pd
import plotly.express as px
from PIL import Image

import logging
import keras
from keras import layers

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import tensorflow as tf

import shutil
from tensorflow.keras.applications import EfficientNetB0, ResNet50, VGG16

import itertools
import numpy as np
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Importing required libraries.")

# Create directories for models and artifacts
os.makedirs("models", exist_ok=True)
os.makedirs("artefacts", exist_ok=True)
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Set hyperparameters
image_size = (100, 100)
batch_size = 128
epochs = 2
patience = 2

# Afficher les paramètres d'entrainement
logging.info(f"Hyperparameters: image_size={image_size}, batch_size={batch_size}, epochs={epochs}, patience={patience}")


def load_data(data_dir, validation_split=0.25, seed=1337, image_size=(100, 100), batch_size=128, label_mode='int'):
    """Load and split the data into training and validation sets."""
    logging.info(f"Loading data from {data_dir}")
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="both",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode
    )
    return train_ds, val_ds

def data_augmentation(data_dir, augmented_data_dir):
    """Apply data augmentation to the training data."""
    logging.info(f"Applying data augmentation to {data_dir}")
    transforms = [
        A.RandomRotate90(p=1.0),
        A.Transpose(p=1.0),
        A.VerticalFlip(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0),
    ]

    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        augmented_class_path = os.path.join(augmented_data_dir, class_dir)

        if not os.path.exists(augmented_class_path):
            os.makedirs(augmented_class_path)

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            original_img_path = os.path.join(augmented_class_path, img_file)
            shutil.copy(img_path, original_img_path)

            for i, transform in enumerate(transforms):
                augmented = transform(image=img)
                augmented_img = augmented['image']

                transformed_img_file = f'transformed_{i}_{img_file}'
                transformed_img_path = os.path.join(augmented_class_path, transformed_img_file)
                cv2.imwrite(transformed_img_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))

def create_cnn_model(num_classes):
    """Create a CNN model."""
    logging.info("Creating CNN model")
    model = keras.Sequential([
        layers.Input(shape=(100, 100, 3), name='input_layer'),
        layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1'),
        layers.MaxPooling2D((2, 2), name='maxpooling2d_1'),
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        layers.MaxPooling2D((2, 2), name='maxpooling2d_2'),
        layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
        layers.MaxPooling2D((2, 2), name='maxpooling2d_3'),
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])
    return model

def create_resnet_model(num_classes):
    """Create a ResNet model."""
    logging.info("Creating ResNet model")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.Dense(num_classes, activation='relu', name='dense_1'),
        layers.Dropout(0.2, name='dropout_1'),
        layers.Dense(num_classes, activation='relu', name='dense_2'),
        layers.GlobalAveragePooling2D(name='global_avg_pooling'),
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])
    return model

def create_efficientnet_model(num_classes):
    """Create an EfficientNet model."""
    logging.info("Creating EfficientNet model")
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(name='global_avg_pooling'),
        layers.Dense(num_classes*3, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(0.2, name='dropout_1'),
        layers.Dense(num_classes*2, activation='relu', name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(0.2, name='dropout_2'),
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])
    return model

def create_vgg16_model(num_classes):
    """Create a VGG16 model."""
    logging.info("Creating VGG16 model")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(name='global_avg_pooling'),
        layers.Dropout(0.3, name='dropout_1'),
        layers.Dense(512, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(0.5, name='dropout_2'),
        layers.Dense(256, activation='relu', name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(0.4, name='dropout_3'),
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])
    return model

def compile_model(model):
    """Compile the model."""
    logging.info("Compiling model")
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model

def train_model(model, train_ds, val_ds, model_name, epochs=2, patience=2):
    """Train the model and save the best model and training log."""
    logging.info(f"Training {model_name} model")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f"models/best_model_{model_name}.keras", save_best_only=True, monitor="val_acc", mode="max"
        ),
        keras.callbacks.EarlyStopping(monitor='val_acc', patience=patience, mode="max", restore_best_weights=True),
        keras.callbacks.CSVLogger(f'artefacts/training_log_{model_name}.csv')
    ]

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )
    return history

def plot_training_history(history, model_name):
    """Plot training and validation accuracy and loss values."""
    logging.info(f"Plotting training history for {model_name} model")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(f'{model_name} Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.savefig(f'artefacts/training_history_{model_name}.png')

def plot_confusion_matrix(cm, class_names, title='Confusion matrix', cmap=plt.cm.Blues, file_name='confusion_matrix.png'):
    """Plot the confusion matrix."""
    logging.info(f"Plotting confusion matrix for {title}")
    plt.figure(figsize=(200, 200))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]} ({100 * cm[i, j] / cm[i, :].sum():.2f}%)',
                 horizontalalignment="center", verticalalignment='bottom',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_name)

def evaluate_model(model, val_ds, model_name):
    """Evaluate the model and plot the confusion matrix."""
    logging.info(f"Evaluating {model_name} model")
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        y_true.extend(labels.numpy())
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, val_ds.class_names, title=f'{model_name} Confusion Matrix', file_name=f'artefacts/confusion_matrix_{model_name}.png')

    report = classification_report(y_true, y_pred, target_names=val_ds.class_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'artefacts/metrics_{model_name}.csv')
    
# Zip the directories where artifacts and models are saved
def zip_directory(directory_path, zip_path):
    """Zip the contents of a directory."""
    logging.info(f"Zipping directory {directory_path} to {zip_path}")
    shutil.make_archive(zip_path, 'zip', directory_path)

# Load data
train_ds, val_ds = load_data("data/Training")
num_classes = len(train_ds.class_names)

# Data augmentation
data_augmentation("data/Training", "data/train-augmented")
train_ds, val_ds = load_data("data/train-augmented")

# Create and compile models
cnn = create_cnn_model(num_classes)
cnn = compile_model(cnn)

resnet = create_resnet_model(num_classes)
resnet = compile_model(resnet)

efficent_net = create_efficientnet_model(num_classes)
efficent_net = compile_model(efficent_net)

vgg16 = create_vgg16_model(num_classes)
vgg16 = compile_model(vgg16)

# Train models
history_cnn = train_model(cnn, train_ds, val_ds, "cnn", epochs=epochs)
history_resnet = train_model(resnet, train_ds, val_ds, "resnet", epochs=epochs)
history_efficent_net = train_model(efficent_net, train_ds, val_ds, "efficent_net", epochs=epochs)
history_vgg16 = train_model(vgg16, train_ds, val_ds, "vgg16", epochs=epochs)

# Plot training history
plot_training_history(history_cnn, "CNN")
plot_training_history(history_resnet, "ResNet")
plot_training_history(history_efficent_net, "EfficientNet")
plot_training_history(history_vgg16, "VGG16")

# Evaluate models
evaluate_model(cnn, val_ds, "cnn")
evaluate_model(resnet, val_ds, "resnet")
evaluate_model(efficent_net, val_ds, "efficent_net")
evaluate_model(vgg16, val_ds, "vgg16")




# Zip the 'models' and 'artefacts' directories
zip_directory('models', 'models')
zip_directory('artefacts', 'artefacts')