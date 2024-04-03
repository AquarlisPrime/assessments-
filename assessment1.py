import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
import os


# Define the VGG model
def vgg_12(input_shape=(125, 125, 3), num_classes=2):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Define the ResNet model
def build_resnet(input_shape=(125, 125, 3), num_classes=2):
    # Load pre-trained ResNet50 model without classification layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Final model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Extract the zip file
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall("extracted_files")

    # Load the Parquet files
    parquet_file_paths = [
       r"D:\download\QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272.test.snappy.parquet",
       r"D:\download\QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540.test.snappy.parquet",
       r"D:\download\QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494.test.snappy.parquet"
    ]
    
    # Assuming each Parquet file contains data for X and y
    # You might need to adjust this part based on your actual data structure
    X_list = []
    y_list = []
    
    for parquet_file_path in parquet_file_paths:
        df = pd.read_parquet(parquet_file_path)
        # Assuming the columns are named 'features' and 'labels'
        X_list.append(df['features'])
        y_list.append(df['labels'])

    # Concatenate the data from all files
    X = pd.concat(X_list, axis=0)
    y = pd.concat(y_list, axis=0)

    return X, y
# Define the file path where your data is stored
file_path = r"D:\download"
# Load and preprocess the data
X, y = load_and_preprocess_data(file_path)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and compile the VGG model
vgg_model = vgg_12()
vgg_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the VGG model
history_vgg = vgg_model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the VGG model
vgg_loss, vgg_accuracy = vgg_model.evaluate(X_val, y_val)
print("VGG Model - Validation Loss:", vgg_loss)
print("VGG Model - Validation Accuracy:", vgg_accuracy)
