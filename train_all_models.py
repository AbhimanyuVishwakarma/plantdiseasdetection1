import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from collections import Counter
import numpy as np
import os
import json
from tensorflow.keras.models import load_model

# Enable memory growth for GPU (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

# Set your dataset paths
TRAIN_DIR = 'PlantdiseaseDetectionApp/PlantVillage'  # Update if needed
VAL_DIR = 'PlantdiseaseDetectionApp/PlantVillage'    # Update if you have a separate val folder

# Model configs
MODEL_CONFIGS = {
    'InceptionV3': {
        'class': InceptionV3,
        'input_size': (299, 299),  # Standard for InceptionV3
        'weights': 'imagenet',
        'filename': 'InceptionV3_plant_disease_model.h5'
    }
}

def train_and_save_model(model_name, config):
    # Always train the model, even if the file exists
    save_path = os.path.join('PlantdiseaseDetectionApp', config['filename'])
    print(f'Training {model_name}...')
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        brightness_range=[0.7,1.3],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    train_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=config['input_size'],
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    val_gen = datagen.flow_from_directory(
        VAL_DIR,
        target_size=config['input_size'],
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=42
    )
    # Print class distribution
    print('Class indices:', train_gen.class_indices)
    # Save class indices to a JSON file for later use in prediction
    indices_path = os.path.join('PlantdiseaseDetectionApp', 'class_indices.json')
    with open(indices_path, 'w') as f:
        json.dump(train_gen.class_indices, f)
    labels = train_gen.classes
    class_counts = Counter(labels)
    print('Class distribution:', class_counts)
    # Compute class weights
    class_weights = {}
    total = float(sum(class_counts.values()))
    for k in class_counts:
        class_weights[k] = total / (len(class_counts) * class_counts[k])
    print('Class weights:', class_weights)
    base_model = config['class'](weights=config['weights'], include_top=False, input_shape=(*config['input_size'], 3))
    # Unfreeze all layers for full fine-tuning
    for layer in base_model.layers:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(train_gen.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint(
        filepath=f'PlantdiseaseDetectionApp/best_model_{model_name}.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,  # Reduced to 10 epochs as requested
        callbacks=[early_stop, reduce_lr, checkpoint],
        class_weight=class_weights
    )
    save_path = os.path.join('PlantdiseaseDetectionApp', config['filename'])
    model.save(save_path)
    print(f'Saved {model_name} to {save_path}')

def resume_training(model_name, config, epochs=25):
    print(f'Resuming training for {model_name}...')
    # Prepare data generators
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        brightness_range=[0.7,1.3],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    train_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=config['input_size'],
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    val_gen = datagen.flow_from_directory(
        VAL_DIR,
        target_size=config['input_size'],
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=42
    )
    labels = train_gen.classes
    class_counts = Counter(labels)
    class_weights = {}
    total = float(sum(class_counts.values()))
    for k in class_counts:
        class_weights[k] = total / (len(class_counts) * class_counts[k])
    # Load the best checkpoint
    checkpoint_path = f'PlantdiseaseDetectionApp/best_model_{model_name}.h5'
    model = load_model(checkpoint_path)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr, checkpoint],
        class_weight=class_weights
    )
    save_path = os.path.join('PlantdiseaseDetectionApp', config['filename'])
    model.save(save_path)
    print(f'Resumed and saved {model_name} to {save_path}')

if __name__ == '__main__':
    # Train only InceptionV3
    train_and_save_model('InceptionV3', MODEL_CONFIGS['InceptionV3'])

# To resume training, load the best checkpoint file for the model:
# from tensorflow.keras.models import load_model
# model = load_model(f'PlantdiseaseDetectionApp/best_model_{model_name}.h5')
# Then call model.fit(...) again as above.
