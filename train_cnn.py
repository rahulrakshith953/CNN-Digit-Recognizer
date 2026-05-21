# ------------------------------------------------------------
# train_cnn_commented.py
#
# This script trains a Convolutional Neural Network (CNN) for
# handwritten digit recognition using the Kaggle Digit Recognizer
# dataset.
#
# Expected input file:
#   train.csv
#
# Expected CSV format:
#   - One column named "label" containing the digit class: 0 to 9
#   - 784 pixel columns representing a 28 x 28 grayscale image
#
# Output files:
#   - digit_cnn.keras            -> trained CNN model
#   - training_history.json      -> training accuracy/loss history
# ------------------------------------------------------------

# argparse is used to accept command-line inputs such as file paths,
# number of epochs, batch size, etc.
import argparse

# json is used to save the training history in a readable JSON file.
import json

# Path helps us handle file and folder paths safely across operating systems.
from pathlib import Path

# NumPy is used for numerical operations, especially arrays.
import numpy as np

# Pandas is used to read and process the CSV dataset.
import pandas as pd

# train_test_split is used to divide the dataset into training and validation data.
from sklearn.model_selection import train_test_split

# EarlyStopping stops training when validation loss stops improving.
# ReduceLROnPlateau reduces the learning rate when improvement slows down.
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# CNN layers used to build the neural network.
# Conv2D extracts image features.
# MaxPooling2D reduces image size while keeping important features.
# Dropout reduces overfitting.
# Flatten converts 2D feature maps into a 1D vector.
# Dense creates fully connected neural network layers.
# BatchNormalization stabilizes and speeds up training.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

# Sequential is used because the model is built layer by layer.
from tensorflow.keras.models import Sequential

# Adam is the optimizer used to update model weights during training.
from tensorflow.keras.optimizers import Adam

# to_categorical converts labels like 3 into one-hot format like [0,0,0,1,0,0,0,0,0,0].
from tensorflow.keras.utils import to_categorical


def load_kaggle_train_csv(csv_path: str, validation_size: float = 0.1, random_state: int = 42):
    """Load Kaggle Digit Recognizer train.csv and create train/validation splits.

    Parameters:
        csv_path:
            Path to the Kaggle train.csv file.

        validation_size:
            Fraction of data to use for validation.
            Example: 0.1 means 10% validation and 90% training.

        random_state:
            Fixed random seed so that the train/validation split is reproducible.

    Expected format:
        - One column named 'label'
        - 784 pixel columns for 28 x 28 grayscale images

    Returns:
        X_train, X_val, y_train, y_val
    """

    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(csv_path)

    # Check whether the dataset contains the required label column.
    # Without this column, the model will not know the correct answer for each image.
    if "label" not in df.columns:
        raise ValueError(
            "Expected a 'label' column in the training CSV. "
            "Use Kaggle Digit Recognizer train.csv for this script."
        )

    # Store the digit labels separately.
    # Example labels: 0, 1, 2, 3, ..., 9
    y = df["label"].values

    # Remove the label column and keep only pixel values.
    # Pixel values are originally between 0 and 255.
    # Dividing by 255.0 normalizes them to the range 0 to 1.
    X = df.drop(columns=["label"]).values.astype("float32") / 255.0

    # Reshape the flat 784 pixel values into 28 x 28 images with 1 channel.
    # CNN expects image input in the shape: (samples, height, width, channels).
    # Here, channels = 1 because the images are grayscale.
    X = X.reshape(-1, 28, 28, 1)

    # Convert numeric labels into one-hot encoded labels.
    # Example: label 7 becomes [0,0,0,0,0,0,0,1,0,0].
    y = to_categorical(y, num_classes=10)

    # Split the dataset into training and validation sets.
    # stratify keeps the digit distribution balanced in both sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=np.argmax(y, axis=1),
    )

    # Return the prepared training and validation data.
    return X_train, X_val, y_train, y_val


def build_model(input_shape=(28, 28, 1), num_classes=10):
    """Build and compile the CNN model.

    Parameters:
        input_shape:
            Shape of each input image.
            For MNIST/Kaggle Digit Recognizer, it is 28 x 28 x 1.

        num_classes:
            Number of output classes.
            Since digits are from 0 to 9, there are 10 classes.

    Returns:
        A compiled Keras CNN model.
    """

    # Sequential model means layers are added one after another.
    model = Sequential([
        # First convolution layer extracts basic features such as edges and curves.
        # 32 filters means the layer learns 32 different feature detectors.
        # padding="same" keeps the image size unchanged after convolution.
        Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=input_shape),

        # Batch normalization helps stabilize training and can improve speed/performance.
        BatchNormalization(),

        # Second convolution layer learns more detailed patterns from the previous layer.
        Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),

        # Max pooling reduces the image size by taking the maximum value from each 2 x 2 region.
        # This reduces computation and helps the model focus on important features.
        MaxPooling2D(pool_size=(2, 2)),

        # Dropout randomly turns off 25% of neurons during training.
        # This helps prevent overfitting.
        Dropout(0.25),

        # Third convolution layer uses 64 filters to learn deeper and more complex features.
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),

        # Normalize activations again for stable training.
        BatchNormalization(),

        # Fourth convolution layer further improves feature extraction.
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),

        # Another pooling layer reduces the feature map size.
        MaxPooling2D(pool_size=(2, 2)),

        # Dropout again reduces overfitting after the second CNN block.
        Dropout(0.25),

        # Flatten converts the 2D feature maps into a 1D vector.
        # Dense layers need 1D input.
        Flatten(),

        # Fully connected layer that learns final decision-making patterns.
        Dense(128, activation="relu"),

        # Higher dropout before final classification to reduce overfitting.
        Dropout(0.5),

        # Final output layer.
        # softmax gives probabilities for each digit class from 0 to 9.
        Dense(num_classes, activation="softmax"),
    ])

    # Compile the model before training.
    # Adam is a commonly used optimizer for deep learning.
    # categorical_crossentropy is used because labels are one-hot encoded.
    # accuracy is used as the performance metric.
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Return the compiled model.
    return model


def main():
    """Main function that controls the full training pipeline."""

    # Create an argument parser so the script can accept command-line options.
    parser = argparse.ArgumentParser(description="Train a CNN on Kaggle Digit Recognizer data.")

    # Path to the training CSV file.
    # Default assumes train.csv is in the same folder where the script is run.
    parser.add_argument("--train_csv", type=str, default="train.csv", help="Path to Kaggle train.csv")

    # Path where the trained model will be saved.
    parser.add_argument("--model_out", type=str, default="digit_cnn.keras", help="Path to save trained model")

    # Path where the training history JSON file will be saved.
    parser.add_argument("--history_out", type=str, default="training_history.json", help="Path to save training history")

    # Number of complete passes over the training dataset.
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")

    # Number of samples processed before the model updates its weights.
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")

    # Percentage of data used for validation.
    parser.add_argument("--validation_size", type=float, default=0.1, help="Validation split size")

    # Random seed for reproducibility.
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")

    # Read all command-line arguments into the args object.
    args = parser.parse_args()

    # Set NumPy random seed so results are more reproducible.
    np.random.seed(args.random_state)

    # Load, normalize, reshape, one-hot encode, and split the data.
    print("Loading data...")
    X_train, X_val, y_train, y_val = load_kaggle_train_csv(
        csv_path=args.train_csv,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    # Show how many samples are used for training and validation.
    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")

    # Build the CNN model.
    print("Building model...")
    model = build_model()

    # Print the model architecture, including layer names, output shapes, and parameters.
    model.summary()

    # Define callbacks used during training.
    callbacks = [
        # Stop training early if validation loss does not improve for 3 epochs.
        # restore_best_weights=True brings back the best model weights.
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),

        # If validation loss does not improve for 2 epochs,
        # reduce the learning rate by half.
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    # Train the model using the training data and validate on validation data.
    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate the trained model on the validation set.
    print("Evaluating on validation set...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    # Print final validation performance.
    print(f"Validation loss     : {val_loss:.4f}")
    print(f"Validation accuracy : {val_acc:.4f}")

    # Prepare the model output path.
    model_out = Path(args.model_out)

    # Create the output folder if it does not already exist.
    model_out.parent.mkdir(parents=True, exist_ok=True)

    # Save the trained model in Keras format.
    model.save(model_out)
    print(f"Saved trained model to: {model_out.resolve()}")

    # Prepare the history output path.
    history_out = Path(args.history_out)

    # Create the output folder if it does not already exist.
    history_out.parent.mkdir(parents=True, exist_ok=True)

    # Store training history and final validation results in a dictionary.
    payload = {
        "history": history.history,
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_acc),
    }

    # Write the dictionary to a JSON file with indentation for readability.
    history_out.write_text(json.dumps(payload, indent=2))
    print(f"Saved training history to: {history_out.resolve()}")


# This ensures main() runs only when this file is executed directly.
# It will not run automatically if the file is imported into another Python file.
if __name__ == "__main__":
    main()
