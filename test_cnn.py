# argparse is used to read command-line options such as --mode, --model_path, etc.
import argparse

# pathlib makes file/folder path handling cleaner and more cross-platform.
from pathlib import Path

# NumPy is used for array operations and converting image pixels into model-ready input.
import numpy as np

# Pandas is used to read CSV files for Kaggle-style testing or evaluation.
import pandas as pd

# Tkinter is Python's built-in GUI library. Here it is used to create the drawing window.
import tkinter as tk

# Pillow is used to create, draw, resize, crop, and save images.
from PIL import Image, ImageDraw, ImageOps

# load_model loads the trained CNN model saved from the training script.
from tensorflow.keras.models import load_model

# to_categorical converts labels like 0, 1, 2... into one-hot encoded labels.
# Example: label 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
from tensorflow.keras.utils import to_categorical


# Size of the drawing canvas shown to the user.
# 280x280 is used because it scales nicely down to 28x28.
GRID_SIZE = 280

# Size expected by the trained digit CNN model.
# MNIST/Kaggle Digit Recognizer images are 28x28 pixels.
MODEL_IMG_SIZE = 28

# Thickness of the brush used when drawing on the canvas.
BRUSH_SIZE = 16


def preprocess_features(df: pd.DataFrame) -> np.ndarray:
    """
    Convert CSV pixel values into the input format expected by the CNN model.

    The CSV contains one row per image.
    Each row has 784 pixel values, representing a flattened 28x28 grayscale image.

    Steps:
    1. Convert values to float32.
    2. Normalize pixel values from 0-255 to 0-1.
    3. Reshape the flat 784 values into 28x28x1 images.
    """
    # Convert the DataFrame to a NumPy array and normalize pixel values.
    X = df.values.astype("float32") / 255.0

    # Reshape data into CNN input format:
    # number_of_images x height x width x channels.
    return X.reshape(-1, 28, 28, 1)


class DigitDrawerApp:
    """
    A simple Tkinter application that lets the user draw a digit
    and test the trained CNN model on that handwritten drawing.
    """

    def __init__(self, model):
        # Store the loaded CNN model so it can be used for prediction.
        self.model = model

        # Create the main application window.
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Tester")

        # Prevent the user from resizing the window.
        self.root.resizable(False, False)

        # Create the drawing canvas.
        # The user draws with white color on a black background,
        # matching the style of MNIST/Kaggle digit images.
        self.canvas = tk.Canvas(
            self.root,
            width=GRID_SIZE,
            height=GRID_SIZE,
            bg="black",
            cursor="cross",
            highlightthickness=1,
            highlightbackground="#888",
        )

        # Place the canvas in the window.
        # columnspan=4 means it stretches across 4 button columns.
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Label used to show the final predicted digit and confidence.
        self.result_label = tk.Label(
            self.root,
            text="Draw a digit and click Predict",
            font=("Arial", 14, "bold"),
        )
        self.result_label.grid(row=1, column=0, columnspan=4, pady=(0, 10))

        # Label used to show the top 3 predictions with percentages.
        self.top3_label = tk.Label(
            self.root,
            text="Top predictions will appear here",
            font=("Arial", 10),
            justify="left",
        )
        self.top3_label.grid(row=2, column=0, columnspan=4, pady=(0, 10))

        # Button to run the model on the drawn digit.
        tk.Button(
            self.root,
            text="Predict",
            width=12,
            command=self.predict_digit,
        ).grid(row=3, column=0, padx=5, pady=10)

        # Button to clear the drawing area.
        tk.Button(
            self.root,
            text="Clear",
            width=12,
            command=self.clear_canvas,
        ).grid(row=3, column=1, padx=5, pady=10)

        # Button to save the processed 28x28 image.
        # This is useful for checking what the model is actually seeing.
        tk.Button(
            self.root,
            text="Save 28x28",
            width=12,
            command=self.save_processed_image,
        ).grid(row=3, column=2, padx=5, pady=10)

        # Button to close the application.
        tk.Button(
            self.root,
            text="Exit",
            width=12,
            command=self.root.destroy,
        ).grid(row=3, column=3, padx=5, pady=10)

        # Bind mouse events to the draw function.
        # <B1-Motion> means drawing while left mouse button is held down.
        # <Button-1> means drawing a dot when the user first clicks.
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        # Create a hidden grayscale image that stores the user's drawing.
        # This image is used for preprocessing and prediction.
        self.image = Image.new("L", (GRID_SIZE, GRID_SIZE), 0)

        # Create a drawing object for the hidden image.
        self.draw_handle = ImageDraw.Draw(self.image)

        # Draw the grid lines on the visible canvas.
        self.draw_grid()

    def draw_grid(self):
        """
        Draw a light grid on the canvas.

        The grid helps the user understand how the 280x280 drawing area
        maps down to the 28x28 model input.
        """
        # Each model pixel corresponds to a 10x10 area on the canvas.
        step = GRID_SIZE // MODEL_IMG_SIZE

        # Draw vertical and horizontal grid lines.
        for i in range(0, GRID_SIZE, step):
            self.canvas.create_line(i, 0, i, GRID_SIZE, fill="#222")
            self.canvas.create_line(0, i, GRID_SIZE, i, fill="#222")

    def draw(self, event):
        """
        Draw white circles wherever the mouse moves.

        The drawing is added to:
        1. The visible Tkinter canvas.
        2. The hidden Pillow image used for prediction.
        """
        # Calculate the brush circle boundaries around the mouse pointer.
        x1, y1 = event.x - BRUSH_SIZE, event.y - BRUSH_SIZE
        x2, y2 = event.x + BRUSH_SIZE, event.y + BRUSH_SIZE

        # Draw on the visible canvas.
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")

        # Draw the same shape on the hidden image.
        # Pixel value 255 means white in grayscale.
        self.draw_handle.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        """
        Clear the drawing canvas and reset prediction labels.
        """
        # Remove everything from the visible canvas.
        self.canvas.delete("all")

        # Restore black background.
        self.canvas.configure(bg="black")

        # Redraw grid lines after clearing.
        self.draw_grid()

        # Reset the hidden image to a blank black image.
        self.image = Image.new("L", (GRID_SIZE, GRID_SIZE), 0)
        self.draw_handle = ImageDraw.Draw(self.image)

        # Reset labels.
        self.result_label.config(text="Draw a digit and click Predict")
        self.top3_label.config(text="Top predictions will appear here")

    def preprocess_drawn_image(self) -> np.ndarray:
        """
        Convert the user's drawing into a 28x28 image for the CNN model.

        Steps:
        1. Copy the hidden drawing image.
        2. Find the bounding box around the drawn digit.
        3. Crop the digit.
        4. Resize it so it fits inside a 20x20 area.
        5. Center it inside a 28x28 black background.
        6. Normalize values and reshape for CNN input.
        """
        # Work on a copy so the original drawing is not changed.
        img = self.image.copy()

        # Find the bounding box around non-black pixels.
        bbox = img.getbbox()

        # If nothing was drawn, return a blank 28x28 image.
        if bbox is None:
            return np.zeros((1, 28, 28, 1), dtype="float32")

        # Crop the image tightly around the drawn digit.
        img = img.crop(bbox)

        # Resize the digit while preserving its aspect ratio.
        # The digit is contained within a 20x20 area, leaving padding around it.
        img = ImageOps.contain(img, (20, 20))

        # Create a blank 28x28 black background.
        background = Image.new("L", (28, 28), 0)

        # Calculate offsets to center the resized digit.
        offset_x = (28 - img.width) // 2
        offset_y = (28 - img.height) // 2

        # Paste the resized digit into the center of the 28x28 background.
        background.paste(img, (offset_x, offset_y))

        # Convert the final image into a normalized NumPy array.
        arr = np.array(background).astype("float32") / 255.0

        # Reshape to match the model input format:
        # batch_size x height x width x channels.
        arr = arr.reshape(1, 28, 28, 1)

        return arr

    def predict_digit(self):
        """
        Run the trained CNN model on the drawn digit
        and display the prediction result.
        """
        # Prepare the drawn image for the model.
        x = self.preprocess_drawn_image()

        # Get prediction probabilities for all 10 digit classes.
        probs = self.model.predict(x, verbose=0)[0]

        # Choose the digit with the highest probability.
        pred = int(np.argmax(probs))

        # Convert the highest probability into a percentage confidence.
        conf = float(np.max(probs)) * 100

        # Get indexes of the top 3 predicted digits.
        top3_idx = np.argsort(probs)[-3:][::-1]

        # Format the top 3 predictions as text.
        top3_text = "\n".join(
            [f"{digit}: {probs[digit] * 100:.2f}%" for digit in top3_idx]
        )

        # Update the main result label.
        self.result_label.config(text=f"Prediction: {pred}   |   Confidence: {conf:.2f}%")

        # Update the top 3 predictions label.
        self.top3_label.config(text=f"Top 3 predictions:\n{top3_text}")

    def save_processed_image(self):
        """
        Save the 28x28 image after preprocessing.

        This helps debug the model input.
        If predictions are bad, check this saved image to see
        whether the digit is centered and clear.
        """
        # Get the processed image and convert values back from 0-1 to 0-255.
        x = self.preprocess_drawn_image()[0, :, :, 0] * 255.0

        # Convert the NumPy array back into a grayscale Pillow image.
        img = Image.fromarray(x.astype(np.uint8), mode="L")

        # Save the image in the current folder.
        output_path = Path("drawn_digit_28x28.png")
        img.save(output_path)

        # Show the saved file path in the app.
        self.result_label.config(text=f"Saved processed image to: {output_path.resolve()}")

    def run(self):
        """
        Start the Tkinter application loop.
        """
        self.root.mainloop()


def run_csv_mode(model, input_csv: str, output_csv: str):
    """
    Run the saved CNN model on a CSV file.

    This function supports two CSV types:

    1. Labeled CSV:
       - Contains a 'label' column.
       - Used for evaluation because true answers are available.

    2. Unlabeled CSV:
       - Does not contain a 'label' column.
       - Used for Kaggle-style prediction/submission generation.
    """
    print("Reading input CSV...")

    # Read the input CSV file.
    df = pd.read_csv(input_csv)

    # If the CSV has labels, evaluate model accuracy.
    if "label" in df.columns:
        print("Detected labeled CSV. Running evaluation and sample predictions...")

        # Extract true labels.
        y_true = df["label"].values

        # Convert image pixel columns into CNN input format.
        X = preprocess_features(df.drop(columns=["label"]))

        # Convert labels into one-hot format for model.evaluate().
        y_true_cat = to_categorical(y_true, num_classes=10)

        # Evaluate the model on this dataset.
        loss, acc = model.evaluate(X, y_true_cat, verbose=0)

        # Predict class probabilities.
        probs = model.predict(X, verbose=0)

        # Convert probabilities into final predicted digit labels.
        preds = np.argmax(probs, axis=1)

        # Print evaluation results.
        print(f"Loss     : {loss:.4f}")
        print(f"Accuracy : {acc:.4f}")
        print("First 10 predictions:")
        print(preds[:10])

    # If the CSV does not have labels, create a submission CSV.
    else:
        print("Detected unlabeled CSV. Running Kaggle-style inference...")

        # Convert CSV pixel data into CNN input format.
        X = preprocess_features(df)

        # Predict probabilities for each digit.
        probs = model.predict(X, verbose=0)

        # Convert probabilities into final predicted digit labels.
        preds = np.argmax(probs, axis=1)

        # Create a Kaggle-style submission DataFrame.
        submission = pd.DataFrame({
            "ImageId": np.arange(1, len(preds) + 1),
            "Label": preds,
        })

        # Create output folder if it does not exist.
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save predictions to CSV.
        submission.to_csv(output_path, index=False)

        # Print output location and first few predictions.
        print(f"Saved predictions to: {output_path.resolve()}")
        print(submission.head())


def main():
    """
    Main entry point of the script.

    This function:
    1. Reads command-line arguments.
    2. Loads the saved CNN model.
    3. Runs either drawing mode or CSV mode.
    """
    # Create a command-line argument parser.
    parser = argparse.ArgumentParser(description="Test a saved CNN model using CSV input or a drawing grid.")

    # Path to the trained Keras model file.
    parser.add_argument(
        "--model_path",
        type=str,
        default="digit_cnn.keras",
        help="Path to saved .keras model",
    )

    # Choose whether to run the GUI drawing app or CSV inference.
    parser.add_argument(
        "--mode",
        type=str,
        choices=["draw", "csv"],
        default="draw",
        help="Use 'draw' to sketch a digit on a grid, or 'csv' to run the old CSV-based inference.",
    )

    # Input CSV file path used only in csv mode.
    parser.add_argument(
        "--input_csv",
        type=str,
        default="test.csv",
        help="CSV path used only when --mode csv",
    )

    # Output CSV file path used only for unlabeled CSV prediction.
    parser.add_argument(
        "--output_csv",
        type=str,
        default="submission.csv",
        help="Prediction output path used only when --mode csv",
    )

    # Convert command-line arguments into the args object.
    args = parser.parse_args()

    # Load the trained CNN model from disk.
    print("Loading model...")
    model = load_model(args.model_path)

    # Start GUI drawing mode.
    if args.mode == "draw":
        app = DigitDrawerApp(model)
        app.run()

    # Otherwise run CSV-based testing/prediction mode.
    else:
        run_csv_mode(model, args.input_csv, args.output_csv)


# This ensures main() runs only when this file is executed directly.
# It prevents automatic execution when the file is imported into another script.
if __name__ == "__main__":
    main()
