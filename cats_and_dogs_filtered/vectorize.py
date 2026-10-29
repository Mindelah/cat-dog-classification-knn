import os
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tkinter import Tk, filedialog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Project settings

DATASET = "cats_and_dogs_filtered"
IMAGE_SIZE = (32, 32)
K = 3

# 2. Load and prepare images

def load_images(folder, label):
    images = []
    labels = []

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        try:
            image = Image.open(path).convert("RGB")
            image = image.resize(IMAGE_SIZE)
            image = np.array(image).flatten()

            images.append(image)
            labels.append(label)

        except (OSError, ValueError):
            continue

    return images, labels


def load_dataset(split):
    cats, cat_labels = load_images(
        os.path.join(DATASET, split, "cats"), 0
    )
    dogs, dog_labels = load_images(
        os.path.join(DATASET, split, "dogs"), 1
    )

    X = np.array(cats + dogs)
    y = np.array(cat_labels + dog_labels)

    print(split, "images:", len(X))
    return X, y


# 3. Load training and testing data

X_train, y_train = load_dataset("train")
X_test, y_test = load_dataset("validation")

# 4. Create and train KNN models

euclidean = KNeighborsClassifier(
    n_neighbors=K, metric="euclidean", n_jobs=-1
)

manhattan = KNeighborsClassifier(
    n_neighbors=K, metric="manhattan", n_jobs=-1
)

euclidean.fit(X_train, y_train)
manhattan.fit(X_train, y_train)

# 5. Predict test images

start = time.time()
y_euclidean = euclidean.predict(X_test)
euclidean_time = time.time() - start

start = time.time()
y_manhattan = manhattan.predict(X_test)
manhattan_time = time.time() - start

# 6. Calculate accuracy

accuracy_e = accuracy_score(y_test, y_euclidean)
accuracy_m = accuracy_score(y_test, y_manhattan)

print("\nEuclidean Accuracy:", round(accuracy_e * 100, 2), "%")
print("Manhattan Accuracy:", round(accuracy_m * 100, 2), "%")

print("Euclidean Prediction Time:", round(euclidean_time, 2), "seconds")
print("Manhattan Prediction Time:", round(manhattan_time, 2), "seconds")

# 7. Display classification reports

print("\nEuclidean Classification Report:")
print(classification_report(
    y_test, y_euclidean,
    labels=[0, 1],
    target_names=["Cat", "Dog"],
    zero_division=0
))

print("\nManhattan Classification Report:")
print(classification_report(
    y_test, y_manhattan,
    labels=[0, 1],
    target_names=["Cat", "Dog"],
    zero_division=0
))

# 8. Display confusion matrices

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_euclidean,
    labels=[0, 1],
    display_labels=["Cat", "Dog"],
    ax=axes[0],
    colorbar=False
)

axes[0].set_title("Euclidean")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_manhattan,
    labels=[0, 1],
    display_labels=["Cat", "Dog"],
    ax=axes[1],
    colorbar=False
)

axes[1].set_title("Manhattan")

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

# 9. Compare accuracy

plt.bar(
    ["Euclidean", "Manhattan"],
    [accuracy_e * 100, accuracy_m * 100]
)

plt.title("KNN Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

plt.savefig("accuracy_comparison.png")
plt.show()

# 10. Predict a new picture

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Choose a Cat or Dog Image",
    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
)

root.destroy()

if image_path:
    image = Image.open(image_path).convert("RGB")
    image_small = image.resize(IMAGE_SIZE)

    new_image = np.array(image_small).flatten().reshape(1, -1)

    result = manhattan.predict(new_image)[0]
    label = "Cat" if result == 0 else "Dog"

    print("\nPredicted Result:", label)

    plt.imshow(image)
    plt.title("Prediction: " + label)
    plt.axis("off")
    plt.show()

# 11. Final result

print("\nEuclidean:", round(accuracy_e * 100, 2), "%")
print("Manhattan:", round(accuracy_m * 100, 2), "%")

if accuracy_e > accuracy_m:
    print("Euclidean performed better.")
elif accuracy_m > accuracy_e:
    print("Manhattan performed better .")
else:
    print("Both methods achieved the same accuracy.")
print("\nDone.")
#python .\cats_and_dogs_filtered\vectorize.py