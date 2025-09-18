import tensorflow_datasets as tfds
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load MNIST dataset from TFDS
print("Loading MNIST dataset from TFDS...")
ds = tfds.load("mnist", split="train+test", as_supervised=True)

# Convert dataset to numpy arrays
X = []
y = []
for image, label in tfds.as_numpy(ds):
    X.append(image.reshape(-1))   # Flatten 28x28 → 784
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape, y.shape)

# 2. Normalize pixel values (0–255 → 0–1)
X = X / 255.0


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train KNN model
print("Training KNN...")
knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto', n_jobs=-1)
knn.fit(X_train, y_train)

# 5. Make predictions
y_pred = knn.predict(X_test)

# 6. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy on MNIST:", accuracy)
