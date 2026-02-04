import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath("file"))
DATA_DIR = os.path.join(BASE_DIR, "train", "train")

print("Searching dataset in:", DATA_DIR)

folders = os.listdir(DATA_DIR)
cat_folder = None
dog_folder = None

for folder in folders:
    if folder.lower() in ["cat", "cats"]:
        cat_folder = folder
    if folder.lower() in ["dog", "dogs"]:
        dog_folder = folder

if cat_folder is None or dog_folder is None:
    print("❌ Cat/Dog folders not found. Found:", folders)
    exit()

CAT_PATH = os.path.join(DATA_DIR, cat_folder)
DOG_PATH = os.path.join(DATA_DIR, dog_folder)

print("Cats folder:", CAT_PATH)
print("Dogs folder:", DOG_PATH)

IMG_SIZE = 64
LIMIT = 800

data = []
labels = []

print("Loading Cat images...")
for img_name in tqdm(os.listdir(CAT_PATH)[:LIMIT]):
    try:
        img_path = os.path.join(CAT_PATH, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data.append(image.flatten())
        labels.append(0)
    except:
        pass

print("Loading Dog images...")
for img_name in tqdm(os.listdir(DOG_PATH)[:LIMIT]):
    try:
        img_path = os.path.join(DOG_PATH, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data.append(image.flatten())
        labels.append(1)
    except:
        pass

print("Total images loaded:", len(data))


X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training SVM model...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 Model Accuracy:", round(accuracy * 100, 2), "%")


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
disp.plot()
plt.title("Confusion Matrix - Cats vs Dogs")

cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
print("📊 Confusion matrix saved as:", cm_path)
plt.show()

unique, counts = np.unique(y_pred, return_counts=True)

labels_names = ["Cat", "Dog"]
counts_list = [0, 0]

for i, val in enumerate(unique):
    counts_list[val] = counts[i]

plt.figure()
plt.bar(labels_names, counts_list)
plt.title("Cats vs Dogs Classification Results")
plt.xlabel("Predicted Class")
plt.ylabel("Number of Images")

graph2_path = os.path.join(BASE_DIR, "classification_results.png")
plt.savefig(graph2_path)

print("Classification graph saved as:", graph2_path)
plt.show()

