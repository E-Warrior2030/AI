
#Getting the Data from Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Just making sure that the mounting worked
import os
data_dir = '/content/drive/MyDrive/ML Capstone Project/Data/Raw/chest_xray'
print(os.listdir(data_dir))

#Here we make directories easier to access
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

print(os.listdir(train_dir))

#Knowing the numbers of photos in every directory to make sure there isnt any imbalances


#Training
norm_train_dir = os.path.join(train_dir, 'NORMAL')
print('total training normal images:', len(os.listdir(norm_train_dir)))
pneumonia_train_dir = os.path.join(train_dir, 'PNEUMONIA')
print('total training pneumonia images:', len(os.listdir(pneumonia_train_dir)))
print("\n")

#Val
norm_val_dir = os.path.join(val_dir, 'NORMAL')
print('total val normal images:', len(os.listdir(norm_val_dir)))
pneumonia_val_dir = os.path.join(val_dir, 'PNEUMONIA')
print('total val pneumonia images:', len(os.listdir(pneumonia_val_dir)))
print("\n")

#Test
norm_test_dir = os.path.join(test_dir, 'NORMAL')
print('total test normal images:', len(os.listdir(norm_test_dir)))
pneumonia_test_dir = os.path.join(test_dir, 'PNEUMONIA')
print('total test pneumonia images:', len(os.listdir(pneumonia_test_dir)))

#Have a look at the images
import matplotlib.pyplot as plt
from PIL import Image

fig, axes = plt.subplots(2, 3, figsize=(10, 6))

# NORMAL images
normal_images = [f for f in os.listdir(norm_train_dir) if f.endswith('.jpeg')][:3]
for i, img_name in enumerate(normal_images):
    img = Image.open(os.path.join(norm_train_dir, img_name))
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title('NORMAL')
    axes[0, i].axis('off')

# PNEUMONIA images
pneumonia_images = [f for f in os.listdir(pneumonia_train_dir) if f.endswith('.jpeg')][:3]
for i, img_name in enumerate(pneumonia_images):
    img = Image.open(os.path.join(pneumonia_train_dir, img_name))
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].set_title('PNEUMONIA')
    axes[1, i].axis('off')

plt.suptitle('Sample X-Ray Images')
plt.show()

#Checking Image sizes to know if they are suitable to use

#Normal
img = Image.open(os.path.join(norm_train_dir, normal_images[0]))
print(img.size)
print(img.mode)
print("\n")
#Pneumonia
img = Image.open(os.path.join(pneumonia_train_dir, pneumonia_images[0]))
print(img.size)
print(img.mode)

#Now We notice that the image sizes are too big and we have to reduce them so that all images have a consistent input size for ML models"
from sklearn.preprocessing import LabelEncoder
import numpy as np

IMG_SIZE = 64

def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpeg'):
            # load image, resize, convert to numpy array, flatten
            img = Image.open(os.path.join(folder, filename)).convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img).flatten()
            images.append(img)
            labels.append(label)
    return images, labels

# Here we are running the function or resizing the images. It will take time because we are resizing more than 5000 images
X_normal, y_normal = load_images(norm_train_dir, 0)
X_pneumonia, y_pneumonia = load_images(pneumonia_train_dir, 1)

X = np.array(X_normal + X_pneumonia)
y = np.array(y_normal + y_pneumonia)

print(X.shape)
print(y.shape)

#Now that we have finished scaling pixel values to 0-1 range the images its time to split the data

X = X / 255.0 #Normalization

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

# Now we do the same thing but for the test set. we wont touch this until the very end
X_normal_test, y_normal_test = load_images(norm_test_dir, 0)
X_pneumonia_test, y_pneumonia_test = load_images(pneumonia_test_dir, 1)

X_test = np.array(X_normal_test + X_pneumonia_test) / 255.0
y_test = np.array(y_normal_test + y_pneumonia_test)

print(X_test.shape)
print(y_test.shape)

#Here we make a function that will evaluate every model we use to determine which is the best
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    # calculate and print all 4 metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Pneumonia'])
    disp.plot()
    plt.title(f'Confusion Matrix')
    plt.show()
    return accuracy, precision, recall, f1

#Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
evaluate_model(lr_model, X_val, y_val)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, X_val, y_val)

# SVC
from sklearn.svm import SVC

svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
evaluate_model(svm_model, X_val, y_val)

#Here we make a small dataframe to just compare the data easier

import pandas as pd

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
    'Accuracy': [0.9607, 0.9502, 0.9550],
    'Precision': [0.9711, 0.9560, 0.9696],
    'Recall': [0.9749, 0.9762, 0.9683],
    'F1': [0.9730, 0.9660, 0.9689]
})

print(results)

from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(8, 6))

for model, name in [(lr_model, 'Logistic Regression'),
                     (rf_model, 'Random Forest'),
                     (svm_model, 'SVM')]:
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                           param_grid,
                           cv=5,
                           scoring='f1')

grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best F1:", grid_search.best_score_)

# Best model with tuned params
best_model = LogisticRegression(C=1, max_iter=1000, random_state=42)
best_model.fit(X_train, y_train)

print("Before tuning:")
evaluate_model(lr_model, X_val, y_val)

print("\nAfter tuning:")
evaluate_model(best_model, X_val, y_val)

print("Final Test Set Evaluation:")
evaluate_model(best_model, X_test, y_test)