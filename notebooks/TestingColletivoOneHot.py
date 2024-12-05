from keras.models import load_model
import numpy as np
import os
from tqdm import tqdm
import pickle
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.utils import to_categorical

TEST_DIR = ''
IMG_SIZE = 224
CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
MODEL_SAVE_PATH = ''

def create_testing_data(test_dirs):
    testing_data = []
    for test_dir in test_dirs:
        for category in CATEGORIES:
            path = os.path.join(test_dir, category)
            class_num = CATEGORIES.index(category)
            for img in tqdm(os.listdir(path)):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([new_array, class_num])
    random.shuffle(testing_data)
    return testing_data

def is_one_hot_encoded(labels):
    if len(labels.shape) > 1 and labels.shape[1] == len(CATEGORIES):
        return True
    return False

def evaluate_and_plot(model, test_dirs, labels):
    accuracies = []
    for test_dir, label in zip(test_dirs, labels):
        testing_data = create_testing_data([test_dir])
        X_test = np.array([i[0] for i in testing_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        Y_test = np.array([i[1] for i in testing_data])
        X_test = X_test / 255.0

        # Convert Y_test to one hot encoding if not already
        if not is_one_hot_encoded(Y_test):
            Y_test = to_categorical(Y_test, num_classes=len(CATEGORIES))

        # Predict and evaluate
        scores = model.evaluate([X_test, X_test], Y_test, verbose=1)
        accuracies.append(scores[1])
        print(f'Test loss on {test_dir}:', scores[0])
        print(f'Test accuracy on {test_dir}:', scores[1])

        # Generate predictions and compute confusion matrix
        Y_pred = model.predict([X_test, X_test])
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        Y_test_classes = np.argmax(Y_test, axis=1)
        cm = confusion_matrix(Y_test_classes, Y_pred_classes)

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CATEGORIES)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {label}')
        plt.show()

    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.bar(labels, accuracies, color='skyblue')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy on Different Datasets')
    plt.show()

    model = load_model(MODEL_SAVE_PATH)

    # Define the test directories and labels
test_dirs = [
    '',
    ''
    ]
labels = ['Testing Set SartaJ(pre)', 'Fused Testing Set (pre)']
          
# Evaluate the model and plot the results
evaluate_and_plot(model, test_dirs, labels)