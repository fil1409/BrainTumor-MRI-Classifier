import os
import imagehash
from PIL import Image

# Funzione per calcolare l'hash di tutte le immagini in una directory con sottocartelle
def calcola_hash(base_directory):
    hashes = {}
    for root, _, files in os.walk(base_directory):
        for filename in files:
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, filename)
                image = Image.open(image_path)
                image_hash = imagehash.phash(image)
                relative_path = os.path.relpath(image_path, base_directory)
                hashes[relative_path] = image_hash
    return hashes

# Percorsi dei dataset
dataset1_path = r""
dataset2_path = r""

# Calcolare gli hash per i due dataset
hashes_dataset1 = calcola_hash(dataset1_path)
hashes_dataset2 = calcola_hash(dataset2_path)

# Confrontare gli hash
immagini_comuni = []
for file1, hash1 in hashes_dataset1.items():
    for file2, hash2 in hashes_dataset2.items():
        if hash1 == hash2:
            immagini_comuni.append((file1, file2))

# Stampa le immagini comuni
print("Immagini comuni nei due dataset:")
for img1, img2 in immagini_comuni:
    print(f"{img1} in dataset1 e {img2} in dataset2")




import os
import shutil
import imagehash
from PIL import Image

# Funzione per calcolare l'hash di tutte le immagini in una directory con sottocartelle
def calcola_hash(base_directory):
    hashes = {}
    for root, _, files in os.walk(base_directory):
        for filename in files:
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, filename)
                image = Image.open(image_path)
                image_hash = imagehash.phash(image)
                relative_path = os.path.relpath(image_path, base_directory)
                hashes[relative_path] = image_hash
    return hashes

# Percorsi dei dataset
dataset1_path = r""
dataset2_path = r""
dataset1_updated_path = r""

# Calcolare gli hash per i due dataset
hashes_dataset1 = calcola_hash(dataset1_path)
hashes_dataset2 = calcola_hash(dataset2_path)

# Confrontare gli hash e trovare immagini duplicate
immagini_duplicate = set()
for file1, hash1 in hashes_dataset1.items():
    for file2, hash2 in hashes_dataset2.items():
        if hash1 == hash2:
            immagini_duplicate.add(file1)

# Creare il nuovo dataset senza immagini duplicate
for root, _, files in os.walk(dataset1_path):
    for filename in files:
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(root, filename)
            relative_path = os.path.relpath(image_path, dataset1_path)
            new_image_path = os.path.join(dataset1_updated_path, relative_path)

            if relative_path not in immagini_duplicate:
                os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                shutil.copy2(image_path, new_image_path)

# Numero totale di immagini duplicate
numero_duplicati = len(immagini_duplicate)

# Stampa le immagini duplicate e il numero totale di immagini duplicate
print("Immagini duplicate rimosse dal dataset1:")
for img in immagini_duplicate:
    print(f"{img}")

print(f"\nNumero totale di immagini duplicate rimosse: {numero_duplicati}")


import os
import cv2
import shutil

def crop_to_content(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions + dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Apply crop
    ADD_PIXELS = 0
    new_image = image[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

    return new_image

def process_images(input_dir, output_dir):
    # Create output directories for each class if they don't exist
    class_folders = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    for folder in class_folders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # Iterate through class folders
    for folder in class_folders:
        input_class_dir = os.path.join(input_dir, folder)
        output_class_dir = os.path.join(output_dir, folder)

        # Iterate through files in the class folder
        for filename in os.listdir(input_class_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust based on your file extensions
                input_path = os.path.join(input_class_dir, filename)
                output_path = os.path.join(output_class_dir, filename)

                # Read the image
                image = cv2.imread(input_path)
                if image is None:
                    continue  # Skip if image cannot be read

                # Process the image
                processed_image = crop_to_content(image)

                # Save processed image
                cv2.imwrite(output_path, processed_image)

                print(f"Processed: {os.path.join(folder, filename)}")

# Paths to input and output directories
input_directory = r""
output_directory = r""

# Process images
process_images(input_directory, output_directory)


