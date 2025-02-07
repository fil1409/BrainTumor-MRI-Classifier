# Brain Tumor Classification with MRI Images Using Fusion Model   

## 🧠 Introduzione  
Il progetto mira a sviluppare un sistema di classificazione automatica dei tumori cerebrali utilizzando immagini di risonanza magnetica (MRI) e modelli di deep learning combinati. Questo sistema mira a supportare i medici nella diagnosi precoce e accurata dei tumori cerebrali, migliorando così i trattamenti e riducendo il rischio di errori umani.

---

## 🚀 Sistema Proposto  
Il sistema è stato sviluppato seguendo i seguenti passaggi:  
1. **Preprocessing**: Identificazione dei duplicati, ridimensionamento a 224x224 pixel, normalizzazione e filtraggio.  
2. **Data Augmentation**: Rotazioni, modifiche di luminosità, contrasto e distorsioni.  
3. **Feature Extraction**: Utilizzo dei modelli pre-addestrati ResNet50 e VGG16 per estrarre le caratteristiche chiave.  
4. **Modello Fuso**: Combinazione delle caratteristiche estratte per migliorare le prestazioni.

---

## 📊 Dataset  
- **Sartaj Dataset**: 3.265 immagini suddivise in 4 classi (glioma, meningioma, pituitary, no_tumor).
  **Distribuzione del Dataset:**
  - **Training**: 80% delle immagini.
  - **Testing**: 20% delle immagini.
- **Figshare + Br35H**: Unione dei due dataset per ottenere 903 immagini usate esclusivamente per testare il modello.


---

## 🛠️ Preprocessing  
- **Calcolo dell'Hash**: Identificazione e rimozione dei duplicati.  
- **Ridimensionamento**: Tutte le immagini sono ridimensionate a 224x224 pixel per garantire uniformità.  
- **Normalizzazione**: I valori dei pixel sono normalizzati per migliorare la qualità dei dati in input.  
- **Filtraggio Gaussiano**: Riduce il rumore e mantiene la regione di interesse.
---

## 🔄 Data Augmentation  
- **Rotazioni e Flip**: Rotazioni casuali tra -7° e +7° con flip orizzontale e verticale.  
- **Modifiche di Luminosità e Contrasto**: Alterazioni casuali per aumentare la varietà delle immagini.  
- **Distorsioni Elastiche e Equalizzazione**: Per migliorare la rappresentazione delle immagini.

---

## 🔧 Modello Fuso  
Il modello fuso utilizza le caratteristiche estratte dagli ultimi layer convoluzionali di ResNet50 e VGG16 per costruire un sistema più robusto e preciso.  

---

## 📈 Testing e Risultati  
- Il modello ha ottenuto un'accuratezza complessiva del **92%**.  
- La matrice di confusione mostra prestazioni solide nelle quattro classi:  
![Matrice di Confusione](images/Immagine1.jpg)

## 🏁 Come Eseguire
1)  Clonare la repository

    - git clone https://github.com/username/BrainTumor-MRI-Classifier.git
    - cd BrainTumor-MRI-Classifier

2)  Eseguire il preprocessing

    - Utilizzare  OperazioniPreProcessing.py per eseguire le operazioni di pre processing.

3)  Training del modello
   
    - Addestrare il modello utilizzando ModelloTrainingCollettivo.py .

4)  Testing del modello
   
    - Visualizzare i risultati del modello attraverso TestingCollettivo.ipynb .


 # Brain Tumor Classification with MRI Images Using Fusion Model   

## 🧠 Introduction
This project aims to develop an automatic brain tumor classification system using MRI images and combined deep learning models. The system is designed to assist doctors in the early and accurate diagnosis of brain tumors, thereby improving treatments and reducing the risk of human errors.

---

## 🧠 Proposed System
The system was developed following these steps:  
1. **Preprocessing**: Identification of duplicates, resizing to 224x224 pixels, normalization, and filtering.  
2. **Data Augmentation**: Rotations, brightness and contrast adjustments, and distortions.  
3. **Feature Extraction**: Using pre-trained ResNet50 and VGG16 models to extract key features.  
4. **Fusion Model**: Combining extracted features to enhance performance.

---

## 📊 Dataset  
- **Sartaj Dataset**: 3,265 images divided into 4 classes (glioma, meningioma, pituitary, no_tumor).
  **Dataset Distribution:**
  - **Training**: 80% of the images.
  - **Testing**: 20% of the images.
- **Figshare + Br35H**: A combination of both datasets, resulting in 903 images used exclusively for model testing.

---

## 🛠️ Preprocessing  
- **Hash Calculation**: Identifying and removing duplicates.  
- **Resizing**: All images are resized to 224x224 pixels to ensure uniformity.  
- **Normalization**: Pixel values are normalized to improve input data quality. 
- **Gaussian Filtering**: Reduces noise while preserving the region of interest.

## 🔄 Data Augmentation  
- **Rotations and Flipping**: Random rotations between -7° and +7° with horizontal and vertical flips.
- **Brightness and Contrast Adjustments**: Random alterations to increase image variety.
- **Elastic Distortions and Equalization**: To enhance image representation.

---

## 🔧 Fusion Model   
The fusion model utilizes features extracted from the last convolutional layers of ResNet50 and VGG16 to build a more robust and accurate system. 

---

## 📈 Testing e Result  
- The model achieved an overall accuracy of **92%**.  
- The confusion matrix demonstrates solid performance across all four classes: 
![Confusion Matrix](images/Immagine1.jpg)

## 🏁 How to Run
1) Clone the repository

    - git clone https://github.com/username/BrainTumor-MRI-Classifier.git
    - cd BrainTumor-MRI-Classifier

2) Run preprocessing

    - Use OperazioniPreProcessing.py to execute preprocessing operations.

3)  Train the model
   
    - Train the model using ModelloTrainingCollettivo.py.

4)  Test the model
   
    - View the model results through TestingCollettivo.ipynb.

