
## **# HandDraw Sketch Classifier 🖍️**  

This project is a **hand-drawn sketch classification app** using a **Convolutional Neural Network (CNN)** trained on the **Google QuickDraw dataset**. The model can recognize different hand-drawn objects and classify them into predefined categories.  

---

## **📌 Features**
- Trains a **CNN model** to recognize sketches.
- Uses the **QuickDraw dataset** to generate training images.
- Supports **real-time sketch classification**.
- Simple architecture with **PyTorch** for training.
- Saves trained weights in `quickdraw_model.pth` for reuse.

---

## **🛠️ Installation**  
### **1. Clone the Repository**  
```bash
git clone https://github.com/your-username/handdraw-classifier.git
cd handdraw-classifier
```

### **2. Install Dependencies**  
```bash
pip install torch torchvision numpy opencv-python pillow
```

---

## **📂 Project Structure**
```
📦 handdraw-classifier
 ┣ 📜 data_download.py       # Downloads and processes the dataset
 ┣ 📜 train.py               # Trains the CNN model
 ┣ 📜 model.py               # Defines the CNN architecture
 ┣ 📜 drawing_app            # Starts hand tracker to draw
 ┣ 📜 README.md              # Documentation
```

---

## **🚀 Usage**
### **1. Download the Dataset**
```bash
python data_download.py
```
This script downloads and extracts **balanced** image datasets for training.

### **2. Train the Model**
```bash
python train.py
```
This will train the CNN model on the QuickDraw dataset and save the trained weights in `quickdraw_model.pth`.

---

## **🧠 Model Architecture**
The model is a **Convolutional Neural Network (CNN)** with the following layers:
- **2 Convolutional layers** with **ReLU activation**.
- **MaxPooling layers** to reduce dimensions.
- **Fully connected layers** for classification.
- Uses **CrossEntropyLoss** and the **Adam optimizer**.

---

## **🎯 Classes**
The model is trained on **7 categories** of sketches:
✅ Apple  
✅ Banana  
✅ Cat  
✅ Dog  
✅ Tree  
✅ Car  
✅ Fish  

To add more classes, modify the `class_labels` list in **model.py**.

---

## **📌 Future Improvements**
- Improve accuracy with **data augmentation**.
- Implement **real-time sketch recognition**.
- Deploy as a **web app** using Flask or Streamlit.

---

## **📜 License**
This project is open-source and available under the **MIT License**.
