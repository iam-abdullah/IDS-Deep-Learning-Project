# Intrusion Detection System (IDS) using Deep Learning

This project implements a Deep Learning based Intrusion Detection System (IDS) using a Deep Neural Network (DNN) built with TensorFlow and Keras. It analyzes network traffic data to classify activities as normal or malicious.

## 📌 Project Overview
The model takes preprocessed network traffic data, performs feature analysis, and trains a Sequential Neural Network to predict binary labels (Intrusion vs. Normal).

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib

## 📂 Dataset
The project expects a CSV file located at: `data/work/ids_dataset.csv`.
* **Target Variable:** `Bin Lebel` (Binary Label)
* **Preprocessing:** The script automatically handles infinite values, missing data, and normalization.

## 🚀 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/IDS-Deep-Learning.git](https://github.com/iam-abdullah/IDS-Deep-Learning-Project.git)
    cd IDS-Deep-Learning
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Model**
    ```bash
    python main.py
    ```

## 🧠 Model Architecture
The neural network consists of:
* **Input Layer:** Matches dataset feature dimensions.
* **Hidden Layers:** Dense layers with `ReLU` activation, Batch Normalization, and L1 Regularization.
* **Dropout:** Applied to prevent overfitting.
* **Output Layer:** Sigmoid activation for binary classification.

## 📊 Results
The script generates training accuracy and loss graphs, saved as:
* `graph_training_acc.png`
* `graph_training_loss.png`

## 🤝 Contributing
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature-branch`).
3.  Commit your changes (`git commit -m 'Add new feature'`).
4.  Push to the branch (`git push origin feature-branch`).
5.  Open a Pull Request.
