"""
Project: Network Intrusion Detection using Deep Learning (CICIDS Dataset 2017)
Objective: Binary Classification of Network Traffic (BENIGN vs. ATTACK)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# ---------------------------------------------------------
# DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------

def perform_correlation_analysis(df, threshold=0.95):
    """Identifies columns with high correlation (for analysis purposes)."""
    print("\n--- Performing Correlation Analysis ---")
    X_num = df.select_dtypes(include=[np.number])
    corr_matrix = X_num.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop_corr = [
        column for column in upper.columns
        if any(upper[column] > threshold)
    ]

    print(f"Columns identified for removal due to high correlation (> {threshold}):")
    print(to_drop_corr)
    return to_drop_corr

# ---------------------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------------------

def build_model(input_dim):
    """Builds the Sequential Neural Network model."""
    model = Sequential()
    
    # Input Layer
    model.add(Dense(1024, activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    
    # Hidden Layers
    model.add(Dense(512, activation='relu', kernel_regularizer='l1'))
    model.add(BatchNormalization())
    
    model.add(Dense(512, activation='relu', kernel_regularizer='l1'))
    model.add(BatchNormalization())
    
    model.add(Dense(256, activation='relu', kernel_regularizer='l1'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    
    # Output Layer
    model.add(Dense(1, activation='sigmoid'))
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model

# ---------------------------------------------------------
# TRAINING & EVALUATION
# ---------------------------------------------------------

def train_and_test(x_data, y_data, epochs=50, batch_size=256):
    """Splits data, trains the model, and evaluates performance."""
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=42)
    
    # Reshape targets
    y_train = y_train.values.reshape(y_train.shape[0], 1)
    y_test = y_test.values.reshape(y_test.shape[0], 1)

    print(f"X train shape: {X_train.shape}, y train shape: {y_train.shape}")
      
    model = build_model(x_data.shape[1])
    
    history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    pred = model.evaluate(X_test, y_test, verbose=1)
    
    val_acc, val_loss = pred[1] * 100, pred[0] * 100
    print('Test Data: accuracy: {:.2f}%: loss: {:.2f}'.format(val_acc, val_loss))

    return model, history, val_acc, val_loss

def plot_results(history):
    """Plots and saves training accuracy and loss graphs."""
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs_range = range(len(acc))

    plt.figure()
    plt.plot(epochs_range, acc, 'r', label='Training accuracy')
    plt.title('Training accuracy')
    plt.legend()
    plt.savefig('graph_training_acc.png')

    plt.figure()
    plt.plot(epochs_range, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    plt.savefig('graph_training_loss.png')
    plt.show()

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    # 1. Load Data
    print("Loading dataset...")
    try:
        file = "data/merged/cicids_merged.csv"
        df = pd.read_csv(file)
    except FileNotFoundError:
        print(f"Error: {file} not found.")
        exit()

    # 2. Initial Cleaning
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Label transformation
    df['Label'] = df['Label'].apply(lambda x: 'ATTACK' if x != 'BENIGN' else 'BENIGN')
    
    # Drop potential bias columns
    drop_cols = ['Flow ID', 'Destination Port', 'Timestamp']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Label encoding
    df['Bin Lebel'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    Y = df['Bin Lebel']
    df = df.drop(columns=['Label', 'Bin Lebel'])

    # 3. Type Conversion
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            print(f"Column {col} could not be converted to float.")

    # Column-wise Min/Max Analysis
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"Column {col}: Min = {df[col].min()}, Max = {df[col].max()}")

    # Manual Drop of Zero/Constant Variance columns
    to_drop_column = ['Bwd URG Flags', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 
                      'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 
                      'Bwd PSH Flags', 'Fwd Avg Bytes/Bulk']
    df = df.drop(columns=to_drop_column)

    # 4. Correlation Analysis
    perform_correlation_analysis(df)

    # 5. Feature Scaling
    try:
        x_data, y_data = df.astype(np.float64), Y.astype(np.float64)
    except KeyError:
        print("Error: Target column not found.")
        exit()

    # Normalize features
    x_data = preprocessing.normalize(x_data.values)

    # 6. Train and Test (Set to 12 epochs per user request)
    model, history, val_acc, val_loss = train_and_test(x_data, y_data, epochs=12)

    # 7. Plot Results
    plot_results(history)
