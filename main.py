#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 17:53:21 2026

@author: jamal
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# ---------------------------------------------------------
# DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------

def clean_dataset(df):
    """
    Replaces infinite values with NaN and drops rows with missing values.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df.astype(np.float64)

def perform_correlation_analysis(df, threshold=0.95):
    """
    Identifies columns with high correlation (for analysis purposes).
    """
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
    """
    Builds the Sequential Neural Network model.
    """
    model = Sequential()
    
    # Input Layer
    model.add(Dense(1024, activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    
    # Hidden Layers
    model.add(Dense(512, activation='relu', kernel_regularizer='l1'))
    model.add(BatchNormalization())
    
    model.add(Dense(256, activation='relu', kernel_regularizer='l1'))
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
    """
    Splits data, trains the model, and evaluates performance.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=42)
    
    # Reshape targets
    y_train = y_train.values.reshape(y_train.shape[0], 1)
    y_test = y_test.values.reshape(y_test.shape[0], 1)

    print(f"X train shape: {X_train.shape}, y train shape: {y_train.shape}")
      
    # Build model
    model = build_model(x_data.shape[1])
    
    # Train
    history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Evaluate
    pred = model.evaluate(X_test, y_test, verbose=1)
    
    print(model.metrics_names)
    val_acc, val_loss = pred[1] * 100, pred[0] * 100
    print('Test Data: accuracy: {:.2f}%: loss: {:.2f}'.format(val_acc, val_loss))

    return model, history, val_acc, val_loss

def plot_results(history):
    """
    Plots and saves training accuracy and loss graphs.
    """
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs_range = range(len(acc))

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs_range, acc, 'r', label='Training accuracy')
    plt.title('Training accuracy')
    plt.legend()
    plt.savefig('graph_training_acc.png')

    # Plot Loss
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
        df = pd.read_csv("data/ids_dataset.csv")
    except FileNotFoundError:
        print("Error: 'data/ids_dataset.csv' not found. Please check the path.")
        exit()

    # 2. Initial Cleaning
    df = df.drop(columns=['Label'])

    # Convert columns to float
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            print(f"Column {col} could not be converted to float.")

    # Column-wise Analysis (Min/Max)
    print("\n--- Column Analysis ---")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            # print(f"Column {col}: Min = {col_min}, Max = {col_max}") # Uncomment to view details

    # 3. Drop specified columns which have all column fill with zero or ones
    to_drop_column = ['Bwd URG Flags', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 
                      'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
    df = df.drop(columns=to_drop_column)

    # 4. Correlation Analysis (Logic preserved: calculates but does not drop based on list)
    perform_correlation_analysis(df)

    # 5. Prepare Data for Neural Network
    data = df
    data = clean_dataset(data)

    # Separate validation set (first 1000 rows) as per original logic
    validation_data = data[0:1000] 
    data = data[1000:]

    # Separate Features and Target
    # Note: Ensure 'Bin Lebel' exists in your CSV. Corrected typo 'Lebel' -> 'Label' if needed, 
    # but keeping 'Bin Lebel' as per your code.
    try:
        x_data, y_data = data.drop(columns=['Bin Lebel']), data['Bin Lebel']
    except KeyError:
        print("Error: Column 'Bin Lebel' not found. Please check your dataset column names.")
        exit()

    # Normalize
    x_data = preprocessing.normalize(x_data.values)

    # 6. Train and Test
    model, history, val_acc, val_loss = train_and_test(x_data, y_data, epochs=15)

    # 7. Plot Results
    plot_results(history)
