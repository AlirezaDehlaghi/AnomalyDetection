import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# Load your dataset
# Assuming your dataset is in a CSV file named 'network_data.csv'
data = pd.read_csv('input/Dataset.csv')

# Drop columns you want to remove
# Assuming 'column_to_remove_1' and 'column_to_remove_2' are the columns you want to drop
columns_to_remove = ['sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'startDate', 'endDate', 'start', 'end', 'startOffset', 'endOffset']
data = data.drop(columns=columns_to_remove)

# Fill missing values with 0
data = data.fillna(0)

# Separate features and labels
label_columns_all = ['IT_B_Label', 'IT_M_Label', 'NST_B_Label', 'NST_M_Label']
label_column = ['NST_M_Label']
X = data.drop(label_columns_all, axis=1)  # Replace 'anomaly_label_column' with your actual label column
y = data[label_column]

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert labels to one-hot encoding
y_one_hot = pd.get_dummies(y_encoded).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42, stratify=y_encoded)

# Standardize features using Min-Max normalization
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Define the ANN model
def create_model(input_dim, output_dim, hidden_layer_size=64, activation='relu'):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=input_dim, activation=activation))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier for use with GridSearchCV
model = KerasClassifier(build_fn=create_model, input_dim=X_train_normalized.shape[1], output_dim=y_one_hot.shape[1], epochs=10, batch_size=32, verbose=0)

# Define hyperparameters to tune
param_grid = {
    'hidden_layer_size': [32, 64, 128],
    'activation': ['relu', 'tanh', 'sigmoid']
}

# Initialize StratifiedKFold for cross-validation
n_splits = 5  # You can adjust the number of folds
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Use GridSearchCV for hyperparameter tuning
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kf)
grid_result = grid.fit(X_train_normalized, y_train)

# Print the best hyperparameters
print(f"Best Hyperparameters: {grid_result.best_params_}")

# Get the best model
best_model = grid_result.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test_normalized)
y_pred_classes = np.argmax(y_pred, axis=1)

# Inverse transform one-hot encoded predictions to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred_classes)
print(f'Test Accuracy: {accuracy}')
