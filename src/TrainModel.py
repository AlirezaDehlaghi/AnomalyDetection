from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib
from ColumnDropperTransformer import ColumnDropperTransformer
from sklearn.compose import ColumnTransformer
from scikeras.wrappers import KerasClassifier
from ModelCreator import create_model

input_dataset = '../input/Dataset.csv'
output_model_name = '../output/IDS.joblib'

# Load your dataset
# Assuming your dataset is in a CSV file named 'Dataset.csv'
data = pd.read_csv(input_dataset)

# Preprocess Labels
label_column = ['NST_M_Label']
label_columns_all = ['IT_B_Label', 'IT_M_Label', 'NST_B_Label', 'NST_M_Label']

# Encode labels to numerical values
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(data[label_column])
class_mapping = dict(zip(label_encoder_y.classes_, label_encoder_y.transform(label_encoder_y.classes_)))
print(class_mapping)

# Convert labels to one-hot encoding
y_one_hot = pd.get_dummies(y_encoded).values

# Drop all label columns
data = data.drop(columns=label_columns_all)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, y_one_hot, test_size=0.2, random_state=42, stratify=y_encoded)

# Preprocess Data
unused_columns = ['sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'startDate', 'endDate', 'start', 'end',
                  'startOffset', 'endOffset']
categorical_columns = ['protocol']

categorial_transformer = ColumnTransformer(
    [
        # Due to some indexing problem categorical_columns = ['protocol'] transfered via [0] index of 0
        ('ohe_encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [0])
    ], remainder='passthrough'
)

preprocess = Pipeline([
    ('drop_unused', ColumnDropperTransformer(unused_columns)),
    ('fill_missing', SimpleImputer(strategy='constant', fill_value=0)),
    ('transform_categorial', categorial_transformer),
    ('normalizer', StandardScaler())
])

# Preprocessing pipeline
X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)

model = KerasClassifier(model=create_model, input_dim=X_train.shape[1], output_dim=y_train.shape[1], epochs=10,
                        batch_size=32, verbose=2)
param_grid = {
    'model__hidden_layer_size': [32, 64, 128],
    'model__activation': ['relu', 'tanh', 'sigmoid'],
    # 'model__alpha': [0.0001, 0.001, 0.01]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2)
grid_result = grid_search.fit(X_train, y_train)

print(f"Best Hyperparameters: {grid_result.best_params_}")
best_model = grid_result.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Inverse transform one-hot encoded predictions to original labels
# y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred_classes)
print(f'Test Accuracy: {accuracy}')

# Assuming 'model' is your Keras model
predict_model_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('prediction', best_model)
])
joblib.dump(predict_model_pipeline, output_model_name)
