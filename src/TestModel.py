import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load dataset
data = pd.read_csv('../input/Dataset.csv')
label_columns_all = ['IT_B_Label', 'IT_M_Label', 'NST_B_Label', 'NST_M_Label']

# Encode labels to numerical values
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(data['NST_M_Label'])
y_one_hot = pd.get_dummies(y_encoded).values

data = data.drop(columns=label_columns_all)
# model = joblib.load('output/classifier_pipeline_python.joblib')
model =  pickle.load(open('../output/finalmodel.pk1', 'rb'))
print('using python')


# Evaluate the model on the test set
y_pred = model.predict(data)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_one_hot.argmax(axis=1), y_pred_classes)
conf_matrix = confusion_matrix(y_one_hot.argmax(axis=1), y_pred_classes)
print(f'Test Accuracy: {accuracy}')
print("Confusion Matrix:")
print(conf_matrix)