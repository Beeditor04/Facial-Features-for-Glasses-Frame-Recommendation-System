from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Load the model, encoder, and scaler
model = joblib.load('nearest_neighbors_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# Function to preprocess new item data
def preprocess_item(item, encoder, scaler):
    item_df = pd.DataFrame([item])
    categorical_features = ['gender', 'face_shape', 'skin']
    encoded_item = encoder.transform(item_df[categorical_features]).toarray()
    scaled_numerical_item = scaler.transform(item_df[['width']])
    return np.hstack([encoded_item, scaled_numerical_item])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_data = request.json
    new_user_features = preprocess_item(user_data, encoder, scaler).reshape(1, -1)
    n_neighbors = 10  # Adjust this number to get more recommendations
    distances, indices = model.kneighbors(new_user_features, n_neighbors=n_neighbors)
    
    matches = df.iloc[indices[0]].to_dict(orient='records')
    
    return render_template('result.html', matches=matches)

if __name__ == '__main__':
    # Load the dataset globally
    global df
    df = pd.read_csv('eyewear_data_full_v1.csv')
    
    app.run(debug=True)
