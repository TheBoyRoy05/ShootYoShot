import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

# Get the absolute path to the data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to reach root
data_path = os.path.join(root_dir, 'data', 'player_info.csv')

df = pd.read_csv(data_path)

X = df[['Height_inches', 'Weight']]
y = df['Archetype']

le = LabelEncoder()
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2
)

rf = RandomForestClassifier(max_depth=5, min_samples_split=20, n_estimators=35)
rf.fit(X_train, y_train)


def predict_position(height, weight, is_male = True):
    avg_male_height, avg_male_weight = 69.1, 199.8
    avg_female_height, avg_female_weight = 63.7, 170.8

    std_male_height, std_male_weight = 2.9, 40.8
    std_female_height, std_female_weight = 2.7, 40.5

    if is_male:
        height_scaled = (height - avg_male_height) / std_male_height
        weight_scaled = (weight - avg_male_weight) / std_male_weight
    else:
        height_scaled = (height - avg_female_height) / std_female_height
        weight_scaled = (weight - avg_female_weight) / std_female_weight

    user_scaled = np.array([[height_scaled, weight_scaled]])
    encoded_pred = rf.predict(user_scaled)
    decoded_pred = le.inverse_transform(encoded_pred)[0]
    return decoded_pred

# predict_player_type(76, 100, False)