
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

# 1. Recreate your handwritten data
data = {
    'Outlook': ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Overcast', 'Overcast', 'Sunny'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'PlayGolf': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# 2. Encode categorical text to numerical values for the model
X = df[['Outlook', 'Temperature', 'Humidity', 'Windy']]
y = df['PlayGolf']

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# 3. Train the Naive Bayes Classifier
model = CategoricalNB()
model.fit(X_encoded, y)

# 4. Predict a new future day
# Example: Outlook='Sunny', Temperature='Cool', Humidity='High', Windy=True
new_day = pd.DataFrame([['Rainy', 'Hot', 'Normal', False]], columns=['Outlook', 'Temperature', 'Humidity', 'Windy'])
new_day_encoded = encoder.transform(new_day)

prediction = model.predict(new_day_encoded)
probabilities = model.predict_proba(new_day_encoded)

print(f"Prediction for future weather: Play Golf? -> {prediction[0]}")
print(f"Probabilities [No, Yes]: {probabilities[0]}")