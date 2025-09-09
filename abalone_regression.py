import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense
from joblib import dump

# 1. Load Data
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
data = pd.read_csv('abalone.data.csv', names=column_names)

# Convert 'Rings' to a numeric type
data['Rings'] = pd.to_numeric(data['Rings'], errors='coerce')
data.dropna(subset=['Rings'], inplace=True)
data['Rings'] = data['Rings'].astype(int)

data['Age'] = data['Rings'] + 1.5

# 2. Pre-processing
X = data.drop(['Rings', 'Age'], axis=1)
y = data['Rings']

categorical_features = ['Sex']
numerical_features = X.columns.drop(categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor and transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 3. Build Model
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(X_train_processed.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# 4. Compile and Train
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train_processed, y_train, epochs=200, batch_size=32, verbose=0)

# 5. Evaluate
loss, mae = model.evaluate(X_test_processed, y_test, verbose=0)
print(f'Mean Absolute Error on Test Set: {mae:.2f} rings')

# 6. Save Model and Preprocessor
model.save('abalone_model.keras')
dump(preprocessor, 'abalone_preprocessor.joblib')
print("\nModel and preprocessor saved successfully.")

# 7. Predict
predictions = model.predict(X_test_processed[:5])
predicted_ages = predictions + 1.5

print("\n--- Predictions ---")
for i in range(len(predictions)):
    print(f"Predicted Rings: {predictions[i][0]:.2f}, Predicted Age: {predicted_ages[i][0]:.2f}")
    print(f"Actual Rings: {y_test.iloc[i]}, Actual Age: {y_test.iloc[i] + 1.5}")
    print("-" * 20)
