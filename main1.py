# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import seaborn as sns
import joblib

# Step 1: Load and preprocess the dataset
data = pd.read_csv('dataset1.csv')  # Replace with your dataset path

# Remove unwanted columns, if any
if 'Unnamed' in data.columns:
    data = data.drop(columns=['Unnamed'])

# Fill missing values with the median of each column
data.fillna(data.median(), inplace=True)

# Step 2: Relabel the target variable to your required mapping
# Mapping the classes according to your specific requirement
relabel_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
data['y'] = data['y'].map(relabel_map)

# Separate features (X) and target (y)
X = data.drop(columns=['y'])
y = data['y']

# Step 3: Perform stratified sampling to select a smaller but representative subset
sample_size = 1000  # Adjust the sample size as needed
data_sampled = data.groupby('y', group_keys=False).apply(
    lambda x: x.sample(int(sample_size / len(data['y'].unique())), random_state=42)
)

# Update X and y after sampling
X_sampled = data_sampled.drop(columns=['y'])
y_sampled = data_sampled['y']

# Step 4: Apply band-pass filtering for denoising
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

fs = 128  # Modify based on your dataset
X_filtered = bandpass_filter(X_sampled.values, lowcut=0.5, highcut=40, fs=fs)

# Step 5: Extract frequency-domain features using FFT
def extract_frequency_features(data, fs):
    fft_data = np.fft.fft(data)
    magnitude = np.abs(fft_data)
    return magnitude[:len(magnitude)//2]  # Retain positive frequencies

X_frequency_features = np.array([extract_frequency_features(row, fs=fs) for row in X_filtered])

# Combine time-domain and frequency-domain features
X_combined = np.hstack([X_filtered, X_frequency_features])

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_sampled, test_size=0.2, stratify=y_sampled, random_state=42)

# Step 7: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 8: Define and tune the SVM model using RandomizedSearchCV and cross-validation

# SVM pipeline
svm_pipeline = Pipeline([
    ('scaler', MinMaxScaler()), 
    ('svc', SVC(kernel='rbf', class_weight='balanced', random_state=42))
])

# Define parameter grid for RandomizedSearchCV
param_dist = {
    'svc__C': [0.1, 1, 10, 100, 1000],
    'svc__gamma': [0.1, 1, 10, 0.001, 0.0001],
    'svc__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

# RandomizedSearchCV for hyperparameter optimization
random_search = RandomizedSearchCV(svm_pipeline, param_distributions=param_dist, n_iter=50, cv=3, scoring='f1_weighted', verbose=1, random_state=42)
random_search.fit(X_train_res, y_train_res)

# Best model from RandomizedSearchCV
best_svm_model = random_search.best_estimator_

# Step 9: Try Ensemble Method - Combine SVM with Random Forest for better results

# Random Forest model with class_weight set to balanced for handling class imbalance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Voting Classifier (ensemble of SVM and Random Forest)
ensemble_model = VotingClassifier(estimators=[('svm', best_svm_model), ('rf', rf_model)], voting='hard')

# Fit the ensemble model
ensemble_model.fit(X_train_res, y_train_res)

# Step 10: Evaluate the model
y_pred = ensemble_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_sampled), yticklabels=np.unique(y_sampled))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification Report and Weighted F1 Score
print("Classification Report:")
print(classification_report(y_test, y_pred))
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Weighted F1 Score: {f1:.2f}")

# Step 11: Save the trained model (optional)
joblib.dump(ensemble_model, 'ensemble_model.pkl')
print("Optimized ensemble model saved as 'ensemble_model.pkl'")
