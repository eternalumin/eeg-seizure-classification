# EEG-Seizure-Classification


This project focuses on accurately classifying EEG data into six different classes, with a strong emphasis on improving the F1 scores for underrepresented classes (2 and 3). A combination of advanced data preprocessing, feature extraction, class balancing, and machine learning techniques is used to address the challenges of imbalanced data and enhance overall classification performance.

---

## Project Workflow

### 1. **Dataset Overview**  
The dataset contains EEG data labeled into six classes:  
- **0**: Healthy with eyes closed  
- **1**: Healthy with eyes open  
- **2**: Epileptic seizure activity  
- **3**: Brain activity during seizure-free intervals in the epileptic region  
- **4**: Brain activity in the tumor region  
- **5**: Other non-seizure activity  

The dataset is imbalanced, with fewer samples in some classes, particularly classes **2** and **3**, making it crucial to address these disparities for better model performance.

---

### 2. **Data Preprocessing**  
- **Missing Values**: Filled using the median of each feature column to maintain data consistency.  
- **Relabeling**: The target variable (`y`) was mapped directly to the desired class labels for clear differentiation.  
- **Stratified Sampling**: A balanced subset of 1000 samples was created using stratified sampling to ensure an equal representation of all classes during initial testing.  
- **Band-Pass Filtering**: EEG signals were denoised using a band-pass filter (0.5–40 Hz) to retain relevant frequency ranges for seizure detection.  
- **Feature Engineering**:  
  - **Time-Domain Features**: Original features from the dataset.  
  - **Frequency-Domain Features**: Extracted using Fast Fourier Transform (FFT), retaining only positive frequencies.  
  - **Feature Combination**: Time and frequency features were combined to provide a richer feature space for model training.

---

### 3. **Class Balancing**  
- **SMOTE (Synthetic Minority Oversampling Technique)**: Oversampling was applied to the training data to generate synthetic samples for underrepresented classes (2 and 3).  
- **Class Weights**: Adjusted in the model to penalize misclassification of underrepresented classes more heavily, ensuring a focus on these categories.

---

### 4. **Modeling**  
- **Support Vector Machine (SVM)**:  
  - Hyperparameter tuning was performed using `RandomizedSearchCV` to optimize the regularization parameter (`C`), kernel type, and kernel coefficient (`gamma`).  
  - Class weights were adjusted to focus on underrepresented classes.  

- **Random Forest**:  
  - Used as a complementary model with 100 estimators and adjusted class weights for better generalization.

- **Ensemble Learning**:  
  - A Voting Classifier was created to combine SVM and Random Forest models.  
  - **Soft Voting**: Probability-based weighting was used to make final predictions, leveraging the strengths of both models.

---

### 5. **Evaluation and Results**  
- **Metrics**:  
  - Confusion Matrix  
  - Classification Report (Precision, Recall, and F1-Score)  
  - Weighted F1-Score  
- **Focus on Classes 2 and 3**: Significant improvements were made to ensure the F1 scores for these classes exceeded 0.8.  

---

### 6. **Key Improvements**  
- Enhanced F1 scores for classes 2 and 3 through focused oversampling and hyperparameter tuning.  
- Leveraged ensemble learning for robust classification performance.  
- Combined time-domain and frequency-domain features for a richer representation of EEG data.  

---

## How to Use  
1. Clone the repository:  
   ```bash
   git clone <repository_url>
   cd epileptic-seizure-classification

