# import numpy as np
# import pandas as pd
# import shap
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, 
#     f1_score, roc_auc_score, classification_report
# )
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split

# def generate_synthetic_mental_health_data(num_samples=1000):
#     """
#     Generate synthetic mental health dataset
    
#     Columns:
#     - Symptoms: sleep_disturbance, mood_changes, anxiety_level, 
#                 energy_level, concentration_difficulty
#     - Target: mental health condition
#     """
#     np.random.seed(42)
    
#     # Generate symptom features
#     sleep_disturbance = np.random.uniform(0, 10, num_samples)
#     mood_changes = np.random.uniform(0, 10, num_samples)
#     anxiety_level = np.random.uniform(0, 10, num_samples)
#     energy_level = np.random.uniform(0, 10, num_samples)
#     concentration_difficulty = np.random.uniform(0, 10, num_samples)
    
#     # Define conditions based on symptom combinations
#     def assign_condition(sleep, mood, anxiety, energy, concentration):
#         if sleep > 7 and mood > 7 and anxiety > 7:
#             return 'Severe Depression'
#         elif sleep > 5 and anxiety > 5:
#             return 'Anxiety Disorder'
#         elif mood > 6 and energy < 3:
#             return 'Mild Depression'
#         elif concentration > 6:
#             return 'Attention Deficit'
#         else:
#             return 'No Significant Condition'
    
#     # Create DataFrame
#     df = pd.DataFrame({
#         'sleep_disturbance': sleep_disturbance,
#         'mood_changes': mood_changes,
#         'anxiety_level': anxiety_level,
#         'energy_level': energy_level,
#         'concentration_difficulty': concentration_difficulty
#     })
    
#     # Assign conditions
#     df['condition'] = df.apply(
#         lambda row: assign_condition(
#             row['sleep_disturbance'], 
#             row['mood_changes'], 
#             row['anxiety_level'], 
#             row['energy_level'], 
#             row['concentration_difficulty']
#         ), 
#         axis=1
#     )
    
#     return df

# class MentalHealthDataPreprocessor:
#     def __init__(self, datasets):
#         """
#         Initialize preprocessor with multiple datasets
        
#         :param datasets: List of dataset file paths or pandas DataFrames
#         """
#         self.datasets = datasets
#         self.label_encoder = LabelEncoder()
#         self.feature_scaler = StandardScaler()
    
#     def prepare_data(self, test_size=0.2, random_state=42):
#         """
#         Prepare data for model training
        
#         :param test_size: Proportion of test set
#         :param random_state: Random seed for reproducibility
#         :return: Training and testing datasets
#         """
#         # Combine datasets if multiple are provided
#         if len(self.datasets) > 1:
#             raw_data = pd.concat(self.datasets, ignore_index=True)
#         else:
#             raw_data = self.datasets[0]
        
#         # Encode target variable
#         raw_data['condition'] = self.label_encoder.fit_transform(raw_data['condition'])
        
#         # Separate features and target
#         X = raw_data.drop('condition', axis=1)
#         y = raw_data['condition']
        
#         # Scale features
#         X_scaled = self.feature_scaler.fit_transform(X)
        
#         # Split into train and test sets
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y, 
#             test_size=test_size, 
#             random_state=random_state, 
#             stratify=y
#         )
        
#         return X_train, X_test, y_train, y_test
    
#     def get_condition_mapping(self):
#         """
#         Get mapping of encoded conditions to original labels
#         """
#         return dict(enumerate(self.label_encoder.classes_))

# class MentalHealthPredictor:
#     def __init__(self, datasets):
#         """
#         Initialize mental health predictor
        
#         :param datasets: List of datasets for training
#         """
#         self.preprocessor = MentalHealthDataPreprocessor(datasets)
#         self.model = None
#         self.condition_mapping = None
    
#     def train_model(self, model_type='randomforest', **kwargs):
#         """
#         Train machine learning model
        
#         :param model_type: Type of model to train
#         :param kwargs: Additional model parameters
#         """
#         # Prepare data
#         X_train, X_test, y_train, y_test = self.preprocessor.prepare_data()
#         self.condition_mapping = self.preprocessor.get_condition_mapping()
        
#         # Model selection
#         if model_type == 'randomforest':
#             self.model = RandomForestClassifier(
#                 n_estimators=100, 
#                 random_state=42,
#                 **kwargs
#             )
        
#         # Train model
#         self.model.fit(X_train, y_train)
        
#         # Evaluate model
#         y_pred = self.model.predict(X_test)
        
#         # Print evaluation metrics
#         print("Model Evaluation Metrics:")
#         print(classification_report(y_test, y_pred, 
#             target_names=list(self.condition_mapping.values())))
        
#         # Calculate additional metrics
#         print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
#         print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
#         print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
#         print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")
    
#     def predict(self, symptoms):
#         """
#         Predict mental health condition
        
#         :param symptoms: List or array of symptom features
#         :return: Predicted condition and probability
#         """
#         if self.model is None:
#             raise ValueError("Model not trained. Call train_model() first.")
        
#         # Scale input symptoms
#         scaled_symptoms = self.preprocessor.feature_scaler.transform([symptoms])
        
#         # Predict
#         prediction = self.model.predict(scaled_symptoms)
#         probabilities = self.model.predict_proba(scaled_symptoms)
        
#         # Map prediction to original condition
#         predicted_condition = self.condition_mapping[prediction[0]]
#         condition_prob = max(probabilities[0])
        
#         return predicted_condition, condition_prob
    
#     def explain_prediction(self, symptoms):
#         """
#         Generate model interpretation using SHAP
        
#         :param symptoms: List or array of symptom features
#         :return: SHAP explanation
#         """
#         explainer = shap.TreeExplainer(self.model)
#         shap_values = explainer.shap_values(symptoms)
        
#         return shap_values
    
#     def save_model(self, filepath='mental_health_model.joblib'):
#         """
#         Save trained model
        
#         :param filepath: Path to save model
#         """
#         joblib.dump({
#             'model': self.model,
#             'condition_mapping': self.condition_mapping,
#             'feature_scaler': self.preprocessor.feature_scaler
#         }, filepath)
    
#     def load_model(self, filepath='mental_health_model.joblib'):
#         """
#         Load pre-trained model
        
#         :param filepath: Path to load model from
#         """
#         loaded_data = joblib.load(filepath)
#         self.model = loaded_data['model']
#         self.condition_mapping = loaded_data['condition_mapping']
#         self.preprocessor.feature_scaler = loaded_data['feature_scaler']

# # Main execution
# if __name__ == '__main__':
#     # Generate synthetic dataset
#     dataset = generate_synthetic_mental_health_data()
    
#     # Initialize predictor
#     predictor = MentalHealthPredictor([dataset])
    
#     # Train model
#     predictor.train_model()
    
#     # Save model
#     predictor.save_model('mental_health_model.joblib')
    
#     print("Model trained and saved successfully!")
import numpy as np
import pandas as pd
import shap
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def generate_synthetic_mental_health_data(num_samples=1000):
    """
    Generate synthetic mental health dataset
    
    Columns:
    - Symptoms: sleep_disturbance, mood_changes, anxiety_level, 
                energy_level, concentration_difficulty
    - Target: mental health condition
    """
    np.random.seed(42)
    
    # Generate symptom features
    sleep_disturbance = np.random.uniform(0, 10, num_samples)
    mood_changes = np.random.uniform(0, 10, num_samples)
    anxiety_level = np.random.uniform(0, 10, num_samples)
    energy_level = np.random.uniform(0, 10, num_samples)
    concentration_difficulty = np.random.uniform(0, 10, num_samples)
    
    # Define conditions based on symptom combinations
    def assign_condition(sleep, mood, anxiety, energy, concentration):
        if anxiety > 7 and mood > 7 and sleep > 7:
            return 'Severe Anxiety'
        elif anxiety > 6 and concentration > 6:
            return 'Anxiety Disorder'
        elif mood < 3 and energy < 3:
            return 'Mild Depression'
        elif concentration > 6:
            return 'Attention Deficit'
        else:
            return 'No Significant Condition'
    
    # Create DataFrame
    df = pd.DataFrame({
        'sleep_disturbance': sleep_disturbance,
        'mood_changes': mood_changes,
        'anxiety_level': anxiety_level,
        'energy_level': energy_level,
        'concentration_difficulty': concentration_difficulty
    })
    
    # Assign conditions
    df['condition'] = df.apply(
        lambda row: assign_condition(
            row['sleep_disturbance'], 
            row['mood_changes'], 
            row['anxiety_level'], 
            row['energy_level'], 
            row['concentration_difficulty']
        ), 
        axis=1
    )
    
    return df

class MentalHealthDataPreprocessor:
    def __init__(self, datasets):
        """
        Initialize preprocessor with multiple datasets
        
        :param datasets: List of dataset file paths or pandas DataFrames
        """
        self.datasets = datasets
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.feature_names = ['sleep_disturbance', 'mood_changes', 'anxiety_level', 
                               'energy_level', 'concentration_difficulty']
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data for model training
        
        :param test_size: Proportion of test set
        :param random_state: Random seed for reproducibility
        :return: Training and testing datasets
        """
        # Combine datasets if multiple are provided
        if len(self.datasets) > 1:
            raw_data = pd.concat(self.datasets, ignore_index=True)
        else:
            raw_data = self.datasets[0]
        
        # Encode target variable
        raw_data['condition'] = self.label_encoder.fit_transform(raw_data['condition'])
        
        # Separate features and target
        X = raw_data[self.feature_names]
        y = raw_data['condition']
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_condition_mapping(self):
        """
        Get mapping of encoded conditions to original labels
        """
        return dict(enumerate(self.label_encoder.classes_))

class MentalHealthPredictor:
    def __init__(self, datasets):
        """
        Initialize mental health predictor
        
        :param datasets: List of datasets for training
        """
        self.preprocessor = MentalHealthDataPreprocessor(datasets)
        self.model = None
        self.condition_mapping = None
        self.feature_names = self.preprocessor.feature_names
    
    def train_model(self, model_type='randomforest', **kwargs):
        """
        Train machine learning model
        
        :param model_type: Type of model to train
        :param kwargs: Additional model parameters
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_data()
        self.condition_mapping = self.preprocessor.get_condition_mapping()
        
        # Model selection
        if model_type == 'randomforest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                **kwargs
            )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Print evaluation metrics
        print("Model Evaluation Metrics:")
        print(classification_report(y_test, y_pred, 
            target_names=list(self.condition_mapping.values())))
        
        # Calculate additional metrics
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
        print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
        print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")
    
    def predict(self, symptoms):
        """
        Predict mental health condition
        
        :param symptoms: List or array of symptom features
        :return: Predicted condition and probability
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Convert symptoms to DataFrame with correct column names
        symptoms_df = pd.DataFrame([symptoms], columns=self.feature_names)
        
        # Scale input symptoms
        scaled_symptoms = self.preprocessor.feature_scaler.transform(symptoms_df)
        
        # Predict
        prediction = self.model.predict(scaled_symptoms)
        probabilities = self.model.predict_proba(scaled_symptoms)
        
        # Map prediction to original condition
        predicted_condition = self.condition_mapping[prediction[0]]
        condition_prob = max(probabilities[0])
        
        return predicted_condition, condition_prob
    
    def explain_prediction(self, symptoms):
        """
        Generate model interpretation using SHAP
        
        :param symptoms: List or array of symptom features
        :return: SHAP explanation
        """
        # Convert symptoms to DataFrame with correct column names
        symptoms_df = pd.DataFrame([symptoms], columns=self.feature_names)
        
        # Scale input symptoms
        scaled_symptoms = self.preprocessor.feature_scaler.transform(symptoms_df)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(scaled_symptoms)
        
        # If multiclass, get the SHAP values for the predicted class
        if isinstance(shap_values, list):
            predicted_class = self.model.predict(scaled_symptoms)[0]
            shap_values = shap_values[predicted_class]
        
        # Create a dictionary of feature impacts
        feature_impacts = {}
        for i, feature in enumerate(self.feature_names):
            feature_impacts[feature] = float(shap_values[0][i])
        
        return feature_impacts
    
    def save_model(self, filepath='mental_health_model.joblib'):
        """
        Save trained model
        
        :param filepath: Path to save model
        """
        joblib.dump({
            'model': self.model,
            'condition_mapping': self.condition_mapping,
            'feature_scaler': self.preprocessor.feature_scaler,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_model(self, filepath='mental_health_model.joblib'):
        """
        Load pre-trained model
        
        :param filepath: Path to load model from
        """
        loaded_data = joblib.load(filepath)
        self.model = loaded_data['model']
        self.condition_mapping = loaded_data['condition_mapping']
        self.preprocessor.feature_scaler = loaded_data['feature_scaler']
        self.feature_names = loaded_data.get('feature_names', 
            ['sleep_disturbance', 'mood_changes', 'anxiety_level', 
             'energy_level', 'concentration_difficulty'])

# Main execution
if __name__ == '__main__':
    # Generate synthetic dataset
    dataset = generate_synthetic_mental_health_data()
    
    # Initialize predictor
    predictor = MentalHealthPredictor([dataset])
    
    # Train model
    predictor.train_model()
    
    # Save model
    predictor.save_model('mental_health_model.joblib')
    
    print("Model trained and saved successfully!")