import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class MentalHealthDataPreprocessor:
    def __init__(self, datasets):
        """
        Initialize preprocessor with multiple datasets
        
        :param datasets: List of dataset file paths or pandas DataFrames
        """
        self.datasets = datasets
        self.processed_data = None
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
    
    def load_datasets(self):
        """
        Load and concatenate multiple datasets
        """
        combined_data = []
        for dataset in self.datasets:
            if isinstance(dataset, str):
                df = pd.read_csv(dataset)
            else:
                df = dataset
            combined_data.append(df)
        
        return pd.concat(combined_data, ignore_index=True)
    
    def clean_data(self, data):
        """
        Clean and preprocess the data
        
        :param data: pandas DataFrame
        :return: Cleaned DataFrame
        """
        # Remove duplicate rows
        data.drop_duplicates(inplace=True)
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with median
        numeric_imputer = SimpleImputer(strategy='median')
        data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
        
        # Impute categorical columns with mode
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])
        
        return data
    
    def encode_features(self, data, target_column='condition'):
        """
        Encode categorical features and target variable
        
        :param data: pandas DataFrame
        :param target_column: Name of the target column
        :return: Processed DataFrame with encoded features
        """
        # Encode categorical columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != target_column:
                data[col] = self.label_encoder.fit_transform(data[col].astype(str))
        
        # Encode target variable
        data[target_column] = self.label_encoder.fit_transform(data[target_column])
        
        return data
    
    def feature_engineering(self, data):
        """
        Perform feature engineering
        
        :param data: pandas DataFrame
        :return: DataFrame with engineered features
        """
        # Create interaction features
        symptom_columns = [col for col in data.columns if col not in ['condition']]
        for i in range(len(symptom_columns)):
            for j in range(i+1, len(symptom_columns)):
                data[f'interaction_{symptom_columns[i]}_{symptom_columns[j]}'] = \
                    data[symptom_columns[i]] * data[symptom_columns[j]]
        
        return data
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data for model training
        
        :param test_size: Proportion of test set
        :param random_state: Random seed for reproducibility
        :return: Training and testing datasets
        """
        # Load and clean data
        raw_data = self.load_datasets()
        cleaned_data = self.clean_data(raw_data)
        
        # Feature engineering and encoding
        processed_data = self.encode_features(cleaned_data)
        processed_data = self.feature_engineering(processed_data)
        
        # Split features and target
        X = processed_data.drop('condition', axis=1)
        y = processed_data['condition']
        
        # Scale numerical features
        X_scaled = self.feature_scaler.fit_transform(X)
        
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

# Main execution for generating preprocessed dataset
if __name__ == '__main__':
    # Load the synthetic dataset
    preprocessor = MentalHealthDataPreprocessor(['mental_health_dataset.csv'])
    
    # Prepare the data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    
    # Get condition mapping for reference
    condition_mapping = preprocessor.get_condition_mapping()
    print("Condition Mapping:", condition_mapping)
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)