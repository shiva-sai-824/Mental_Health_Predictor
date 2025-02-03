import sys
import numpy as np
import json
from model_training import MentalHealthPredictor, generate_synthetic_mental_health_data

def load_symptoms_from_args():
    """
    Load symptoms from command-line arguments
    """
    if len(sys.argv) < 2:
        print("Usage: python predict_mental_health.py [symptom1] [symptom2] ...")
        sys.exit(1)
    
    # Convert arguments to float
    try:
        symptoms = [float(arg) for arg in sys.argv[1:]]
        
        # Ensure exactly 5 symptoms are provided
        if len(symptoms) != 5:
            print("Error: Exactly 5 symptoms are required.")
            print("Order: sleep_disturbance, mood_changes, anxiety_level, energy_level, concentration_difficulty")
            sys.exit(1)
        
        return symptoms
    except ValueError:
        print("Error: All arguments must be numeric values.")
        sys.exit(1)

def main():
    # If model doesn't exist, train and save it first
    try:
        # Generate synthetic dataset
        dataset = generate_synthetic_mental_health_data()
        
        # Initialize predictor
        predictor = MentalHealthPredictor([dataset])
        
        # Train model
        predictor.train_model()
        
        # Save model
        predictor.save_model('mental_health_model.joblib')
    except Exception as e:
        print(f"Error training model: {e}")
        sys.exit(1)
    
    # Get symptoms from command line
    symptoms = load_symptoms_from_args()
    
    # Predict mental health condition
    predicted_condition, probability = predictor.predict(symptoms)
    
    # Prepare result dictionary
    result = {
        'predicted_condition': predicted_condition,
        'confidence': float(probability),
        'symptoms': {
            'sleep_disturbance': symptoms[0],
            'mood_changes': symptoms[1],
            'anxiety_level': symptoms[2],
            'energy_level': symptoms[3],
            'concentration_difficulty': symptoms[4]
        }
    }
    
    # Output result as JSON
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()