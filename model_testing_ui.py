# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from model_training import MentalHealthPredictor, generate_synthetic_mental_health_data

# def create_symptom_inputs(symptoms_list):
#     """
#     Create input fields for symptoms
#     """
#     symptom_values = {}
#     for symptom in symptoms_list:
#         symptom_values[symptom] = st.slider(
#             f"{symptom.replace('_', ' ').title()}",
#             min_value=0.0,
#             max_value=10.0,
#             value=0.0,
#             step=0.5
#         )
#     return list(symptom_values.values())

# def main():
#     st.title("Mental Health Self-Analysis Model")
    
#     # Ensure model exists
#     try:
#         # Load or train model if not exists
#         predictor = MentalHealthPredictor([generate_synthetic_mental_health_data()])
        
#         # Try to load existing model, if fails, train and save
#         try:
#             predictor.load_model('mental_health_model.joblib')
#         except FileNotFoundError:
#             predictor.train_model()
#             predictor.save_model('mental_health_model.joblib')
#     except Exception as e:
#         st.error(f"Error initializing model: {e}")
#         return

#     # Define symptoms list (this should match your training data)
#     symptoms_list = [
#         'sleep_disturbance', 
#         'mood_changes', 
#         'anxiety_level', 
#         'energy_level', 
#         'concentration_difficulty'
#     ]
    
#     st.write("Rate your symptoms on a scale of 0-10:")
    
#     # Create symptom input sliders
#     symptoms = create_symptom_inputs(symptoms_list)
    
#     # Prediction button
#     if st.button("Analyze Mental Health"):
#         # Predict condition
#         predicted_condition, probability = predictor.predict(symptoms)
        
#         # Generate explanation
#         try:
#             shap_explanation = predictor.explain_prediction([symptoms])
#         except Exception as e:
#             st.warning(f"Could not generate SHAP explanation: {e}")
#             shap_explanation = [np.zeros(len(symptoms))]
        
#         # Display results
#         st.subheader("Results")
#         st.write(f"Predicted Condition: **{predicted_condition}**")
#         st.write(f"Confidence: {probability * 100:.2f}%")
        
#         # Visualization of symptom impact
#         st.subheader("Symptom Impact Explanation")
        
#         # Create a bar plot of symptom impacts
#         plt.figure(figsize=(10, 6))
#         plt.bar(symptoms_list, shap_explanation[0])
#         plt.title("Symptom Impact on Prediction")
#         plt.xlabel("Symptoms")
#         plt.ylabel("Impact Score")
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
        
#         # Display the plot
#         st.pyplot(plt)
        
#         # Detailed explanation of symptoms
#         st.subheader("Symptom Breakdown")
#         for symptom, impact, value in zip(symptoms_list, shap_explanation[0], symptoms):
#             st.write(f"{symptom.replace('_', ' ').title()}: ")
#             st.write(f"  - Value: {value:.2f}")
#             st.write(f"  - Impact: {impact:.4f}")

# if __name__ == '__main__':
#     main()

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model_training import MentalHealthPredictor, generate_synthetic_mental_health_data

def create_symptom_inputs(symptoms_list):
    """
    Create input fields for symptoms
    """
    symptom_values = {}
    for symptom in symptoms_list:
        symptom_values[symptom] = st.slider(
            f"{symptom.replace('_', ' ').title()}",
            min_value=0.0,
            max_value=10.0,
            value=5.0,  # Default to middle value
            step=0.5
        )
    return list(symptom_values.values())

def get_symptom_impact(predictor, symptoms):
    """
    Generate symptom impact explanation with error handling
    """
    try:
        # Use the new explain_prediction method
        feature_impacts = predictor.explain_prediction(symptoms)
        return feature_impacts
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {e}")
        # Return a default dictionary with zero impacts
        return {
            'sleep_disturbance': 0.0,
            'mood_changes': 0.0,
            'anxiety_level': 0.0,
            'energy_level': 0.0,
            'concentration_difficulty': 0.0
        }

def main():
    st.title("Mental Health Self-Analysis Model")
    
    # Ensure model exists
    try:
        # Generate synthetic dataset
        dataset = generate_synthetic_mental_health_data()
        
        # Initialize predictor
        predictor = MentalHealthPredictor([dataset])
        
        # Try to load existing model, if fails, train and save
        try:
            predictor.load_model('mental_health_model.joblib')
        except FileNotFoundError:
            predictor.train_model()
            predictor.save_model('mental_health_model.joblib')
    
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return 
    
    # Define symptoms list (this should match your training data)
    symptoms_list = [
        'sleep_disturbance', 
        'mood_changes', 
        'anxiety_level', 
        'energy_level', 
        'concentration_difficulty'
    ]
    
    st.write("Rate your symptoms on a scale of 0-10:")
    
    # Create symptom input sliders
    symptoms = create_symptom_inputs(symptoms_list)
    
    # Prediction button
    if st.button("Analyze Mental Health"):
        # Predict condition
        predicted_condition, probability = predictor.predict(symptoms)
        
        # Generate explanation
        shap_explanation = get_symptom_impact(predictor, symptoms)
        
        # Display results
        st.subheader("Results")
        st.write(f"Predicted Condition: **{predicted_condition}**")
        st.write(f"Confidence: {probability * 100:.2f}%")
        
        # Visualization of symptom impact
        st.subheader("Symptom Impact Explanation")
        
        # Create a bar plot of symptom impacts
        plt.figure(figsize=(10, 6))
        impacts = [shap_explanation[symptom] for symptom in symptoms_list]
        plt.bar(symptoms_list, impacts)
        plt.title("Symptom Impact on Prediction")
        plt.xlabel("Symptoms")
        plt.ylabel("SHAP Impact Score")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Detailed explanation of symptoms
        st.subheader("Symptom Breakdown")
        for symptom, value in zip(symptoms_list, symptoms):
            st.write(f"{symptom.replace('_', ' ').title()}: ")
            st.write(f"  - Value: {value:.2f}")
            st.write(f"  - Impact: {shap_explanation[symptom]:.4f}")

if __name__ == '__main__':
    main()