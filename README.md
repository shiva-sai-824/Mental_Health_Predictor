# Mental Health Predictor 🧠💡

## 🌟 Project Overview

A comprehensive machine learning application designed to provide mental health condition predictions based on self-reported symptoms. This tool leverages advanced data science techniques to offer insights into potential mental health conditions.

## 🚨 Disclaimer

**Important**: This is a screening tool and NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for accurate diagnosis and personalized treatment.

## ✨ Features

- Machine learning-powered mental health condition prediction
- Synthetic data generation for model training
- Advanced feature analysis using SHAP (SHapley Additive exPlanations)
- Interactive Streamlit user interface
- Robust model evaluation and interpretation

## 🛠 Technology Stack

- Python 3.8+
- Scikit-learn
- Pandas
- NumPy
- SHAP
- Streamlit
- Joblib

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/shiva-sai-824/Mental_Health_Predictor.git
cd Mental_Health_Predictor
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train the Model

```bash
python model_training.py
```

This script will:
- Generate a synthetic mental health dataset
- Train a Random Forest Classifier
- Save the trained model as `mental_health_model.joblib`

### 2. Run Streamlit User Interface

```bash
streamlit run model_testing_ui.py
```

## 📊 Model Characteristics

- **Algorithm**: Random Forest Classifier
- **Input Features**:
  1. Sleep Disturbance
  2. Mood Changes
  3. Anxiety Level
  4. Energy Level
  5. Concentration Difficulty

- **Output**: Predicted Mental Health Condition
- **Model Interpretability**: SHAP Value Analysis

## 📈 Performance Metrics

The model provides comprehensive evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score

## 🔍 How It Works

1. Generate synthetic mental health data
2. Preprocess and scale features
3. Train a Random Forest Classifier
4. Use SHAP for feature importance and model interpretation
5. Provide user-friendly prediction interface

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## ⚠️ Limitations

- Uses synthetic data for training
- Not a substitute for professional medical advice
- Predictions are probabilistic and should be interpreted cautiously

## 📜 License

This project is open-source. Check the `LICENSE` file for details.

## 📞 Contact

- **Author**: Shiva Sai
- **GitHub**: [shiva-sai-824](https://github.com/shiva-sai-824)
- **Project Link**: [Mental_Health_Predictor](https://github.com/shiva-sai-824/Mental_Health_Predictor)

## 🙏 Acknowledgements

- Scikit-learn
- SHAP
- Streamlit
- Open-source community
