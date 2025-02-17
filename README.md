Live Demo:https://huggingface.co/spaces/anusomepalli/diseasedetection
# AI Companion for Personalized Healthcare

## Overview
This project is a Streamlit-based Machine 
Learning web application that predicts diseases based on user-provided symptoms and provides personalized healthcare recommendations. The app uses a trained machine learning model (Random Forest) for disease prediction.

## Features
- Select symptoms from a predefined list.
- Predicts the most probable disease based on user input.
- Provides relevant information, including:
  - Disease description
  - Prescription recommendations
  - Precautionary measures
  - Suggested diet plans
  - Recommended workouts
- Interactive and user-friendly UI with Streamlit.

## Technologies Used
- Python
- Streamlit
- Scikit-Learn
- NumPy
- Pandas
- Pickle (for model storage)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Anugna15/Disease-Detection-and-Healthcare-recommendations-using-symptoms.git
   cd Disease-Detection-and-Healthcare-recommendations-using-symptoms
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the necessary dataset files (`diseases.csv`, `symptoms.csv`) are available in the project directory.

## Running the Application
1. Train the model and save it as `random_forest.pkl` if not already done.
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the generated local URL in your browser.

## File Structure
```
📂 ai-healthcare
├── app.py               # Streamlit application script
├── diseases.csv         # Dataset containing diseases and symptoms
├── symptoms.csv         # Dataset with disease suggestions
├── random_forest.pkl    # Pre-trained machine learning model
├── requirements.txt     # Dependencies list
├── README.md            # Project documentation
```

## Usage
- Open the web application.
- Select your symptoms from the provided list.
- Click the 'Predict Disease' button.
- View the predicted disease and related recommendations.

## Notes
- This model is for educational purposes and should not replace professional medical advice.
- Ensure all required files are present before running the app.

## License
This project is licensed under the MIT License.

## Contact
For any queries, feel free to contact:
- Email: anusomepalli@gmail.com
- GitHub: [Anugna15](https://github.com/Anugna15)

