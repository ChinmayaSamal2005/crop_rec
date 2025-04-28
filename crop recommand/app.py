from flask import Flask, request, render_template
import joblib

app = Flask(__name__,template_folder='template')

# Load the model, scaler, and class mapping
try:
    model = joblib.load('crop_recommendation_model.pkl')
    scaler = joblib.load('scaler.pkl')
    class_mapping = joblib.load('class_mapping.pkl')  # Mapping of indices to crop names
    print("Model, scaler, and class mapping loaded successfully.")
except Exception as e:
    print(f"Error loading model, scaler, or class mapping: {e}")


# Function to recommend soil improvements
def recommend_soil_improvements(n, p, k, ph):
    recommendations = []
    if n < 50:
        recommendations.append("Add nitrogen-rich fertilizer (e.g., urea).")
    if p < 40:
        recommendations.append("Add phosphorus-rich fertilizer (e.g., DAP).")
    if k < 40:
        recommendations.append("Add potassium-rich fertilizer (e.g., potash).")
    if ph < 6.0:
        recommendations.append("Add lime to increase pH.")
    elif ph > 7.5:
        recommendations.append("Add sulfur to lower pH.")
    return recommendations


@app.route('/')
def home():
    return render_template('index.html')  # Input form page


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        data = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

        # Scale the input data
        scaled_data = scaler.transform([data])

        # Predict suitable crops
        probabilities = model.predict_proba(scaled_data)[0]
        crop_indices = probabilities.argsort()[-5:][::-1]  # Get top 5 crops
        crop_names_and_probs = [(class_mapping[index], round(probabilities[index] * 100, 2)) for index in crop_indices]

        # Generate soil improvement suggestions
        soil_improvements = recommend_soil_improvements(data[0], data[1], data[2], data[5])

        # Identify unsuitable crops (low probabilities)
        unsuitable_indices = probabilities.argsort()[:5]
        unsuitable_crops = [class_mapping[index] for index in unsuitable_indices]

        return render_template(
            'result.html',
            crops=crop_names_and_probs,
            unsuitable=unsuitable_crops,
            improvements=soil_improvements
        )
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == '__main__':
    app.run(debug=True)