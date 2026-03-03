# import the necessary libraries
from flask import Flask, render_template, request
import joblib
import numpy as np

# create a Flask app

app = Flask(__name__)

# load the trained model
model = joblib.load("titanic_model.pkl")

# define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# define the predict route
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get input data from the form
        pclass = int(request.form['pclass'])
        # sex = 1 if request.form['sex'] == 'male' else 0
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])

        # create a numpy array with the input data
        features = np.array([[pclass, sex, age, sibsp, parch, fare]])

        # make a prediction using the loaded model
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0]

        # Return the prediction result
        if prediction[0] == 1:
            result = "Survived! 🎉"
            confidence = probability[1] * 100
        else:
            result = "Did not survive! 😥"
            confidence = probability[0] * 100

        return render_template('result.html', 
                               result=result, 
                               confidence=confidence)
    
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    
if __name__ == '__main__':
    app.run(debug=True)

    # C:\Users\admin\OneDrive\Desktop\Flask\app.py