from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load your pre-trained model
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['pclass']) if request.form['pclass'] else 0
    sex = int(request.form['sex']) if request.form['sex'] else 0
    age = float(request.form['age']) if request.form['age'] else 0.0
    sibsp = int(request.form['sibsp']) if request.form['sibsp'] else 0
    parch = int(request.form['parch']) if request.form['parch'] else 0
    fare = float(request.form['fare']) if request.form['fare'] else 0.0
    embarked = int(request.form['embarked']) if request.form['embarked'] else 0
    boat = int(request.form['boat']) if request.form['boat'] else 0

    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked,boat]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = 'Survived'
    else:
        result = 'Did not survive'

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)
