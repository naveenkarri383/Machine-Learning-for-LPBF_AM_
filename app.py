import pickle
from flask import Flask, request, jsonify, url_for, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)  # starting point of the application
#Loading the model
knn_model=pickle.load(open('knn_model.pkl', 'rb'))
scaler = pickle.load(open('scalling_new.pkl', 'rb'))

@app.route('/')  # first route and localify the host
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']   # Input given in JSON format
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))   # dictionary values
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = knn_model.predict(new_data)
    print(output[0])  #printing first value
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]  
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=knn_model.predict(final_input)[0]
    #return render_template("index2.html",prediction_text="Predicted Mechanical Properties are Yield Strength{a}, Ultimate Tensile Strength {b}, Elongation{c}".format(a=output[0],b=output[1],c=output[2]))
    # Pass individual predicted values to the HTML template
    return render_template('index.html', prediction_1=output[0], prediction_2=output[1], prediction_3=output[2])


if __name__ == '__main__':
    app.run(debug=True)
