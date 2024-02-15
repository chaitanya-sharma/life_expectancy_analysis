import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, app, jsonify, url_for, render_template
import torch 
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(21, 64),  
            nn.ReLU(),                   
            nn.Linear(64, 64),             
            nn.ReLU(),                  
            nn.Linear(64, 64),             
            nn.ReLU(),                   
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)

app=Flask(__name__)           #starting point of the flask app

# load model weights and biases and the scaler
model=LinearRegression()  
with open("life_expectancy_ann_1.pkl","rb") as fp:
    model.load_state_dict(pickle.load(fp))

scaler=pickle.load(open('scaler_1.pickle','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    new_data = torch.from_numpy(new_data).float()
    # output=model(new_data)
    # print(output[0])
    # return jsonify(output[0])
    output = model(new_data)
    prediction = output.tolist()  # Convert tensor to a list
    data = {"prediction": prediction}  # Create a dictionary
    return jsonify(data)


if __name__=="__main__":
    app.run(debug=True)
