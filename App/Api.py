from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return  "<h1>Proyecto para The bridge, caso Twitter</h1>\
            <p><h5>Luis Valverde</h5></p>"
            

# --PREDICCIÓN--
@app.route('/predict', methods=['GET'])
def predict():
    model = pickle.load(open('../model/sentiment_model','rb'))
    
    text = request.args.get('text', None)
    file={'text':text}
    df_test=pd.DataFrame()
    df_test['text']=file

    
    if text is None:
        return "Faltan argumentos para realizar la predicción"

    prediction = model.predict(df_test)

    if prediction == 0:
         return 'El TEXTO no genera impacto.'
    else:
         return 'Ha conseguido un  TEXTO exitoso, le recomendamos que lo publique'

app.run()