from flask import Flask, request
import os
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('stopwords')


# Path del modelo
MODEL='../model/sentiment_model'


os.chdir(os.path.dirname(__file__))
app = Flask(__name__)
app.config['DEBUG'] = True

# --HOME--
@app.route("/", methods=['GET'])
def hello():    
    return "<h1>Bienvenido al Home</h1>\
        <p><h2>Ejemplo para introducción de datos:</h2></p>\
        <p><h3>URL PREDICT</h3></p>\
        <p><h3>/predict</h3></p>\
        <p><h3>POSITIVO </h3></p>\
        <p><h3>/predict?texto=Me ha gustado todo el proceso, han sido 4 meses muy intensos</h3></p>\
        <p><h3>NEGATIVO O NEUTRO </h3></p>\
        <p><h3>/predict?texto=Ha sido una mala experiencia</h3></p>"

# --PREDICCIÓN--
@app.route('/predict', methods=['GET'])
def predict():
    
    # Input del usuario
    texto = request.args.get('texto', None)

    if texto is None:
        return "Texto a analizar no encontrado. Por favor, introduzca un texto"
    else:

        # Pasar a Dataframe
        texto1_df = pd.DataFrame(columns=['tweet'])
        texto1_df = texto1_df.append({'tweet': texto}, ignore_index=True)

        # Quitar signos de puntiación
        signos = re.compile("(\.)|(\;)|(\:)|(\!)|(\¡)|(\?)|(\¿)|(\@)|(\,)|(\")|(\#)|(\()|(\))|(\[)|(\])|(\d+)")

        def signs_tweets(tweet):
            return signos.sub('', tweet.lower())

        texto1_df['tweet'] = texto1_df['tweet'].apply(signs_tweets)

        # Quitar links
        def remove_links(df):
            return " ".join(['' if ('http') in word else word for word in df.split()])

        texto1_df['tweet'] = texto1_df['tweet'].apply(remove_links)

        # Aplicamos Stopwords
        spanish_stopwords = stopwords.words('spanish')

        def remove_stopwords(df):
            return " ".join([word for word in df.split() if word not in spanish_stopwords])

        texto1_df['tweet'] = texto1_df['tweet'].apply(remove_stopwords)

        # Aplicamos SnowballStemmer
        def spanish_stemmer(x):
            stemmer = SnowballStemmer('spanish')
            return " ".join([stemmer.stem(word) for word in x.split()])
            
        texto1_df['tweet'] = texto1_df['tweet'].apply(spanish_stemmer)

        # Quitar emojis
        texto1_df=texto1_df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
        
        # Cargar modelo
        model = pickle.load(open(MODEL,'rb'))

        # Sacar predicción
        prediction = model.predict(texto1_df['tweet'])
        
        # Resultado
        if prediction == 1:
            return 'El texto proporcionado tiene un sentimiento malo o neutro'
        else:
            return 'El texto proporcinado tiene un sentimiento positivo'

app.run()