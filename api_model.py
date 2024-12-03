from tensorflow.keras.models import model_from_json
import joblib
import json
import numpy as np
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd


# Carregando infos das ações para usar na API:
dataset_petr = pd.read_csv(r"C:\Users\lucaa\Desktop\TechChallenge - Fase4\data\raw\dataset_PETR_st.csv")
input_janela = np.array(dataset_petr["Adj Close"][len(dataset_petr)-10:len(dataset_petr)].tolist())

def load_model(model_dir):

    with open(f'{model_dir}/modelo_arquitetura.json', 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(f'{model_dir}/modelo.weights.h5')
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    with open(f'{model_dir}/parametros.json', 'r') as f:
        params = json.load(f)
    
    return model, scaler, params

def valida_input(input_data, scaler, sequence_length):
    # Normalizar e formatar entrada
    scaled_data = scaler.transform(input_data.reshape(-1, 1))
    return scaled_data.reshape(1, sequence_length, 1)

def pred(model, processed_input):
    return model.predict(processed_input)

def cria_nova_lista(lista_real, valor_pred):
    nova_lista = lista_real.copy()
    nova_lista = nova_lista[1:]
    return np.append(nova_lista, valor_pred) 

app = FastAPI()

# Carregar componentes do modelo
model, scaler, params = load_model('C:\\Users\\lucaa\\Desktop\\TechChallenge - Fase4\\models')
seq_length = params['sequence_length']


@app.post("/predict")
async def predict(n_days: int):
    # Validando tamanho da série passada:
    if n_days is None:
        return {"error": f"Precisa informar uma janela"}
    
    # Puxando e modificando os inputs:
    processed_input_inicial = valida_input(input_janela, scaler, seq_length) # Normalizando o input dos 10 dias - sequence
    lista_pred = []

    # Predizendo recursivamente:
    for i in range(n_days):
        v_pred = float(pred(model, processed_input_inicial)[0][0])
        new_input = cria_nova_lista(processed_input_inicial[0], v_pred)
        processed_input_inicial = np.array(new_input).reshape(1, seq_length, 1)
        lista_pred.append(v_pred)
        
    # Tirando a normalização:
    return {"predito": [float(scaler.inverse_transform(np.array([[x]]))[0][0]) for x in lista_pred]}

if __name__ == "__main__":
    uvicorn.run(app)