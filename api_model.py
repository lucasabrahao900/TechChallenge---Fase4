# Exemplo de como usar o modelo em uma API

from tensorflow.keras.models import model_from_json
import joblib
import json
import numpy as np

def load_model_components(model_dir):
    # Carregar arquitetura
    with open(f'{model_dir}/modelo_arquitetura.json', 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    
    # Carregar pesos
    model.load_weights(f'{model_dir}/modelo.weights.h5')
    
    # Carregar scaler
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    
    # Carregar parâmetros
    with open(f'{model_dir}/parametros.json', 'r') as f:
        params = json.load(f)
    
    return model, scaler, params

def preprocess_input(input_data, scaler, sequence_length):
    # Normalizar e formatar entrada
    scaled_data = scaler.transform(input_data.reshape(-1, 1))
    return scaled_data.reshape(1, sequence_length, 1)

def make_prediction(model, processed_input):
    return model.predict(processed_input)

# Exemplo de uso em FastAPI:
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

# Carregar componentes do modelo
model, scaler, params = load_model_components('C:\\Users\\lucaa\\Desktop\\TechChallenge - Fase4\\models')
sequence_length = params['sequence_length']

class PredictionInput(BaseModel):
    sequence: list[float]

@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Verificar tamanho da sequência
    if len(input_data.sequence) != sequence_length:
        return {"error": f"Input deve ter {sequence_length} valores"}
    
    # Preparar dados
    input_array = np.array(input_data.sequence)
    processed_input = preprocess_input(input_array, scaler, sequence_length)
    
    # Fazer previsão
    prediction = make_prediction(model, processed_input)
    
    # Reverter normalização
    final_prediction = scaler.inverse_transform(prediction)[0][0]
    
    return {"prediction": float(final_prediction)}

if __name__ == "__main__":
    uvicorn.run(app)