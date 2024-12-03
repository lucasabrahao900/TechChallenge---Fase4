# Modelo LSTM para Previsão de Séries Temporais

Este projeto implementa um modelo LSTM (Long Short-Term Memory) para previsão de séries temporais, com uma API REST para servir as previsões.

## Estrutura do Projeto

modelo_lstm/
│
├── modelo_lstm.h5 # Modelo completo salvo
├── modelo_arquitetura.json # Arquitetura do modelo
├── modelo.weights.h5 # Pesos do modelo
├── scaler.pkl # Normalizador dos dados
├── parametros.json # Parâmetros do modelo
└── exemplo_uso_api.py # Código exemplo da API

## Requisitos

```bash
pip install tensorflow scikit-learn pandas numpy fastapi uvicorn requests yfinance
```

## Notas Importantes

- A sequência de entrada deve ter o mesmo tamanho usado no treino do modelo
- Os valores devem ser numéricos
- A ordem temporal dos valores é importante
- O modelo espera uma sequência de preços/valores normalizados

## Limitações e Considerações

- O modelo foi treinado com dados históricos e sua precisão depende da qualidade desses dados
- As previsões são baseadas em padrões históricos e podem não capturar eventos imprevisíveis
- Recomenda-se retrainer o modelo periodicamente com dados mais recentes