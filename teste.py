import torch
from transformers import T5Tokenizer, T5ForSequenceClassification

# --- 1. Definições ---

# Mude para o caminho correto do seu modelo salvo
# Pode ser './meu_modelo_f1_instagram' ou o do Reddit
MODEL_PATH = "C:/Users/Felipe/Desktop/Nova pasta/t5_f1_sentiment_model" 

# Mapeamento dos rótulos (Índice 0 = Boa, Índice 1 = Ruim)
labels_map = {0: "Boa", 1: "Ruim"}

# --- 2. Carregar o Modelo e Tokenizador Salvos ---
print(f"Carregando modelo salvo de '{MODEL_PATH}'...")
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Modelo e tokenizador carregados com sucesso!")
except OSError:
    print(f"Erro: Não foi possível encontrar um modelo em '{MODEL_PATH}'.")
    print("Verifique se o caminho está correto e se o modelo foi treinado e salvo.")
    exit()


# --- 3. Texto para Análise ---
# 
# !!! MUDE ESTE TEXTO PARA TESTAR O MODELO !!!
#
# new_text = "Que corrida espetacular! O Hamilton pilotou muito hoje."
# new_text = "A Ferrari errou a estratégia de novo, que desastre."
new_text = "Péssima qualificação, o carro não tinha ritmo nenhum."
#new_text = "O Norris conseguiu um pódio incrível, estou muito feliz!"

print(f"\nAnalisando o texto: '{new_text}'")


# --- 4. Processo de Previsão ---

# Tokeniza o texto e o converte para tensores do PyTorch
inputs = tokenizer(
    ["classifique o sentimento: " + new_text],
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=512
)

# Faz a previsão
with torch.no_grad():
    outputs = model(**inputs)

# O 'outputs.logits' contém a pontuação bruta para cada classe (Boa, Ruim)
logits = outputs.logits[0]

# Pega o índice (0 ou 1) que teve a maior pontuação
prediction_index = torch.argmax(logits).item()

# Mapeia o índice de volta para a etiqueta legível ("Boa" ou "Ruim")
prediction_label = labels_map[prediction_index]

# --- 5. Exibir Resultado ---
print(f"\nResultado da Análise: {prediction_label} (Score: {prediction_index})")

# Opcional: Mostrar os scores brutos
print(f"Scores (Logits): {logits.numpy()}")