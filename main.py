import pickle
import json
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from flask import Flask, request, jsonify
from preprocessor import preprocess_input, load_encoder_state


DNN_INPUT_SHAPE = 769
DNN_PATH = "custom-dnn.pth"
GBR_PATH = "gbr.pkl"

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define one-hot encoder state
encoder_state = load_encoder_state()
# define models here
with open(GBR_PATH, "rb") as file:
    gbr = pickle.load(file)

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]  # Assuming you want the last hidden state
        if return_embedding:
            # Return raw embeddings - [CLS] token embedding
            return hidden_state[:, 0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class BERTFineTunerAndEmbeddingExtractor():
    def __init__(self, model, device='cuda', batch_size=16, learning_rate=5e-5, epochs=3):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = nn.L1Loss()

    def fine_tune(self, dataset):
        self.model.train()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                optimizer.zero_grad()
                input_ids = batch['ids'].to(self.device)  # Changed from 'input_ids' to 'ids'
                attention_mask = batch['mask'].to(self.device)  # Changed from 'attention_mask' to 'mask'
                targets = batch['targets'].to(self.device)  # Ensure targets are included correctly
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss(outputs, targets.view(-1, 1))  # Use an appropriate loss function
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Average loss: {total_loss / len(dataloader)}")

    def extract_embeddings(self, descriptions):
        self.model.eval()
        all_embeddings = []
        for i in tqdm(range(0, len(descriptions), self.batch_size)):
            batch = descriptions[i:i+self.batch_size]

            inputs = self.tokenizer(batch, return_tensors='pt', max_length=512, truncation=True, padding='max_length', add_special_tokens=True)

            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_embedding=True)
                embeddings = outputs.cpu().numpy()
                all_embeddings.extend(embeddings)

        return np.array(all_embeddings)


class CustomDNN(nn.Module):
    def __init__(self, input_shape, num_neurons=[128, 64, 32], dropout_rate=0.2):
        super(CustomDNN, self).__init__()
        self.layers = nn.ModuleList()
        for output_features in num_neurons:
            self.layers.append(nn.Linear(input_shape, output_features))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            input_shape = output_features

        self.layers.append(nn.Linear(num_neurons[-1], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, input_shape, num_neurons=[128, 64, 32], dropout_rate=0.2, learning_rate=0.001, epochs=100, batch_size=32, pretrained_model=None):
        self.model_class = model_class
        self.input_shape = input_shape
        self.num_neurons = num_neurons
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        if pretrained_model is None:
            self.model = self.model_class(input_shape, num_neurons, dropout_rate).to(device)
        else:
            self.model = self.model_class(input_shape, num_neurons, dropout_rate)
            self.model.load_state_dict(torch.load(pretrained_model, map_location=torch.device(device)))
            self.model.to(device)
        
            
    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        pbar = tqdm(total=self.epochs, desc="Epochs", leave=True)
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # Update the progress bar
            pbar.set_postfix({'loss': f'{epoch_loss/len(dataloader):.4f}'}, refresh=True)
            pbar.update(1)
        pbar.close()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            predictions = self.model(X_tensor).squeeze()
        return predictions.cpu().numpy()  # Move predictions back to CPU for compatibility with sklearn

    def score(self, X, y):
        predictions = self.predict(X)
        return -mean_absolute_error(y, predictions)

bertFT = DistilBERTClass().to(device)
bertFT.load_state_dict(torch.load('ft_bert.pth', map_location=torch.device(device)))

fine_tuner = BERTFineTunerAndEmbeddingExtractor(model=bertFT, device=device, batch_size=16, learning_rate=5e-5, epochs=3)

dnn_model = PyTorchRegressor(CustomDNN, input_shape=DNN_INPUT_SHAPE, num_neurons=[256, 128, 64], dropout_rate=0.2, learning_rate=0.0001, batch_size=64, pretrained_model=DNN_PATH)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    (X, description) = preprocess_input(data, encoder_state)
    gbr_prediction = gbr.predict(X)

    embeddings = fine_tuner.extract_embeddings(description.to_list())
    embeddings_df = pd.DataFrame(embeddings, columns=[f"bert_feature_{i}" for i in range(embeddings.shape[1])])
    embeddings_df['gbm_pred'] = gbr_prediction
    
    prediction = dnn_model.predict(embeddings_df.to_numpy()).tolist()

    return jsonify(prediction)
    


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
    