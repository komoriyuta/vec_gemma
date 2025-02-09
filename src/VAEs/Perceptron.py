import torch 
from torch import nn

class Perceptron(nn.Module):
    def __init__(self, bert_output_dim, gpt_input_dim, hidden_dim=400, latent_dim=32):
        super(Perceptron, self).__init__()
        self.input_dim = bert_output_dim
        self.output_dim = gpt_input_dim
        
        
        self.fc1 = nn.Linear(bert_output_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, gpt_input_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        y = self.fc2(x)
        return y,x,x 
    
    def encode(self, x):
        return self.fc1(x), self.fc1(x)
    def decode(self, z):
        return self.fc2(z)
    