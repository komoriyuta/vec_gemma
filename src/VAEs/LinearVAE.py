import torch 
from torch import nn
from torch.nn import functional as F
import torch.nn.functional as F

class LinearVAE(nn.Module):
    def __init__(self, bert_output_dim, gpt_input_dim, hidden_dim=400, latent_dim=32):
        super(LinearVAE, self).__init__()
        self.input_dim = bert_output_dim
        self.output_dim = gpt_input_dim
        
        # Encoder
        self.fc11 = nn.Linear(bert_output_dim, latent_dim)
        self.fc12 = nn.Linear(bert_output_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, gpt_input_dim)
        
    def encode(self, x):
        return self.fc11(x), self.fc12(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.fc3(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar