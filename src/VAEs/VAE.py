import torch 
from torch import nn
from torch.nn import functional as F
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, bert_output_dim, gpt_input_dim, hidden_dim=400, latent_dim=32):
        super(VAE, self).__init__()
        self.input_dim = bert_output_dim
        self.output_dim = gpt_input_dim
        
        # Encoder
        self.fc1 = nn.Linear(bert_output_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # μ
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # log(σ^2)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, gpt_input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar