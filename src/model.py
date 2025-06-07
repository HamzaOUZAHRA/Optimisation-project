# src/model.py
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_in=784, h1=256, h2=128, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h1)
        self.fc2 = nn.Linear(h1,  h2)
        self.fc3 = nn.Linear(h2,  d_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # aplatissement en (batch_size, 784)
        x = F.relu(self.fc1(x))    # couche cachée 1
        x = F.relu(self.fc2(x))    # couche cachée 2
        return self.fc3(x)         # logits

# Ajouter ce bloc pour pouvoir exécuter directement le fichier
if __name__ == "__main__":
    # Instancie un modèle MLP et affiche sa structure
    model = MLP()
    print("✔️ MLP instantiated successfully:")
    print(model)
