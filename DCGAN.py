#%%
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size=64

train_data = datasets.MNIST(
    train=True,
    download=True,
    root='./MNIST-data',
    transform=transforms.ToTensor()
)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

#%%
def sample(writer=None, device='cpu'):
    if writer is None:
        writer = SummaryWriter(log_dir=f'runs/DCGAN/{time()}')
    z = torch.randn(batch_size, latent_vec_size).to(device)
    for img in G(z):
        writer.add_image(f'test', img)

#%%

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, 3, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(2304, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

d = Discriminator()

x, _ = train_data[0]
x = x.unsqueeze(0)
print(d(x))

# %%

class Unflatten(nn.Module):
    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1152),
            # nn.LeakyReLU(),
            # nn.Linear(1024, 1152),
            nn.LeakyReLU(),
            Unflatten((128, 3, 3)),
            nn.ConvTranspose2d(128, 64, 3, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.layers(z)

g = Generator()
latent_vec_size = 100
ran_batch = torch.rand(batch_size, latent_vec_size)
fake = g(ran_batch)
print(fake.shape)

#%%
def show(x):
    pass

    
# %%
from torch.utils.tensorboard import SummaryWriter
from time import time

toPILImg = transforms.ToPILImage()

criterion = nn.BCELoss()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(G, D, epochs=1):
    optimiser_d = torch.optim.Adam(D.parameters(), lr=0.00001)
    optimiser_g = torch.optim.Adam(G.parameters(), lr=0.0001)
    writer = SummaryWriter(log_dir=f'runs/DCGAN/{time()}')
    G = G.to(device)
    D = D.to(device)
    batch_idx = 0
    
    for epoch in range(epochs):
        for idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            z = torch.randn(batch_size, latent_vec_size)
            z = z.to(device)

            # GENERATOR UPDATE
            optimiser_g.zero_grad()
            labels = torch.ones(batch_size).to(device)
            G_loss = criterion(D(G(z)), labels)
            # G_loss = - torch.log(1 - D(G(z)))
            # G_loss = torch.mean(G_loss)
            G_loss.backward(retain_graph=True)
            optimiser_g.step()

            # DISCRIMINATOR UPDATE
            optimiser_d.zero_grad()
            labels = torch.zeros(batch_size).to(device)
            D_loss = criterion(D(G(z)), labels) # loss on fake examples
            # D_loss = - (torch.log(D(x)) + torch.log(1 - D(G(z))))
            # D_loss = torch.mean(D_loss)
            D_loss.backward()
            labels = torch.ones(x.shape[0]).to(device)
            D_loss = criterion(D(x), labels)
            D_loss.backward()
            optimiser_d.step()
            
            writer.add_scalar('Loss/G', G_loss.item(), batch_idx)
            writer.add_scalar('Loss/D', D_loss.item(), batch_idx)
            batch_idx += 1
            print(
                'Epoch:', epoch ,
                'Batch:', idx,
                'Loss G:', G_loss.item(),
                'Loss D:', D_loss.item()
            )
            if idx % 100 == 0:
                print('sampling')
                sample(writer, device)

G = Generator()
D = Discriminator()
train(G, D, epochs=10)
        

# %%
torch.save(G, f'GAN-{time()}.pt')
# %%


# %%
