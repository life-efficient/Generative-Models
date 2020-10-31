#%%
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 128
latent_vec_size = 128

train_data = datasets.MNIST(
    train=True,
    download=True,
    root='./MNIST-data',
    transform=transforms.ToTensor()
)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

#%%
def sample(writer=None, device='cpu', tag='test'):
    if writer is None:
        writer = SummaryWriter(log_dir=f'runs/DCGAN/{time()}')
    z = torch.randn(batch_size, latent_vec_size).to(device)
    for img in G(z):
        writer.add_image(f'DCGAN/{tag}', img)

#%%

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128*7*7, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.layers(x)
        # print('discriminator output:', x.shape)
        return x

d = Discriminator()

# x, _ = train_data[0]
# x = x.unsqueeze(0)
# print(d(x))

# %%

class Unflatten(nn.Module):
    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(x.shape[0], *self.out_shape)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # nn.Linear(128, 3*3*64),
            # nn.LeakyReLU(),
            # # nn.Linear(256, 16),
            # # nn.LeakyReLU(),
            # Unflatten((64, 3, 3)),
            # nn.BatchNorm2d(64),
            # nn.ConvTranspose2d(64, 64, 3, 2),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(64),
            # nn.ConvTranspose2d(64, 16, 2, 2),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(16),
            # nn.ConvTranspose2d(16, 1, 2, 2),
            # nn.Sigmoid(),

            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 128*7*7),
            nn.LeakyReLU(),
            Unflatten((128, 7, 7)),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), #128x7x7 -> 64x14x14
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1), #64x14x14 -> 1x28x28
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.layers(z)
        return z

g = Generator()
ran_batch = torch.rand(batch_size, latent_vec_size)
fake = g(ran_batch)
print('generated shape:', fake.shape)
# sdfs

#%%
def show(x):
    pass

    
# %%
from torch.utils.tensorboard import SummaryWriter
from time import time

criterion = nn.BCELoss()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(G, D, epochs=1):
    optimiser_d = torch.optim.Adam(D.parameters(), lr=0.0001)
    optimiser_g = torch.optim.Adam(G.parameters(), lr=0.001)
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
            D_loss = criterion(D(G(z).detach()), labels) # loss on fake examples
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
                sample(writer, device, tag=f'{time()}')

G = Generator()
D = Discriminator()
train(G, D, epochs=10)
        

# %%
torch.save(G, f'GAN-{time()}.pt')
# %%


# %%
