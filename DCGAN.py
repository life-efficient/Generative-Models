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
            nn.Linear(2304, 1),
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
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1152),
            nn.LeakyReLU(),
            Unflatten((128, 3, 3)),
            nn.ConvTranspose2d(128, 128, 3, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 3, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 1, 2, 1),
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

    
# %%
from torch.utils.tensorboard import SummaryWriter
from time import time

toPILImg = transforms.ToPILImage()

def train(G, D, epochs=1):
    optimiser_d = torch.optim.SGD(D.parameters(), lr=0.01)
    optimiser_g = torch.optim.SGD(G.parameters(), lr=0.01)
    writer = SummaryWriter(log_dir=f'runs/DCGAN/{time()}')

    batch_idx = 0
    for epoch in range(epochs):
        for idx, (x, _) in enumerate(train_loader):
            z = torch.randn(batch_size, latent_vec_size)
            dgz = D(G(z))

            # GENERATOR UPDATE
            optimiser_g.zero_grad()
            G_loss = torch.log(1 - D(G(z)))
            G_loss = torch.mean(G_loss)
            G_loss.backward(retain_graph=True)
            optimiser_g.step()

            # DISCRIMINATOR UPDATE
            optimiser_d.zero_grad()
            D_loss = - (torch.log(D(x)) + torch.log(1 - D(G(z))))
            D_loss = torch.mean(D_loss)
            D_loss.backward()
            optimiser_d.step()

            print('Epoch:', epoch ,'Batch:', idx)
            print('Loss G:', G_loss.item())
            print('Loss D:', D_loss.item())
            writer.add_scalar('Loss/G', G_loss.item(), batch_idx)
            writer.add_scalar('Loss/D', D_loss.item(), batch_idx)
            batch_idx += 1
            # if idx > 50:
            #     break
        for img in G(z):
            writer.add_image(f'Epoch {epoch}', img)



G = Generator()
D = Discriminator()
train(G, D)
        

# %%
torch.save(G, f'GAN-{time()}.pt')
# %%

def sample():
    writer = SummaryWriter(log_dir=f'runs/DCGAN/{time()}')
    z = torch.randn(batch_size, latent_vec_size)
    for img in G(z):
        writer.add_image(f'test', img)
        print(img)

sample()
# %%
