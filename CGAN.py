from dataloader import *
from evaluator import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
import matplotlib.pyplot as plt
import random

'''Hyper parameters'''
batch_size = 64
nz = 100 - 24
# hyperparam for Adam optimizers
beta1 = 0.5
'''Hyper parameters'''

manualSeed = 7
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fixed_noise = torch.randn(32, nz, 1, 1, device=device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nz+24, 
                out_channels=64*8,
                kernel_size=(4, 4),
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                in_channels=64*8, 
                out_channels=64*4,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                in_channels=64*4, 
                out_channels=64*2,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                in_channels=64*2, 
                out_channels=64,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                in_channels=64, 
                out_channels=3,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False),
            nn.Tanh()
        )
    
    def forward(self, input, condition):
        input = torch.cat((input, condition), dim=1)
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(64*64*3+24, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input, condition):
        input = torch.cat((input.view(input.size(0), -1), condition), dim=1)
        return self.main(input)

def train(train_loader, test_loader, G, D, evaluator, epochs=150):
    real_label = 1
    fake_label = 0

    lr_D = 2e-4
    lr_G = 1e-3

    optimizerD = optim.Adam(D.parameters(), lr=lr_D, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr_G, betas=(beta1, 0.999))

    criterion=nn.BCELoss()

    for epoch in range(epochs):
        G.train()
        D.train()
        G_loss_sum = 0
        D_loss_sum = 0
        for data, condition in train_loader:
            batch_size = data.size(0)
            condition = condition.clone().float().view(batch_size, -1, 1, 1).to(device)
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            ''' Train with all-real batch '''
            optimizerD.zero_grad()
            real_labels = torch.full((batch_size,), real_label, device=device)
            
            output = D(data.to(device), condition.squeeze()).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, real_labels)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            # D(x)
            D_x = output.mean().item()

            ''' Train with all-fake batch '''
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = G(noise, condition)
            fake_labels = torch.full((batch_size,), fake_label, device=device)
            # Classify all fake batch with D
            output = D(fake.detach(), condition.squeeze()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, fake_labels)
            # Calculate the gradients for this batch
            errD_fake.backward()
            # D(G(z))
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ############################
            optimizerG.zero_grad()
            output = D(fake, condition.squeeze()).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, real_labels)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            G_loss_sum += errG.item()
            D_loss_sum += errD.item()
        
        print("Epochs[%4d/%4d] \n G_Loss : %f D_Loss : %f\n D(x) : %f D_G_z1 : %f / D_G_z2 : %f" % (
                    epoch, epochs, 
                    G_loss_sum, D_loss_sum, 
                    D_x, D_G_z1, D_G_z2
                    )
        )

        utils.save_image(fake,
                        'Res_Train/fake_samples_epoch_%03d.png' % (epoch), 
                        normalize=True, range=(-1, 1)
        )

        torch.save(G.state_dict(), 'model/G_%03d.pth' % (epoch))

        with torch.no_grad():
            D.eval()
            G.eval()
            for condition in test_loader:
                condition = condition.clone().float().view(32, -1, 1, 1).to(device)
                fake = G(fixed_noise, condition)
                fake = fake.squeeze().cpu()
                utils.save_image(fake,
                    'Res_Evaluation/fake_samples_epoch_%03d.png' % (epoch),
                    normalize=True, range=(-1, 1)
                )
                acc = evaluator.eval(fake, condition.squeeze())
                print("Evaluation Accuracy: %f" % (acc))

def evaluate(test_loader, G, evaluator):
    G.load_state_dict(torch.load('model/G_115.pth', map_location=torch.device('cpu')))
    G.eval()
    for condition in test_loader:
        noise = torch.randn(32, nz, 1, 1, device=device)
        condition = condition.clone().float().view(32, -1, 1, 1).to(device)
        fake = G(fixed_noise, condition)
        fake = fake.squeeze().cpu()
        utils.save_image(fake, 
            'Res_Evaluation/fake.png', 
            normalize=True, range=(-1, 1)
        )
        acc = evaluator.eval(fake, condition.squeeze())
        print("Evaluation Accuracy: %f" % (acc))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = JsonDataloader('dataset', 'train')
    test_dataset = JsonDataloader('dataset', 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    G = Generator().to(device)
    G.apply(weights_init)
    D = Discriminator().to(device)
    D.apply(weights_init)

    evaluator = evaluation_model()
    train(train_loader, test_loader, G, D, evaluator)
    evaluate(test_loader, G, evaluator)