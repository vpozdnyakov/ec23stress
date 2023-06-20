import torch
from torch.optim import Adam
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from ecstress.tcn_model import TCNModule

class TCNGANModule(LightningModule):
    def __init__(
        self,
        latent_dim,
        target_dim,
        num_layers,
        num_filters,
        dropout,
        num_disc_steps,
        lr,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.dropout = dropout
        self.num_disc_steps = num_disc_steps
        self.lr = lr
        self.automatic_optimization = False

        self.gen = TCNModule(
            input_size=self.latent_dim,
            target_size=self.target_dim,
            nr_params=1,
            kernel_size=2,
            num_filters=self.num_filters,
            num_layers=self.num_layers,
            dilation_base=2,
            dropout=self.dropout,
            weight_norm=False,
        )

        self.disc = TCNModule(
            input_size=self.target_dim,
            target_size=1,
            nr_params=1,
            kernel_size=2,
            num_filters=self.num_filters,
            num_layers=self.num_layers,
            dilation_base=2,
            dropout=self.dropout,
            weight_norm=False,
        )

    def disc_loss(self, real_pred, fake_pred):
        real_is_real = F.binary_cross_entropy_with_logits(
            real_pred, torch.ones_like(real_pred)
        )
        fake_is_fake = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        )
        return (real_is_real + fake_is_fake) / 2

    def gen_loss(self, fake_pred):
        fake_is_real = F.binary_cross_entropy_with_logits(
            fake_pred, torch.ones_like(fake_pred)
        )
        return fake_is_real

    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()

        batch_size = batch.shape[0]
        len_seq = batch.shape[1]
        z = torch.randn(
            batch_size, len_seq, self.latent_dim, device=self.device
        )

        for _ in range(self.num_disc_steps):
            real_pred = self.disc(batch)
            with torch.no_grad():
                fake = self.gen(z)[..., 0]
            fake_pred = self.disc(fake)
            d_loss = self.disc_loss(real_pred, fake_pred)

            disc_opt.zero_grad()
            self.manual_backward(d_loss)
            disc_opt.step()

        fake = self.gen(z)[..., 0]
        fake_pred = self.disc(fake)
        g_loss = self.gen_loss(fake_pred)

        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        gen_opt.step()

        self.log_dict({'train_gen_loss': g_loss, 'train_disc_loss': d_loss})
    
    def validation_step(self, batch, batch_idx):
        
        z = torch.randn(
            1, batch.shape[1], self.latent_dim, device=self.device
        )
        fake = self.gen(z)[0, :, :, 0]
        real = batch[0]
               
        self.log_dict({
            'val_mean_loss': F.mse_loss(fake.mean(axis=0), real.mean(axis=0)),
            'val_max_loss': F.mse_loss(fake.amax(axis=0), real.amax(axis=0)),
            'val_min_loss': F.mse_loss(fake.amin(axis=0), real.amin(axis=0)),
            'val_corrcoef_loss': F.mse_loss(fake.corrcoef(), real.corrcoef()),
        })

    def configure_optimizers(self):
        gen_opt = Adam(self.gen.parameters(), lr=self.lr)
        disc_opt = Adam(self.disc.parameters(), lr=self.lr)
        return gen_opt, disc_opt

    def sample(self, seq_len, temp=1.):
        self.eval()
        z = torch.randn(1, seq_len, self.latent_dim) * temp
        with torch.no_grad():
            return self.gen(z)[0, :, :, 0].cpu()
