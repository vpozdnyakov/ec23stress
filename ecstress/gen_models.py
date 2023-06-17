import torch
from torch.optim import Adam
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
        real_is_real = torch.log(torch.sigmoid(real_pred))
        fake_is_fake = torch.log(1 - torch.sigmoid(fake_pred))
        return -(real_is_real + fake_is_fake).mean()
    
    def gen_loss(self, fake_pred):
        fake_is_real = torch.log(torch.sigmoid(fake_pred))
        return -fake_is_real.mean()
    
    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()
        
        batch_size = batch.shape[0]
        len_seq = batch.shape[1]
        z = torch.randn(batch_size, len_seq, self.latent_dim, device=self.device)
        
        for _ in range(self.num_disc_steps):
            real_pred = self.disc(batch)
            with torch.no_grad():
                fake = self.gen(z)[..., 0]
            fake_pred = self.disc(fake)
            d_loss = self.disc_loss(real_pred, fake_pred)
            
            torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 0.5)

            disc_opt.zero_grad()
            self.manual_backward(d_loss)
            disc_opt.step()

        fake = self.gen(z)[..., 0]
        fake_pred = self.disc(fake)
        g_loss = self.gen_loss(fake_pred)
        
        torch.nn.utils.clip_grad_norm_(self.dics.parameters(), 0.5)

        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        gen_opt.step()

        self.log_dict({'train_gen_loss': g_loss, 'train_disc_loss': d_loss})

    def validation_step(self, batch, batch_idx):
        
        batch_size = batch.shape[0]
        len_seq = batch.shape[1]
        z = torch.randn(batch_size, len_seq, self.latent_dim, device=self.device)
        
        real_pred = self.disc(batch)
        fake = self.gen(z)[..., 0]
        fake_pred = self.disc(fake)
        d_loss = self.disc_loss(real_pred, fake_pred)

        fake = self.gen(z)[..., 0]
        fake_pred = self.disc(fake)
        g_loss = self.gen_loss(fake_pred)
        
        self.log_dict({'val_gen_loss': g_loss, 'val_disc_loss': d_loss})
    
    def configure_optimizers(self):
        gen_opt = Adam(self.gen.parameters(), lr=self.lr)
        disc_opt = Adam(self.disc.parameters(), lr=self.lr)
        return gen_opt, disc_opt

    def sample(self, seq_len):
        self.eval()
        z = torch.randn(1, seq_len, self.latent_dim)
        with torch.no_grad():
            return self.gen(z)[0, :, :, 0].cpu()


class TCNVAEModule(LightningModule):
    def __init__(
        self, input_dim, latent_dim, hidden_dim, num_layers,
    ):
        super().__init__()
        self.enc = TCNModule(
            input_size=input_dim,
            target_size=latent_dim,
            nr_params=2,
            kernel_size=2,
            num_filters=hidden_dim,
            num_layers=num_layers,
            dilation_base=2,
            dropout=0.2,
            weight_norm=False,
        )
        self.dec = TCNModule(
            input_size=latent_dim,
            target_size=input_dim,
            nr_params=1,
            kernel_size=2,
            num_filters=hidden_dim,
            num_layers=num_layers,
            dilation_base=2,
            dropout=0.2,
            weight_norm=False,
        )

    def neg_elbo(self, ts, rec_ts, sigma, mu):
        like = -((rec_ts - ts)**2).sum((1,2))
        kld = -0.5 * (1 + (sigma**2).log() - mu**2 - sigma**2).sum((1,2))
        elbo = like - kld
        return -elbo.mean(), like.mean(), kld.mean()

    def forward(self, ts):
        # ts shape: (batch size, seq len, num channels)
        params = self.enc(ts) # (batch size, seq len, 1, 2)
        mu, sigma = params[..., 0], torch.sigmoid(params[..., 1])
        return mu, sigma
    
    def reparametrization_trick(self, mu, sigma):
        z = mu + torch.randn_like(mu) * sigma
        return z

    def training_step(self, ts, batch_idx):
        mu, sigma = self.forward(ts)
        z = self.reparametrization_trick(mu, sigma)
        rec_ts = self.dec(z)[..., 0]
        loss, like, kld = self.neg_elbo(ts, rec_ts, sigma, mu)
        self.log_dict({'val_loss': loss, 'val_like': like, 'val_kld': kld})
        return loss
    
    def validation_step(self, ts, batch_idx):
        mu, sigma = self.forward(ts)
        rec_ts = self.dec(mu)[..., 0]
        loss, like, kld = self.neg_elbo(ts, rec_ts, sigma, mu)
        self.log_dict({'val_loss': loss, 'val_like': like, 'val_kld': kld})
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer 
    
    def sample(self, seq_len):
        self.eval()
        z = torch.randn(1, seq_len, 1)
        with torch.no_grad():
            return self.dec(z)[0, :, :, 0].cpu()
