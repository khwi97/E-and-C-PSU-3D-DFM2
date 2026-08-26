import numpy as np
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import os
from abc import abstractmethod
from accelerate import Accelerator
import requests
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange
from torch.amp import autocast


from tqdm.auto import tqdm
import torch
from torch import nn, einsum
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam, AdamW

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
import random 

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = 'cuda'
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = 1. - betas
    alphas = torch.clip(alphas, 0.001, 1.)
    return torch.clip(betas, 0, 0.999), torch.sqrt(alphas)


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def divisible_by(numer, denom):
    return (numer % denom) == 0

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 500,
        sampling_timesteps = None, #if sampling_timesteps<timesteps, do ddim sampling
        objective = 'pred_x0',
        beta_schedule = 'cosine',
        hybrid_loss_coeff = 0.001,
        ddim_sampling_eta = 1., # 1 for ddpm, 0 for ddim
        offset_noise_strength = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        use_cfg_plus_plus = False
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.module.channels
        if isinstance(image_size, int):
            self.image_height = self.image_width = image_size
        else:
            assert len(image_size) == 2, "image_size must be int or (H, W) tuple"
            self.image_height, self.image_width = image_size
        self.objective = objective
        self.hybrid_loss_coeff = hybrid_loss_coeff
        self.use_cfg_plus_plus = use_cfg_plus_plus
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas, multi_alphas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # sampling related parameters
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        
        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        
        ### Gaussian fixed constant 
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight 
        # the weight in the paper, Variational Diffusion Models (VDM)

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr 
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    # compute x_0 from x_t and pred noise: the reverse of `q_sample`; inverse of Eq.(9) in improved DDPM
    def gaussian_predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def gaussian_predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def gaussian_predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def gaussian_predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def gaussian_q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    @autocast('cuda', enabled = False)
    def gaussian_q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def gaussian_model_predictions(self, x, t, classes, clip_x_start=False):
        model_output = self.model(x, t, classes)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.gaussian_predict_start_from_noise(x, t, model_output)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = maybe_clip(model_output)
            pred_noise = self.gaussian_predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = maybe_clip(self.gaussian_predict_start_from_v(x, t, v))
            pred_noise = self.gaussian_predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)


    def gaussian_p_mean_variance(self, x, t, classes, clip_denoised=True):
        preds = self.gaussian_model_predictions(x, t, classes, clip_x_start=clip_denoised)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(min=-1., max=1.)

        model_mean, posterior_variance, posterior_log_variance = self.gaussian_q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start


    @torch.no_grad()
    def gaussian_p_sample(self, x, t: int, classes, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.gaussian_p_mean_variance(x = x, t = batched_times, classes = classes, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start


    @torch.no_grad()
    def gaussian_ddim_sample(self, classes, shape, clip_denoised = True, preset_sampling_timesteps=None, preset_ddim_sampling_eta=None, save_intermediate=False):

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        
        if preset_sampling_timesteps is not None:
            sampling_timesteps = preset_sampling_timesteps
        if preset_ddim_sampling_eta is not None:
            eta = preset_ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) 

        img = torch.randn(shape, device = device)
        x_start = None

        if save_intermediate:
            noisy_imgs = []

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', leave = False):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            model_output = self.model(img, time_cond, classes)

            if objective == 'pred_noise':
                pred_noise = model_output
                x_start = self.gaussian_predict_start_from_noise(img, time_cond, pred_noise)
                
                if clip_denoised:
                    x_start.clamp_(-1., 1.)
                    alpha = self.alphas_cumprod[time]
                    pred_noise = (img - x_start * alpha.sqrt()) / (1 - alpha).sqrt()

            elif objective == 'pred_x0':
                x_start = model_output
                if clip_denoised:
                    x_start.clamp_(-1., 1.)
                pred_noise = self.gaussian_predict_noise_from_start(img, time_cond, x_start)

            elif objective == 'pred_v':
                v = model_output
                x_start = self.gaussian_predict_start_from_v(img, time_cond, v)
                if clip_denoised:
                    x_start.clamp_(-1., 1.)
                pred_noise = self.gaussian_predict_noise_from_start(img, time_cond, x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if save_intermediate:
                noisy_imgs.append((unnormalize_to_zero_to_one(img[0])*255.0).cpu().detach().int().clamp(0, 255).permute(1, 2, 0).numpy())
        ##end for

        img1 = img[:,0,:,:].unsqueeze(1)
        img2 = img[:,1,:,:].unsqueeze(1)
        img1 = unnormalize_to_zero_to_one(img1)
        img2 = (img2 > 0).long()
        
        img = torch.mul(img1, img2)
        return (img, noisy_imgs) if save_intermediate else img


    @torch.no_grad()
    def gaussian_sample(self, classes, save_intermediate=False):
        batch_size, image_height, image_width, channels = classes.shape[0], self.image_height, self.image_width, self.channels
        return self.p_sample_loop(classes, (batch_size, channels, image_height, image_width), save_intermediate)

    @torch.no_grad()
    def interpolate(self, x1, x2, classes, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img, _ = self.p_sample(img, i, classes)

        return img

    def p_losses(self, x_start, t, *, classes, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Noise sample
        x = self.gaussian_q_sample(x_start = x_start, t = t, noise = noise)

        # Predict and take gradient step
        model_out = self.model(x, t, classes)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.gaussian_predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # 1. Raw MSE for validation
        raw_loss = F.mse_loss(model_out, target, reduction='none')
        raw_loss = reduce(raw_loss, 'b ... -> b', 'mean')
        raw_loss_scalar = raw_loss.mean() # 스칼라 값

        # 2. Weighted Loss for training
        loss = raw_loss * extract(self.loss_weight, t, raw_loss.shape)
        loss = loss.mean()
        
        return loss, raw_loss_scalar
    
    def forward(self, img, *args, fixed_t=None, **kwargs): # [수정] fixed_t 인자 추가
        b, c, h, w, device = *img.shape, img.device
        assert (h, w) == (self.image_height, self.image_width)
        
        if fixed_t is not None:
            if not torch.is_tensor(fixed_t):
                t = torch.full((b,), fixed_t, device=device, dtype=torch.long)
            elif fixed_t.ndim == 0:
                t = torch.full((b,), fixed_t.item(), device=device, dtype=torch.long)
            else:
                t = fixed_t.to(device)
        else:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(img, t, *args, **kwargs)
    

    @torch.no_grad()
    def sample(self, shape, classes):
        b = classes.shape[0]
        device = classes.device
        z_norm = torch.randn(shape, device=device)
        classes = classes.to(device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps, leave = False):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            z_norm, _ = self.gaussian_p_sample(z_norm, i, classes, clip_denoised=False)

        print()

        z = z_norm[:,0,:,:].unsqueeze(1)
        z_cat = z_norm[:,1,:,:].unsqueeze(1)
        z_cat = (z_cat > 0).long()
        z = unnormalize_to_zero_to_one(z)
        sample = torch.cat([z, z_cat], dim=1).cpu()

        return sample
    



def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)   
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, t_emb, c_emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, t_emb, c_emb=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t_emb, c_emb)
            else:
                x = layer(x)
        return x

# use GN for norm layer
def norm_layer(channels, num_groups=32):
    return nn.GroupNorm(num_groups, channels)
    # return nn.BatchNorm2d(channels)

class GaussianSmearing(nn.Module):
    def __init__(self, embed_dim, start=-0.1, stop=1.1, width_scale=1):
        """
        embed_dim: number of bins of rbf kernel per one single parameter
        """
        super().__init__()
        offset = torch.linspace(start, stop, embed_dim)
        width = (offset[1] - offset[0]) * width_scale
        
        self.register_buffer('coeff', -0.5 / (width ** 2))
        self.register_buffer('offset', offset)
        
    def forward(self, x):
        x = x.unsqueeze(-1) # (B, C, 1)
        offset = self.offset.view(1, 1, -1) # (1, 1, E)
        
        diff = x - offset
        y = torch.exp(self.coeff * torch.pow(diff, 2))
        
        return y

class PhysicalConditioning(nn.Module):
    def __init__(self, num_params=10, n_bins=128, feat_dim=512):
        super().__init__()
        
        self.rbf = GaussianSmearing(embed_dim=n_bins)
        
        self.seq_mlp = nn.Sequential(
            nn.Linear(n_bins, feat_dim), 
            nn.LayerNorm(feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim) 
        )
        self.type_embed = nn.Parameter(torch.randn(1, num_params, feat_dim))

        
        self.scale_mlp = nn.Sequential(
            nn.Linear(num_params, feat_dim), 
            nn.LayerNorm(feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
        self.shift_mlp = nn.Sequential(
            nn.Linear(num_params, feat_dim), 
            nn.LayerNorm(feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
        # Zero Init
        nn.init.zeros_(self.scale_mlp[-1].weight)
        nn.init.zeros_(self.scale_mlp[-1].bias)
        nn.init.zeros_(self.shift_mlp[-1].weight)
        nn.init.zeros_(self.shift_mlp[-1].bias)

    def forward(self, params_raw):
        x_rbf = self.rbf(params_raw) 
        tokens = self.seq_mlp(x_rbf) + self.type_embed
        
        params_centered = (params_raw - 0.5) * 2.0
        scale = self.scale_mlp(params_centered)
        shift = self.shift_mlp(params_centered)
        
        return tokens, scale, shift

class ResidualBlock(TimestepBlock):
    def __init__(self, channels, out_channels, time_channels, cond_channels, dropout, use_scale_shift_norm=False, num_groups=32):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
                
        self.conv1 = nn.Sequential(
            norm_layer(channels, num_groups=num_groups),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, 2 * out_channels if use_scale_shift_norm else out_channels)
        )

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_channels, 2 * out_channels if use_scale_shift_norm else out_channels)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels, num_groups=num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if channels != out_channels:
            self.shortcut = nn.Conv2d(channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb, c_info=None):
        """
        x: (Batch, Channels, H, W)
        time_emb: (Batch, TimeDim)
        c_info: (tokens, scale, shift) - PhysicalConditioning output
        """
        h = self.conv1(x)

        t_out = self.time_mlp(time_emb)
        t_out = rearrange(t_out, 'b c -> b c 1 1')
        
        _, c_scale, c_shift = c_info
        
        c_raw = torch.cat([c_scale, c_shift], dim=1)
        
        c_out = self.cond_proj(c_raw)
        c_out = rearrange(c_out, 'b c -> b c 1 1')

        total_emb = t_out + c_out
        
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(total_emb, 2, dim=1)
            h = self.conv2[0](h) * (1 + scale) + shift 
            h = self.conv2[1:](h)
        else:
            h = self.conv2(h + total_emb)
            
        return h + self.shortcut(x)


class AttentionBlock(TimestepBlock):
    def __init__(self, channels, num_heads=1, num_groups=32, cond_dim=512):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        self.norm = norm_layer(channels, num_groups=num_groups)
        # image Self-Attention
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        # parameter Cross-Attention 
        self.cond_proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, t_emb, c_info):
        # c_info: (tokens, scale, shift)
        c_tokens, _, _ = c_info 
        
        B, C, H, W = x.shape
        
        # -----------------------------------------------------------
        # 1. image Q, K, V (Self-Attention)
        # -----------------------------------------------------------
        qkv = self.qkv(self.norm(x))
        # (B*Heads, HeadDim, HW)
        q, k_img, v_img = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)

        # -----------------------------------------------------------
        # 2. Parameter K, V (Cross-Attention)
        # -----------------------------------------------------------
        # (B, 10, C*2)
        cond_kv = self.cond_proj(c_tokens)
        
        # (B, 10, Heads, 2*HeadDim)
        cond_kv = cond_kv.reshape(B, 10, self.num_heads, 2 * (C // self.num_heads))
        # (B, Heads, 2*HeadDim, 10)
        cond_kv = cond_kv.permute(0, 2, 3, 1)
        # (B*Heads, 2*HeadDim, 10)
        cond_kv = cond_kv.reshape(B * self.num_heads, -1, 10)
        
        # (B*Heads, HeadDim, 10)
        k_cond, v_cond = cond_kv.chunk(2, dim=1) 

        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        
        # -----------------------------------------------------------
        # 3. Self-Attention 
        # -----------------------------------------------------------
        attn_self = torch.einsum("bct,bcs->bts", q * scale, k_img * scale)
        attn_self = attn_self.softmax(dim=-1)
        h_self = torch.einsum("bts,bcs->bct", attn_self, v_img)

        # -----------------------------------------------------------
        # 4. Cross-Attention
        # -----------------------------------------------------------
        attn_cross = torch.einsum("bct,bcs->bts", q * scale, k_cond * scale)
        attn_cross = attn_cross.softmax(dim=-1) 
        h_cross = torch.einsum("bts,bcs->bct", attn_cross, v_cond)

        # Parallel Attention 
        h = h_self + h_cross
        
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        
        return h + x
    
# upsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

# downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)


# The full UNet model with attention and timestep embedding
class Unet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        channels=2,
        model_channels=128,
        out_channels=None,
        num_res_blocks=2,
        attention_resolutions=(4, 8, 16, 32),
        dropout=0.2,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_heads=4,
        use_scale_shift_norm=True,
        learned_variance=False,
        num_groups=32,
    ):
        super().__init__()

        default_out_dim = channels * (1 if not learned_variance else 2)
        out_channels = default(out_channels, default_out_dim)

        self.num_classes = num_classes
        self.channels = channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_watch = model_channels//2

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # condition embedding (always used)
        cond_embed_dim = model_channels * 4
        self.c_emb_dim = model_channels

        self.fourier_proj = PhysicalConditioning(n_bins=self.c_emb_dim, feat_dim=cond_embed_dim)

        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(channels, model_channels, kernel_size=3, padding=1))
        ])

        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(
                        ch, mult * model_channels,
                        time_embed_dim, cond_embed_dim*2,
                        dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=num_groups
                    )
                ]
                ch = mult * model_channels

                if ds in attention_resolutions:
                    current_head = max(num_heads, ch // self.num_watch)
                    layers.append(AttentionBlock(ch, num_heads=current_head, num_groups=num_groups, cond_dim=cond_embed_dim))

                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle
        current_head = max(num_heads, ch // self.num_watch)
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, cond_embed_dim*2, dropout=dropout,
                          use_scale_shift_norm=use_scale_shift_norm, num_groups=num_groups),
            AttentionBlock(ch, num_heads=current_head, num_groups=num_groups, cond_dim=cond_embed_dim),
            ResidualBlock(ch, ch, time_embed_dim, cond_embed_dim*2, dropout=dropout,
                          use_scale_shift_norm=use_scale_shift_norm, num_groups=num_groups)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        cond_embed_dim*2,
                        dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=num_groups
                    )
                ]
                ch = model_channels * mult

                if ds in attention_resolutions:
                    current_head = max(num_heads, ch // self.num_watch)
                    layers.append(AttentionBlock(ch, num_heads=current_head, num_groups=num_groups, cond_dim=cond_embed_dim))

                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2

                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch, num_groups=num_groups),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps, classes):
        """
        Pure conditional diffusion: always condition on classes.
        """
        batch, device = x.shape[0], x.device
        hs = []

        t_emb = self.time_mlp(timestep_embedding(timesteps, self.model_channels))

        c_tokens, c_scale, c_shift = self.fourier_proj(classes)
        c_info = (c_tokens, c_scale, c_shift)

        # down
        h = x
        for module in self.down_blocks:
            h = module(h, t_emb, c_info)
            hs.append(h)

        # middle
        h = self.middle_block(h, t_emb, c_info)

        # up
        for module in self.up_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), t_emb, c_info)

        return self.out(h)

import copy
import torch
from torch import nn

def exists(val):
    return val is not None

def clamp(value, min_value = None, max_value = None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value

class EMA(nn.Module):
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """
    def __init__(
        self,
        model,
        ema_model = None,           # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
        beta = 0.9999,
        update_after_step = 100,
        update_every = 10,
        inv_gamma = 1.0,
        power = 1,
        min_value = 0.0,
        param_or_buffer_names_no_ema = set(),
        ignore_names = set(),
        ignore_startswith_names = set(),
        include_online_model = True  # set this to False if you do not wish for the online model to be saved along with the ema model (managed externally)
    ):
        super().__init__()
        self.beta = beta

        # whether to include the online model within the module tree, so that state_dict also saves it

        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model] # hack

        # ema model

        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = copy.deepcopy(model)
            except:
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()

        self.ema_model.requires_grad_(False)

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if param.dtype in [torch.float, torch.float16]}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if buffer.dtype in [torch.float, torch.float16]}

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema # parameter or buffer

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0]))

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]
    
    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            ma_params.data.copy_(current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value = 0.)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch <= 0:
            return 0.

        return clamp(value, min_value = self.min_value, max_value = self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            ma_params.data.lerp_(current_params.data, 1. - current_decay)

        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            ma_buffer.data.lerp_(current_buffer.data, 1. - current_decay)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_images,
        train_labels,
        val_images,  
        val_labels,  
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_after_step=1e30,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        sample_every=1000,
        save_every=1000,
        results_folder='./results',
        amp=False,
        mixed_precision_type='fp16',
        split_batches=True,
        max_grad_norm=1.,
        y_visual=None,
    ):
        super().__init__()

        # dataset
        self.train_images = train_images
        self.train_labels = train_labels
        
        # store validation data
        self.val_images = val_images
        self.val_labels = val_labels

        # visualize
        self.y_visual = y_visual

        # accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels


        # sampling and training hyperparameters
        self.sample_every = sample_every
        self.save_every = save_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_height = diffusion_model.image_height
        self.image_width  = diffusion_model.image_width
        self.max_grad_norm = max_grad_norm

        # optimizer
        self.opt = AdamW(self.model.parameters(), lr=train_lr, weight_decay=1e-4, eps=1e-5, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, update_after_step=ema_update_after_step, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, training_steps):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        torch.save(data, str(self.results_folder / f'emulator_model-{training_steps}.pt'))

    def load(self, training_steps, return_ema=False):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(str(self.results_folder / f'emulator_model-{training_steps}.pt'), map_location=device)
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])
            if return_ema:
                return self.ema
        if 'version' in data:
            print(f"loading from version {data['version']}")
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        log_filename = os.path.join(self.results_folder, 'emulator_model{}.txt'.format(self.train_num_steps))
        if not os.path.isfile(log_filename):
            logging_file = open(log_filename, "w")
            logging_file.close()
        with open(log_filename, 'a') as file:
            file.write("\n===================================================================================================")
        

        val_raw_mse = 0.0
        val_weighted_loss = 0.0

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    idx = np.random.choice(self.train_labels.shape[0], size=self.batch_size, replace=True)
                    batch_images = self.train_images[idx]
                    batch_images = batch_images.type(torch.float).to(device)
                    batch_labels = self.train_labels[idx]
                    batch_labels = batch_labels.type(torch.float).to(device)

                    with self.accelerator.autocast():
                        loss, _ = self.model(batch_images, classes=batch_labels)
                        
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                if self.step % 500 == 0:
                    self.ema.ema_model.eval()
                    with torch.no_grad():
                        B = self.batch_size 
                        val_imgs = self.val_images.repeat(B, 1, 1, 1).type(torch.float).to(device)
                        val_labels = self.val_labels.repeat(B, 1).type(torch.float).to(device)
                        t_uniform = torch.linspace(0, 999, B, device=device).long()
                        val_loss_weighted_tensor, val_loss_unweighted_tensor = self.ema.ema_model.eval()(val_imgs, classes=val_labels, fixed_t=t_uniform)
                        if torch.is_tensor(val_loss_unweighted_tensor):
                            val_raw_mse = val_loss_unweighted_tensor.mean().item()
                        else:
                            val_raw_mse = val_loss_unweighted_tensor
                            
                        if torch.is_tensor(val_loss_weighted_tensor):
                            val_weighted_loss = val_loss_weighted_tensor.mean().item()
                        else:
                            val_weighted_loss = val_loss_weighted_tensor
                            
                    self.model.train()
                
                # Update Description with Val Loss
                if self.step % 1000 == 0:
                     pbar.set_description(f'loss: {total_loss:.5f} | val_loss: {val_weighted_loss:.5f} | val_raw: {val_raw_mse:.5f}')
                     
                     with open(log_filename, 'a') as file:
                        file.write("\r Step: {}, Loss: {:.5f}, Val_Loss: {:.5f}, Val_raw_MSE: {:.5f}.".format(
                            self.step, total_loss, val_weighted_loss, val_raw_mse))
                else:
                     pbar.set_description(f'loss: {total_loss:.5f}')

                accelerator.wait_for_everyone()
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()
                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()
                    if self.step != 0 and divisible_by(self.step, self.sample_every):
                        self.ema.ema_model.eval()
                        with torch.inference_mode():
                            gen_imgs =  self.ema.ema_model.gaussian_ddim_sample(
                                classes = torch.cat([val_label.to(device), y_visual[0:2]], dim = 0), 
                                shape = (3, 2, self.image_height, self.image_width), 
                                clip_denoised = True, 
                                preset_sampling_timesteps=25, 
                                preset_ddim_sampling_eta=0,
                                save_intermediate=False
                            )
                            gen_imgs = gen_imgs.detach().cpu()
                            if gen_imgs.min() < 0 or gen_imgs.max() > 1:
                                print("\r Generated images are out of range. (min={}, max={})".format(gen_imgs.min(), gen_imgs.max()))
                            utils.save_image(gen_imgs.data, str(self.results_folder) + '/emulator_model_{}.png'.format(self.step), nrow=int(math.sqrt(len(self.y_visual))), normalize=False)
                    if self.step != 0 and divisible_by(self.step, self.save_every):
                        training_steps = self.step
                        self.ema.ema_model.eval()
                        self.save(training_steps)
                pbar.update(1)
        accelerator.print('training complete')
    




# 데이터 로드
train_images = torch.load('/home/newdata/train_images.pt')
train_labels = torch.load('/home/newdata/train_labels.pt')
test_images = torch.load('/home/newdata/test_images.pt')
test_labels = torch.load('/home/newdata/test_labels.pt')
images = torch.cat((train_images, test_images), dim = 0)
labels = torch.cat((train_labels, test_labels), dim = 0)
maximum = images.max()
images = images / maximum

y_visual = test_labels
y_visual = y_visual.to(device)

idx=400
val_image = images[idx].unsqueeze(0)
val_label = labels[idx].unsqueeze(0)
images = torch.cat([images[:idx], images[idx+1:]], dim=0)
labels = torch.cat([labels[:idx], labels[idx+1:]], dim=0)

_, _, H, W = images.shape

binary_images = (images > 0).float()
binary_images = binary_images * 2 - 1
images = images * 2 - 1

val_binary = (val_image > 0).float()
val_binary = val_binary * 2 - 1
val_image = val_image * 2 - 1
val_image = torch.cat((val_image, val_binary), dim=1)

images = torch.cat((images, binary_images), dim=1)
Totalstep = 50000
hyperparams = {
    'train_batch_size': 16,
    'gradient_accumulate_every': 4,
    'train_lr': 3e-4,
    'train_num_steps': Totalstep,
    'ema_update_after_step': 2000,
    'ema_update_every': 10,
    'ema_decay': 0.999,
    'adam_betas': (0.9, 0.99),
    'sample_every': 1000,
    'save_every': 1000,
    'results_folder': '/home/emulator',
    'amp': True,
    'mixed_precision_type': 'fp16',
    'split_batches': True,
    'max_grad_norm': 1.0,
    'y_visual': y_visual,  
}



unet_model = Unet(
    num_classes = 10,
    model_channels = 128,
    out_channels = 2,
    attention_resolutions = (4, 8, 16),
    channel_mult=(1, 2, 4, 8)
)

diffusion_model = DiffusionModel(model=nn.DataParallel(unet_model), image_size=(H, W), timesteps=1000)
optimizer = Adam(diffusion_model.parameters(), lr=hyperparams['train_lr'], betas=hyperparams['adam_betas'])
accelerator = Accelerator(mixed_precision=hyperparams['mixed_precision_type'] if hyperparams['amp'] else 'no')

trainer = Trainer(
    diffusion_model=diffusion_model,
    train_images=images,
    train_labels=labels,
    val_images=val_image, 
    val_labels=val_label, 
    **hyperparams
)

trainer.train()