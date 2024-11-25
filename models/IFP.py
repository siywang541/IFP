from functools import partial
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import models.modules.Uformer_pretrain_arch as Uformer_pretrain_arch
import models.modules.Restormer_Backbone_arch as Restormer_Backbone_arch

import models.modules.RCAN_Pretrain_Head_arch as RCAN_Pretrain_Head_arch

from .frequency_loss import FrequencyLoss1,FrequencyLoss2,FrequencyLoss3, FrequencyLoss4
from .swin_transformer import SwinTransformer
from .utils import get_2d_sincos_pos_embed
from .vision_transformer import VisionTransformer
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
logger = logging.getLogger('base')


class IFP(nn.Module):
    def __init__(self, encoder, encoder_stride, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.decoder = decoder
        assert config.DATA.FILTER_TYPE in ['IFP', 'sr', 'deblur', 'denoise']
        assert config.MODEL.RECOVER_TARGET_TYPE in ['m_m_masked','m_masked','masked', 'normal']
        self.filter_type = config.DATA.FILTER_TYPE
        self.mask_radius1 = config.DATA.MASK_RADIUS1
        self.mask_radius2 = config.DATA.MASK_RADIUS2
        self.recover_target_type = config.MODEL.RECOVER_TARGET_TYPE
        self.criterion1 = FrequencyLoss1(
            loss_gamma=config.MODEL.FREQ_LOSS.LOSS_GAMMA,
            matrix_gamma=config.MODEL.FREQ_LOSS.MATRIX_GAMMA,
            patch_factor=config.MODEL.FREQ_LOSS.PATCH_FACTOR,
            ave_spectrum=config.MODEL.FREQ_LOSS.AVE_SPECTRUM,
            with_matrix=config.MODEL.FREQ_LOSS.WITH_MATRIX,
            log_matrix=config.MODEL.FREQ_LOSS.LOG_MATRIX,
            batch_matrix=config.MODEL.FREQ_LOSS.BATCH_MATRIX).cuda()
        self.criterion2 = FrequencyLoss2(
            loss_gamma=config.MODEL.FREQ_LOSS.LOSS_GAMMA,
            matrix_gamma=config.MODEL.FREQ_LOSS.MATRIX_GAMMA,
            patch_factor=config.MODEL.FREQ_LOSS.PATCH_FACTOR,
            ave_spectrum=config.MODEL.FREQ_LOSS.AVE_SPECTRUM,
            with_matrix=config.MODEL.FREQ_LOSS.WITH_MATRIX,
            log_matrix=config.MODEL.FREQ_LOSS.LOG_MATRIX,
            batch_matrix=config.MODEL.FREQ_LOSS.BATCH_MATRIX).cuda()
        self.criterion3 = FrequencyLoss4(
            loss_gamma=config.MODEL.FREQ_LOSS.LOSS_GAMMA,
            matrix_gamma=config.MODEL.FREQ_LOSS.MATRIX_GAMMA,
            patch_factor=config.MODEL.FREQ_LOSS.PATCH_FACTOR,
            ave_spectrum=config.MODEL.FREQ_LOSS.AVE_SPECTRUM,
            with_matrix=config.MODEL.FREQ_LOSS.WITH_MATRIX,
            log_matrix=config.MODEL.FREQ_LOSS.LOG_MATRIX,
            batch_matrix=config.MODEL.FREQ_LOSS.BATCH_MATRIX).cuda()
        if self.filter_type == 'sr':
            self.sr_factor = config.DATA.SR_FACTOR
            self.sr_mode = config.DATA.INTERPOLATION
        self.normalize_img = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

        if self.decoder is None:
            if config.MODEL.TYPE == 'SSIE':
                self.num_features = 64
                self.decoder = nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.num_features,
                        out_channels=self.num_features//4, kernel_size=1),
                    nn.Conv2d(
                        in_channels=self.num_features//4,
                        out_channels=3, kernel_size=1),
                ) 
            elif config.MODEL.TYPE == 'swinIR' or config.MODEL.TYPE == 'restormer' or config.MODEL.TYPE == 'promptIR':
                self.decoder = RCAN_Pretrain_Head_arch.MSRResNet_Head(in_c=config.MODEL.DECODER.IN_NC, 
                out_c=config.MODEL.DECODER.OUT_NC, 
                scale=config.MODEL.DECODER.UPSCALE, 
                require_modulation=config.MODEL.DECODER.REQUIRE_MODULATION)

            elif config.MODEL.TYPE == 'uformer':
                self.decoder = Uformer_pretrain_arch.Uformer_Decoder(img_size=config.MODEL.UFORMER.IMAGE_SIZE, embed_dim=32,depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False,
                 degradation_embed_dim = 512, require_modulation=True)
            else:
                self.decoder = nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.encoder.num_features,
                        out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
                    nn.PixelShuffle(self.encoder_stride),
                )

        if config.MODEL.TYPE == 'restormer' :
            self.in_chans = config.MODEL.RESTORMER.INP_CHANNELS
            self.patch_size = config.MODEL.BATCH_SIZE
        elif config.MODEL.TYPE == 'promptIR' :
            self.in_chans = config.MODEL.RESTORMER.INP_CHANNELS
            self.patch_size = config.MODEL.BATCH_SIZE
        elif config.MODEL.TYPE == 'swinIR' :
            self.in_chans = config.MODEL.SWINIR.INP_CHANNELS
            self.patch_size = config.MODEL.BATCH_SIZE 
        elif config.MODEL.TYPE == 'uformer' :
            self.in_chans = config.MODEL.UFORMER.INP_CHANNELS
            self.patch_size = config.MODEL.BATCH_SIZE 
        else:
            self.in_chans = self.encoder.in_chans
            self.patch_size = self.encoder.patch_size



    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        checkpoint = torch.load(load_path)

        # load pretrained model

        load_net = checkpoint['model']
        if any([True if 'encoder.' in k else False for k in load_net.keys()]):
            checkpoint_model = {k.replace('encoder.', ''): v for k, v in load_net.items() if k.startswith('encoder.')}
            print('Detect pre-trained model, remove [encoder.] prefix.')
        else:
            print('Detect non-pre-trained model, pass without doing anything.')

        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v

        network.load_state_dict(load_net_clean, strict=strict)
                   
            
            
    def frequency_transform(self, x, mask):
        # 2D FFT
        x_freq = torch.fft.fft2(x)
        # shift low frequency to the center
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
        # mask a portion of frequencies
        
        x_freq_masked = x_freq * mask
        # restore the original frequency order
        x_freq_masked = torch.fft.ifftshift(x_freq_masked, dim=(-2, -1))
        # 2D iFFT (only keep the real part)
        x_corrupted = torch.fft.ifft2(x_freq_masked).real
        x_corrupted = torch.clamp(x_corrupted, min=0., max=1.)
        return x_corrupted

    def frequency_m_transform(self, x, mask):
        # 2D FFT
        x_freq = torch.fft.fft2(x)
        # shift low frequency to the center
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
        magnitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)
        magnitude_masked = magnitude * mask
        img_mandp = magnitude_masked*np.e**(1j*phase)
        # restore the original frequency order
        x_freq_masked = torch.fft.ifftshift(img_mandp, dim=(-2, -1))
        # 2D iFFT (only keep the real part)
        x_corrupted = torch.abs(torch.fft.ifft2(x_freq_masked))
        x_corrupted = torch.clamp(x_corrupted, min=0., max=1.)
        return x_corrupted

    def interpolate_transform(self, x, scale_factor, mode='bicubic'):
        H, W = x.shape[2:]
        down_x = F.interpolate(x, size=(H // scale_factor, W // scale_factor), mode=mode)
        down_x = down_x.clamp(min=0., max=1.)
        up_x = F.interpolate(down_x, size=(H, W), mode=mode)
        up_x = up_x.clamp(min=0., max=1.)
        return up_x

    def forward(self, x, x_lq=None, mask=None):
        if self.filter_type in ['sr', 'deblur', 'denoise']:
            if self.filter_type == 'sr':
                x_lq = self.interpolate_transform(x, self.sr_factor, self.sr_mode)
            assert x_lq is not None
            x_lq = self.normalize_img(x_lq)
        else:
            assert mask is not None
            mask = mask.unsqueeze(1)
            if self.recover_target_type == 'm_masked' or self.recover_target_type == 'm_m_masked':   # m mask
                x_corrupted = self.frequency_m_transform(x, mask)
            else:
                x_corrupted = self.frequency_transform(x, mask)  

            x_corrupted = self.normalize_img(x_corrupted)

        x = self.normalize_img(x)
        x_in = x.clone()
        if self.filter_type in ['sr', 'deblur', 'denoise']:
            z = self.encoder(x_lq, None)
        else:
            z = self.encoder(x_corrupted)
        x_rec = self.decoder(z)
        x_out = x_rec.clone()
        if self.recover_target_type == 'masked':
            loss_recon = self.criterion1(x_rec, x)
            loss = (loss_recon * (1 - mask.unsqueeze(1))).sum() / (1 - mask).sum() / self.in_chans / loss_recon.shape[1]
        elif self.recover_target_type == 'm_masked':
            loss_recon = self.criterion2(x_rec, x)
            loss = (loss_recon * (1 - mask.unsqueeze(1))).sum() / (1 - mask).sum() / self.in_chans / loss_recon.shape[1]
        elif self.recover_target_type == 'm_m_masked':
            loss_recon1,loss_recon2 = self.criterion3(x_rec, x)
            loss = (loss_recon1.sum()/((1 - mask).sum() + mask.sum()) + (loss_recon2 * (1 - mask.unsqueeze(1))).sum() / (1 - mask).sum()) / self.in_chans / loss_recon1.shape[1]     
        elif self.recover_target_type == 'normal':
            loss_recon = self.criterion1(x_rec, x)
            loss = loss_recon.mean()
        else:
            raise NotImplementedError
        return loss ,x_in,x_corrupted,x_out

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_IFP(config):
    model_type = config.MODEL.TYPE


    if model_type == 'restormer':
        encoder = Restormer_Backbone_arch.Restormer_Backbone(
        inp_channels = config.MODEL.RESTORMER.INP_CHANNELS, 
        out_channels = config.MODEL.RESTORMER.OUT_CHANNELS, 
        dim = config.MODEL.RESTORMER.DIM,
        num_blocks = config.MODEL.RESTORMER.NUM_BLOCKS, 
        num_refinement_blocks = config.MODEL.RESTORMER.NUM_REFINEMENT_BLOCKS,
        heads = config.MODEL.RESTORMER.HEADS,
        ffn_expansion_factor = config.MODEL.RESTORMER.FFN_EXPANSION_FACTOR,
        bias = config.MODEL.RESTORMER.BIAS,
        global_residual = config.MODEL.RESTORMER.GLOBAL_RESIDUAL,
        LayerNorm_type = config.MODEL.RESTORMER.LAYERNORM_TYPE,   
        dual_pixel_task = config.MODEL.RESTORMER.DUAL_PIXEL_TASK         
        )
        encoder_stride = 32
        decoder = None
    elif model_type == 'uformer':
        encoder =  Uformer_pretrain_arch.Uformer_Encoder(img_size=config.MODEL.UFORMER.IMAGE_SIZE, embed_dim=32, depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True)
        encoder_stride = 32
        decoder = None
    
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")


    model = IFP(encoder=encoder, encoder_stride=encoder_stride, decoder=decoder, config=config)

    return model
