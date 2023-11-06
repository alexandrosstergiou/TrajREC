from functools import partial
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import tqdm
from .token_masking import TokenMasking
from timm.models.vision_transformer import Block


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos, dtype=np.float32)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class TrajREC(nn.Module):
    def __init__(self, 
                 input_length, 
                 prediction_length, 
                 global_input_dim=4, 
                 local_input_dim=34,
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=4,
                 decoder_embed_dim=512, 
                 decoder_depth=8, 
                 decoder_num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False,
                 lambdas = [1.0,1.0,1.0]):
        super().__init__()
        
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.global_input_dim = global_input_dim
        self.sequence_length = input_length + prediction_length
        
        total_dim = global_input_dim + local_input_dim
        self.masking = TokenMasking(self.input_length, self.prediction_length)
        
        # --------------------------------------------------------------------------
        # encoder specifics
        self.input_embed = nn.Linear(total_dim, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.sequence_length, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        #self.tokens = nn.Parameter(torch.zeros(1, self.sequence_length, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, self.sequence_length, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.sequence_length, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_out_proj = nn.Linear(decoder_embed_dim, total_dim, bias=True) # decoder to patch
        # project global+local coordinates to full coords
        self.decoder_merge_coords = nn.Linear(total_dim, local_input_dim)
        # --------------------------------------------------------------------------
        self.initialize_weights()
        
        self.lambdas = lambdas # local, global, out
        

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], self.sequence_length)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_embed.shape[-1], self.sequence_length)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, setting='future', no_masking=False):
        
        batch_size = x[0].shape[0]
        mask, target = self.masking(x,setting)
        x = x[:2]
        
        # embed patches
        x = torch.cat(x, dim=2)
        x = self.input_embed(x)
        x+= self.pos_embed
        
        if not no_masking:
            x = x * mask

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, target

    def forward_decoder(self, x, mask, foreval=False):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], 1, 1) # B x T_p x C
        
        x_ = x * mask + mask_tokens * abs(1.-mask)

        # add pos embed
        x = x_ + self.decoder_pos_embed

        # Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        pred = self.decoder_out_proj(x)
        
        pred_global = pred[:, :, :self.global_input_dim] 
        pred_local = pred[:, :, self.global_input_dim:]
        pred_out = self.decoder_merge_coords(pred)

        if foreval:
            return pred_global*abs(1.-mask), pred_local*abs(1.-mask), pred_out*abs(1.-mask)
        else:
            return pred_global, pred_local, pred_out, pred_out*abs(1.-mask)
    
    def loss_fn(self,y_pred, y_true, lambda_x, temp_weights=None, nosum=False):        
        mask = (y_true != 0.0).to(torch.int8)
        a = (y_pred - y_true) ** 2
        if temp_weights is not None:
            a = a * temp_weights.to(a.device)
        if nosum:
            return a * mask * lambda_x
        loss = (a * mask).sum() / (mask.sum() + 1e-8)
        return loss * lambda_x

    def forward(self, x, setting, compute_loss=False, beta=1e-3, gamma=1e-1, foreval=False):
        latent, mask, target = self.forward_encoder(x, setting)
        pred = self.forward_decoder(latent, mask, foreval=foreval)  # ([N, T, G=4], [N, T, L=34], [N, T, C=34], (occluded only) [N, T, C=34])
        if compute_loss:
            dlosses = [self.loss_fn(y_true=t, y_pred=p, lambda_x=l) for p,t,l in zip(pred, x, self.lambdas)]
            dlosses.append(self.loss_fn(y_true=target[-1], y_pred=pred[-1], lambda_x=1.))
            with torch.set_grad_enabled(False):
                gt_latent, _, _ = self.forward_encoder(x, setting, no_masking=True)            
            latent = (latent * mask) + 1e-9
            gt_latent = (gt_latent * mask) + 1e-9 
            pl = F.relu(self.loss_fn(y_true=gt_latent, y_pred=latent, lambda_x=1.0, nosum=True))
            sw = abs(torch.tensor(range(1,gt_latent.shape[1]+1), requires_grad=False).unsqueeze(0) - torch.tensor(range(1,gt_latent.shape[1]+1), requires_grad=False).unsqueeze(1)).unsqueeze(0).unsqueeze(-1)
            sl = self.loss_fn(y_true=gt_latent.unsqueeze(1), y_pred=latent.unsqueeze(2), lambda_x=beta, temp_weights=sw.to(dtype=gt_latent.dtype), nosum=True)
            sl = F.relu(torch.sum(sl,1,keepdim=False))
            hl = F.relu(self.loss_fn(y_true=torch.flip(gt_latent,[0]), y_pred=latent, lambda_x=1.0, nosum=True))
            eloss = torch.sum(F.relu(torch.amax(pl-sl-hl+gamma,dim=(1,2)))*1e-3)
            return dlosses, eloss, pred, target
        return pred, target


def trajrec_tiny(**kwargs):
    model = TrajREC(
        embed_dim=192, depth=6, num_heads=4,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=3., norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def trajrec_small1(**kwargs):
    model = TrajREC(
        embed_dim=384, depth=6, num_heads=6,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def trajrec_small(**kwargs):
    model = TrajREC(
        embed_dim=256, depth=6, num_heads=4,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def trajrec_base(**kwargs):
    model = TrajREC(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def trajrec_large(**kwargs):
    model = TrajREC(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def trajrec_huge(**kwargs):
    model = TrajREC(
        embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model




if __name__=='__main__':
    
    model = trajrec_small(input_length=12, global_input_dim=4,
                          local_input_dim=34, prediction_length=6).cuda()
    
    x = (torch.rand(64, 18, 4).cuda(), torch.rand(64, 18, 34).cuda(), torch.rand(64, 18, 34).cuda())
    
    
    dlosses, closs, pred, target = model(x, setting='future',compute_loss=True)
    
    
    print('future setting processed successfully')
    print('pred_global:',pred[0].shape)
    print('pred_local:',pred[1].shape)
    print('pred_out:',pred[2].shape)
    print('target_global:',target[0].shape)
    print('target_local:',target[1].shape)
    print('target_out:',target[2].shape)
    print('-----------------')
    
    pred, target = model(x, setting='past')
    print('past setting processed successfully')
    print('pred_global:',pred[0].shape)
    print('pred_local:',pred[1].shape)
    print('pred_out:',pred[2].shape)
    print('target_global:',target[0].shape)
    print('target_local:',target[1].shape)
    print('target_out:',target[2].shape)
    print('-----------------')
    
    pred, target = model(x, setting='present')
    print('present setting processed successfully')
    print('pred_global:',pred[0].shape)
    print('pred_local:',pred[1].shape)
    print('pred_out:',pred[2].shape)
    print('target_global:',target[0].shape)
    print('target_local:',target[1].shape)
    print('target_out:',target[2].shape)
    print('-----------------')
    
    
    pred, target = model(x, setting='train') 
    print('train setting processed successfully')
    print('pred_global:',pred[0].shape)
    print('pred_local:',pred[1].shape)
    print('pred_out:',pred[2].shape)
    print('target_global:',target[0].shape)
    print('target_local:',target[1].shape)
    print('target_out:',target[2].shape)
    print('-----------------')
    
    # fwd-bwd Multiple tests
    for _ in tqdm.tqdm(range(500)):
        x = (torch.rand(512, 18, 4).cuda(), torch.rand(512, 18, 34).cuda(), torch.rand(512, 18, 34).cuda())
        losses, eloss, pred, target = model(x, setting='train', compute_loss=True)
        loss = sum(losses[:-1])+eloss
        loss.backward()
    print('FWD and BWD runs succeeded!')
        