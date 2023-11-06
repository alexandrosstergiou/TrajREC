import torch
from torch import nn

class TokenMasking(nn.Module):
    def __init__(self, 
                 input_length, 
                 prediction_length):
        super().__init__()
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.sequence_length = self.input_length + self.prediction_length 
              
        
    def forward(self, x, setting, shuffle=True):
        batch_mask = torch.ones((x[0].shape[0],self.sequence_length,1), requires_grad=False)
        device = x[0].device
        if setting=='future' or 'train' in setting:
            mask_future = torch.nn.functional.one_hot(torch.tensor([i for i in range(self.input_length)], requires_grad=False),self.sequence_length).sum(0).unsqueeze(0).unsqueeze(-1)
            if not 'train' in setting:
                mask = mask_future
                batch_mask = batch_mask.to(device) * mask.to(device)
        if setting=='past' or 'train' in setting:
            mask_past = torch.nn.functional.one_hot(torch.tensor([i for i in range(self.prediction_length,self.sequence_length)], requires_grad=False),self.sequence_length).sum(0).unsqueeze(0).unsqueeze(-1)
            if not 'train' in setting:
                mask = mask_past
                batch_mask = batch_mask.to(device) * mask.to(device)
        if setting=='present' or 'train' in setting:
            half = self.input_length//2
            mask_traj_right = torch.nn.functional.one_hot(torch.tensor([i for i in range(half)], requires_grad=False),self.sequence_length).sum(0).unsqueeze(0).unsqueeze(-1)
            mask_traj_left = torch.nn.functional.one_hot(torch.tensor([i for i in range(half+self.prediction_length,self.sequence_length)], requires_grad=False),self.sequence_length).sum(0).unsqueeze(0).unsqueeze(-1)
            mask_present = mask_traj_right + mask_traj_left
            if not 'train' in setting:
                mask = mask_present
                batch_mask = batch_mask.to(device) * mask.to(device)
        if setting=='train':
            x = x
            i1 = int(batch_mask.shape[0]*0.33)
            i2 = int(batch_mask.shape[0]*0.66)
            batch_mask[:i1] *= mask_future
            batch_mask[i1:i2] *= mask_past
            batch_mask[i2:] *= mask_present
            if shuffle:
                indices = torch.randperm(x[0].shape[0])
                batch_mask = batch_mask[indices]
            batch_mask = batch_mask.to(device)
        target = [x[0] * abs(1.-batch_mask), x[1] * abs(1.-batch_mask), x[2] * abs(1.-batch_mask)]
            
        return batch_mask, target