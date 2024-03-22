import torch

def pad_time_dimension(x, pad_amount=(1, 1), mode='replicate'):
    b, t, c, h, w = x.shape
    padded_x = torch.zeros(b, t + pad_amount[0] + pad_amount[1], c, h, w, dtype=x.dtype, device=x.device)
    padded_x[:, pad_amount[0]:pad_amount[0] + t, :, :, :] = x
    if mode == 'replicate':
        if pad_amount[0] > 0:
            padded_x[:, :pad_amount[0], :, :, :] = torch.stack([x[:, 0, :, :, :] for i in range(pad_amount[0])], dim=1)
        if pad_amount[1] > 0:
            padded_x[:, pad_amount[0]+t:, :, :, :] = torch.stack([x[:, -1, :, :, :] for i in range(pad_amount[1])], dim=1)
        return padded_x
    else:
        raise NotImplementedError