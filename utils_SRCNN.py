import torch
import numpy as np
import torch.nn.functional as F
import logging
import math

#SAM and ERGAS calculation
def compute_index(img_base, img_out, ratio=1):
    h = img_out.shape[0]
    w = img_out.shape[1]
    chanel = img_out.shape[2]
    # 计算SAM
    sum1 = torch.sum(img_base * img_out, 2)
    sum2 = torch.sum(img_base * img_base, 2)
    sum3 = torch.sum(img_out * img_out, 2)
    t = (sum2 * sum3) ** 0.5
    numlocal = torch.gt(t, 0)
    num = torch.sum(numlocal)
    t = sum1 / t
    angle = torch.acos(t)
    sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
    if num == 0:
        averangle = sumangle
    else:
        averangle = sumangle / num
    SAM = averangle * 180 / 3.14159256

    # 计算ERGAS
    summ = 0
    for i in range(chanel):
        a1 = torch.mean((img_base[:, :, i] - img_out[:, :, i]) ** 2)
        m1 = torch.mean(img_base[:, :, i])
        a2 = m1 * m1
        summ = summ + a1 / a2
    ERGAS = 100 * (1 / ratio) * ((summ / chanel) ** 0.5)

    return SAM, ERGAS
# PSNR 计算
def calc_mean_psnr(img1,img2):
    PSNR = 10 * torch.log10(math.pow(1.0, 2) / torch.mean((img1-img2)**2, [0, 1]))
    return torch.mean(PSNR)


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2
    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)
    
def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=1023, size_average=True, full=False):
    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def calc_ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False):

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=True,
                         full=False)
    if size_average:
        ssim_val = ssim_val.mean()
        #cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

# 计算 平均数，求和，长度
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

if __name__ == "__main__":
    shape = (64,64)
    t = make_coord(shape,flatten = False)
    print(t.shape)