import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from torchsummary import summary
from sobel import Sobel
from thop import profile
from thop import clever_format
# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


class LKA_scalable(nn.Module):
    def __init__(self, indim, dim, size0):
        super().__init__()
        self.conv0 = nn.Conv2d(indim, dim, size0,padding = size0//2,dilation = 1)

        self.conv_amp_3 = DeformConv2d(indim,indim)
 
        self.conv_amp_1 = nn.Conv2d(indim, indim, 1,padding = 1//2,dilation = 1)

        self.conv_pha_3 = DeformConv2d(indim,indim)

        self.conv_pha_1 = nn.Conv2d(indim, indim, 1,padding = 1//2,dilation = 1)



       
    def forward(self, x):

        #x = self.conv0(x)

        
        x_ttf = torch.fft.rfft2(x, dim = (2,3))  

        #x_ttf = x_ttf*self.scale
        amp = torch.abs(x_ttf)  #幅度谱，图像的高低频信息
        pha = torch.angle(x_ttf)  #相位谱,相位谱保留了图像的更多位置信息,复数角度
        amp_fea_3 = self.conv_amp_3(amp)

        amp_fea_1 = self.conv_amp_1(amp) 


        pha_fea_3 = self.conv_pha_3(pha)
        pha_fea_1 = self.conv_pha_1(pha)

        real_13 = (amp_fea_1 * torch.cos(pha_fea_3) + 1e-8) #实部
        imag_31 = amp_fea_3 * torch.sin(pha_fea_1) + 1e-8 #虚部

        real_31 = (amp_fea_3 * torch.cos(pha_fea_1) + 1e-8)
        imag_13 = amp_fea_1 * torch.sin(pha_fea_3) + 1e-8
        s_1, s_2 = x.shape[2],x.shape[3]




        out = torch.complex((real_13+real_31), (imag_31+imag_13)) + 1e-8


        
        x = torch.abs(torch.fft.irfft2(out, s=(s_1, s_2)))
        x = self.conv0(x)

        return x

class CCM_PDC(nn.Module):
    def __init__(self, dim,out_dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.ReLU(), 
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


class PDCBlock(nn.Module):
    def __init__(self, inplane,outplane,n_levels=2):
        super(PDCBlock, self).__init__()
        self.n_levels = n_levels

        #self.conv_local = nn.Conv2d(inplane//n_levels,inplane//n_levels,3,padding = 3//2)
        self.conv_global = nn.Sequential(nn.Conv2d(inplane//n_levels,inplane//n_levels,7,stride=1,padding = 7//2,dilation = 1),
                                         #nn.Conv2d(inplane//n_levels,inplane//n_levels,3,stride=1,padding = 3//2,dilation = 1)
                                         )
        self.spa_ST_fusion = nn.Sequential(nn.Conv2d(inplane//n_levels, inplane//n_levels, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(),
                                     nn.Conv2d(inplane//n_levels, inplane, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.aggr = nn.Conv2d(outplane, outplane, 3, 1, 3//2)

        self.conv_local1 = nn.Conv2d(inplane//n_levels,inplane//n_levels,3,padding = 3//2)
        self.conv_global1 = nn.Sequential(#nn.Conv2d(inplane//n_levels,inplane//n_levels,3,stride=1,padding = 5//2,dilation = 2),
                                         nn.Conv2d(inplane//n_levels,inplane//n_levels,5,stride=1,padding = 5//2,dilation = 1)
                                         )
        self.aggr1 = nn.Conv2d(outplane, outplane, 3, 1, 3//2)

        self.sobel = sobel_gate()


    def forward(self, x_init):

        x_local,x_global = x_init.chunk(self.n_levels,dim =1)
        #x_local = F.relu(self.conv_local(x_local))
        x_global = F.relu(self.conv_global(x_global))

        x_global = F.relu(self.conv_global1(x_global))
        #x = self.aggr1(torch.cat((x_local,x_global),dim=1))
        x = self.spa_ST_fusion(x_local+x_global)

        
        y = x*self.sobel(x_init)
        return y

# CCM
class CCM(nn.Module):
    def __init__(self, dim,out_dim, growth_rate=0.5):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.ReLU(), 
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)

 
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, padding = padding, stride=kernel_size, bias=bias)

    def forward(self, x):
        offset = self.p_conv(x) # (batch, 2N, h/stride, w/stride)
        # (b, 2N, h, w) 获取公式中的坐标p
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        ''' 整形，方便卷积计算  → (batch_size, in_channel, kernel*h, kernel*w)'''
        x_offset = self._reshape_x_offset(x_offset, ks)
        

        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    def _reshape_x_offset(self,x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class sobel_gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.se = Sobel()
    def forward(self,x):
        sx = self.se(x)
        sg = sx + x
        return sg

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 


        self.pdc = PDCBlock(32,32)

    def forward(self, x):
        x = self.pdc(self.norm1(x))
        #x = self.pdc(x)
        return x
        
        
class sr_pan(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, spectral_num=8,init_weights = True):
        super().__init__()
        self.to_feat = nn.Conv2d(spectral_num, dim, 3, 1, 1)

        self.to_pan = nn.Conv2d(1,1,3,1,1)
        self.multi_mean = nn.Conv2d(spectral_num,1,3,1,1)        
        self.norm1 = LayerNorm(spectral_num+1)
        self.feats_1 = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks//2)])
        self.fft = LKA_scalable(1,dim,3)
        self.feats_2 = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks//2)])
        self.to_img = nn.Conv2d(dim,spectral_num,kernel_size=3,padding = 3//2)

        self.fft_sr = LKA_scalable(spectral_num,dim,3)
        self.feats_2_sr = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks//2)])
        self.to_sr = nn.Conv2d(dim,spectral_num,kernel_size=3,padding = 3//2)
        


        if init_weights:
            self._initialize_weights()

    def forward(self, pan,x_init):
        #shallow extraction
        x_1 = self.to_feat(x_init) 

        #pan
        x_mean = self.multi_mean(x_init)
        x_sub = pan-x_mean             
        x_mp = x_sub
        x_fft = self.fft(x_mp)
        

        # #sisr
        sr_fft = self.fft_sr(x_init)
        x_sr = self.feats_1(x_1) + sr_fft

        x_pan = self.feats_1(x_1) + x_fft+sr_fft
        x_pan = self.to_img(self.feats_2(x_pan)+x_1)
        x_sr = self.to_img(self.feats_2_sr(x_sr)+x_1)


        return x_pan,x_sr

    def _initialize_weights(self):  # 初始化权重函数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 初始化偏置为0


if __name__== '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = sr_pan(dim=32, n_blocks=3, ffn_scale=2.0, spectral_num=4).to(device)
    # output_pan=model(torch.randn(1,1,64,64).to(device),torch.randn(1,8,64,64).to(device))
    MACs, params = profile(model, inputs=(torch.randn(1,1,256,256).to(device),torch.randn(1,4,256,256).to(device)))

    # 将结果转换为更易于阅读的格式
    MACs, params = clever_format([MACs, params], '%.3f')

    print(f"运算量：{MACs}, 参数量：{params}")

    # summary(model,[(1,64,64), (8,64,64)])
