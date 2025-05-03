import torch
import torch.nn as nn
import torch.nn.functional as F
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from MambaCD.changedetection.models.Spatial_Mamba import SpatialMambaBlock
from einops import rearrange
import numbers

class ChangeDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(ChangeDecoder, self).__init__()

        # Define the VSS Block for Spatio-temporal relationship modelling
        #                hidden_dim=dim,
        self.st_block_41 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        
        self.at_layer_41 = TransformerBlock(dim=128,num_heads=8,ffn_expansion_factor=2.66,bias=False,LayerNorm_type='WithBias')

        self.st_block_42 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),

        )
        self.st_block_43 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),

            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        #self.dim_convert41 = nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1] * 2, out_channels=128)
        self.dim_convert41 = OverlapPatchEmbed(encoder_dims[-1] * 2, 128)
        self.st_block_31 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        #self.dim_convert31 = nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2] * 2, out_channels=128)
        self.dim_convert31 = OverlapPatchEmbed(encoder_dims[-2] * 2, 128)

        self.at_layer_31 = TransformerBlock(dim=128,num_heads=8,ffn_expansion_factor=2.66,bias=False,LayerNorm_type='WithBias')

        self.st_block_32 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_33 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_21 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_22 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_23 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_11 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_12 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_13 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate = 0.,d_state=kwargs['ssm_d_state'], dt_init="random", 
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # Fuse layer  
        self.fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 2, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 2, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 1, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 1, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())

        # Smooth layer
        self.smooth_layer_3 = ICSFBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_2 = ICSFBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_1 = ICSFBlock(in_channels=128, out_channels=128, stride=1) 
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features):

        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        '''
            Stage I
        '''
        p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
        B, C, H, W = pre_feat_4.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
        ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
        p42 = self.st_block_42(ct_tensor_42)

        #ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
        #ct_tensor_43[:, :, :, 0:W] = pre_feat_4
        #ct_tensor_43[:, :, :, W:] = post_feat_4
        #p43 = self.st_block_43(ct_tensor_43)

        #p4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2]], dim=1))
        
        #reshpae使其适合transformer输入
        #print('p41 step 0:', p41.shape)

        cat_input = torch.cat([pre_feat_4, post_feat_4], dim=1)
        p42 = self.dim_convert41(cat_input)


        #batch_size, chan_width, height, width = p42.size()
        #p42 = p42.flatten(2)
        #p42 = p42.permute(0,2,1)
        #p42 = self.at_layer_41(p42)
        #p42 = p42.permute(0,2,1)
        #p42 = p42.view(batch_size, chan_width, height, width)

        p42 = self.at_layer_41(p42)
        p4 = self.fuse_layer_4(torch.cat([p41,p42], dim=1))

        '''
            Stage II
        '''
        p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
        B, C, H, W = pre_feat_3.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
        ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
        p32 = self.st_block_32(ct_tensor_32)

        #ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
        #ct_tensor_33[:, :, :, 0:W] = pre_feat_3
        #ct_tensor_33[:, :, :, W:] = post_feat_3
        #p33 = self.st_block_33(ct_tensor_33)

        #p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2]], dim=1))
        
        #for Transformer block
        p32 = self.dim_convert31(torch.cat([pre_feat_3, post_feat_3], dim=1))
        
        batch_size, chan_width, height, width = p32.size()
        #p32 = p32.flatten(2)
        #p32 = p32.permute(0,2,1)
        #p32 = self.at_layer_31(p32)
        #p32 = p32.permute(0,2,1)
        #p32 = p32.view(batch_size, chan_width, height, width)
        p32 = self.at_layer_31(p32)

        p3 = self.fuse_layer_3(torch.cat([p31, p32], dim=1))
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)
       
        '''
            Stage III
        '''
        p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
        B, C, H, W = pre_feat_2.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
        ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
        p22 = self.st_block_22(ct_tensor_22)

        #ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
        #ct_tensor_23[:, :, :, 0:W] = pre_feat_2
        #ct_tensor_23[:, :, :, W:] = post_feat_2
        #p23 = self.st_block_23(ct_tensor_23)

        #p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2]], dim=1))
        p2 = self.fuse_layer_2(p21)
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)
       
        '''
            Stage IV
        '''
        p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
        B, C, H, W = pre_feat_1.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
        ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
        p12 = self.st_block_12(ct_tensor_12)

        #ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
        #ct_tensor_13[:, :, :, 0:W] = pre_feat_1
        #ct_tensor_13[:, :, :, W:] = post_feat_1
        #p13 = self.st_block_13(ct_tensor_13)

        #p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2]], dim=1))
        p1 = self.fuse_layer_1(p11)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        return p1

   
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ICSFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ICSFBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(out_channels)

        # 通道注意力（轻量SE）
        self.se_fc1 = nn.Conv2d(out_channels, out_channels // 8, kernel_size=1)
        self.se_fc2 = nn.Conv2d(out_channels // 8, out_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn_dw(self.dwconv(out)))

        # Channel Attention（SE模块）
        se = F.adaptive_avg_pool2d(out, 1)
        se = self.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = out * se

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class D_ICSFBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(D_ICSFBlock, self).__init__()
        
        # 对两个输入分别进行卷积
        self.conv_pre = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv_post = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        
        # 融合后空间增强：DWConv
        self.dwconv = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(out_channels)

        # 通道注意力（轻量SE模块）
        self.se_fc1 = nn.Conv2d(out_channels, out_channels // 8, kernel_size=1)
        self.se_fc2 = nn.Conv2d(out_channels // 8, out_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

        # 残差映射
        self.residual = nn.Conv2d(in_channels * 2, out_channels, 1)

    def forward(self, pre_feat, post_feat):
        # 各自处理
        pre = self.relu(self.conv_pre(pre_feat))
        post = self.relu(self.conv_post(post_feat))

        # 差异交互与融合
        diff = torch.abs(pre - post)
        summ = pre + post
        fusion = diff + summ  # 或者 torch.cat([diff, summ], dim=1) + conv

        # 深度卷积增强空间建模
        out = self.relu(self.bn_dw(self.dwconv(fusion)))

        # 通道注意力
        se = F.adaptive_avg_pool2d(out, 1)
        se = self.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = out * se

        # 残差连接
        residual_input = torch.cat([pre_feat, post_feat], dim=1)
        residual = self.residual(residual_input)

        out += residual
        out = self.relu(self.bn(out))
        return out

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        #self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        #self.norm2 = LayerNorm(dim, LayerNorm_type)
        #self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        #x = x + self.attn(self.norm1(x))
        x = x + self.attn(x)

        #x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

