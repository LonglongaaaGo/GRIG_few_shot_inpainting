import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from models.Restormer import TransformerBlock,OverlapPatchEmbed,Downsample,Upsample
seq = nn.Sequential


def weights_init(m):
    classname = m.__class__.__name__
    # if classname.find('Conv') != -1 or classname.find(
    #             'Linear') !=-1 and hasattr(m, 'weight'):
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)        


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)



def UpBlock_v2(in_planes, out_planes):

    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes*2), GLU())
    return block





def UpBlockComp_v2(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU()
        )
    return block



class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat





class GRIG_G(nn.Module):

    """
    GRIG generator
    """
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(GRIG_G, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        #down
        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[512], nfc[256], 4, 2, 1, bias=False),
                batchNorm2d(nfc[256]),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[256], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))

        self.down_4 = DownBlockComp(nfc[256], nfc[128])
        self.down_8 = nn.Sequential( DownBlockComp(nfc[128], nfc[64]),
                AOTBlock(nfc[64], [1,2,4,8]),AOTBlock(nfc[64], [1,2,4,8]))

        self.down_16 = DownBlockComp(nfc[64], nfc[32])
        self.se_256_16 = SEBlock(nfc[256], 96)
        self.se_128_8 = SEBlock(nfc[128], 192)
        self.se_64_4 = SEBlock(nfc[64], 384)

        self.reformer = self.Restormer_( inp_channels=nfc[32], out_channels=nfc[32] )

        ##up

        self.feat_64 = UpBlock_v2(nfc[32], nfc[64])
        self.conv_64_3_3 = nn.Sequential(
            conv2d(nfc[64] * 2, nfc[64] * 2, 1, 1, 0, bias=False),
            batchNorm2d(nfc[64] * 2), GLU(),
            AOTBlock(nfc[64], [1, 2, 4, 8]),
            AOTBlock(nfc[64], [1,2,4,8]))
        self.feat_128 = UpBlockComp_v2(nfc[64], nfc[128])
        self.conv_128_3_3 = nn.Sequential(
            conv2d(nfc[128] * 2, nfc[128] * 2, 1, 1, 0, bias=False),
            batchNorm2d(nfc[128] * 2), GLU())
        self.feat_256 = UpBlock_v2(nfc[128], nfc[256])

        self.up_se_64 = SEBlock( 384,nfc[64])

        self.up_se_128 = SEBlock(192,nfc[128])
        self.up_se_256 = SEBlock(96,nfc[256])

        self.to_big = conv2d(nfc[im_size], 3, 3, 1, 1, bias=False)

        if im_size > 256:
            self.feat_512 = UpBlockComp_v2(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock_v2(nfc[512], nfc[1024])

    def Restormer_(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 # dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 refinement = True
                 ):


        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement_tag= refinement
        if self.refinement_tag:
            self.refinement = nn.Sequential(*[
                TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, residual,mask):
        imgs = torch.cat([residual,mask],dim=1)

        #down
        down_feat_256 = self.down_from_big(imgs)
        down_feat_128 = self.down_4(down_feat_256)
        down_feat_64 = self.down_8(down_feat_128)

        down_feat_32 = self.down_16(down_feat_64)

        # restormer
        inp_enc_level1 = self.patch_embed(down_feat_32)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)#48,32,32

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)#96,16,16

        out_enc_level2 = self.se_256_16(down_feat_256, out_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)#192,8,8

        out_enc_level3 = self.se_128_8(down_feat_128, out_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)#384
        inp_enc_level4 = self.se_64_4(down_feat_64, inp_enc_level4) #384,4,4


        latent = self.latent(inp_enc_level4)


        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        if self.refinement_tag:
            out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + down_feat_32

        feat_32 = out_dec_level1
        #up

        feat_64 = self.feat_64(feat_32)
        cat_feat_64 = torch.cat([feat_64,down_feat_64],1)
        feat_64 = self.conv_64_3_3(cat_feat_64)
        feat_64 = self.up_se_64(inp_enc_level4,feat_64 )

        feat_128 = self.feat_128(feat_64)

        feat_128 = self.up_se_128(inp_enc_level3,feat_128 )

        cat_feat_128 = torch.cat([feat_128,down_feat_128],1)
        feat_128 = self.conv_128_3_3(cat_feat_128)
        feat_256 = self.feat_256(feat_128)
        feat_256 = self.up_se_256(inp_enc_level2, feat_256)

        if self.im_size == 256:
            return self.to_big(feat_256)

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        if self.im_size == 512:
            return self.to_big(feat_512)

        feat_1024 = self.feat_1024(feat_512)

        # im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return im_1024



class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
        )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2



