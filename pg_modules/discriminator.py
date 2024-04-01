from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pg_modules.blocks import DownBlock, DownBlockPatch, conv2d,NormLayer,GLU,UpBlockSmall
from models.ExternalAttention import External_attention
from pg_modules.projector import F_RandomProj
from pg_modules.diffaug import DiffAugment


class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        return self.main(x)



class SingleDisc_patchOut(nn.Module):
    """
    去掉了最后一次层
    然后加入 class activate map
    """
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False,CAM ="one"):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2
        if CAM == "one":
            layers.append(conv2d(nfc[end_sz], 1, 5, 1, 2, bias=False))
        else:
            layers.append(conv2d(nfc[end_sz], 2, 5, 1, 2, bias=False))

        # layers.append(conv2d(nfc[end_sz], 2, 1, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)
        # self.linear = linear(nfc[end_sz], 2,bias=True)



    def forward(self, x, CAM="two"):
        feature = self.main(x)

        out = F.avg_pool2d(feature,kernel_size=(feature.shape[2],feature.shape[3]),ceil_mode=True).view(x.shape[0],-1)
        # out = self.linear(x)
        if CAM == "one":
            return out, feature
        if CAM == "two":
            fake_cam = feature[:,0,:,:]
            real_cam = feature[:,1,:,:]
            # fake_weight = self.linear.weight[0].unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat([4,1,1,1])
            # fake_feature = fake_weight*feature
            # fake_feature = torch.sum(fake_feature,keepdim=1)
            # real_weight = self.linear.weight[1].unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat([4,1,1,1])
            # real_feature = real_weight*feature
            # cam_feature = torch.cat([fake_feature,real_feature],dim = 1)
            return out, fake_cam,real_cam
        return out



class SingleDisc_seg(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        up_layers = []
        layers = []
        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

            up_layers += [conv2d(nfc[256], 1, 3, 1, 1, bias=False),]
        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        UP = partial(UpBlockSmall)

        up_layers.append(conv2d(nfc[start_sz], 1, 3, 1, 1, bias=False))
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            up_layers.append(UP( nfc[start_sz//2],nfc[start_sz]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 3, 1, 1, bias=False))
        up_layers+=[nn.LeakyReLU(0.2, inplace=True),conv2d(1, nfc[end_sz], 3, 1, 1, bias=False)]

        # up_layers.append(conv2d(1, nfc[end_sz], 4, 1, 0, bias=False))

        up_layers.reverse()
        self.up_main = nn.Sequential(*up_layers)

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        logit = self.main(x)
        seg_out = self.up_main(logit)
        return logit,seg_out


class SingleDiscCond(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False, c_dim=1000, cmap_dim=64, embedding_dim=128):
        super().__init__()
        self.cmap_dim = cmap_dim

        # midas channels
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)

        # additions for conditioning on class information
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
        self.embed = nn.Embedding(num_embeddings=c_dim, embedding_dim=embedding_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(self.embed.embedding_dim, self.cmap_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)

        # conditioning via projection
        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out



class ForgeryPatchDiscriminator(nn.Module):
    """Defines a ForgeryPatchDiscriminator """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=None):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ForgeryPatchDiscriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        receptive_filed = [nn.MaxPool2d(kernel_size=kw,stride=2,padding=padw)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True),
                # norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            receptive_filed+= [nn.MaxPool2d(kernel_size=kw,stride=2,padding=padw)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True),
            # norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        receptive_filed += [nn.MaxPool2d(kernel_size=kw, stride=1, padding=padw)]

        sequence += [conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        receptive_filed += [nn.MaxPool2d(kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)
        self.receptive_patch = nn.Sequential(*receptive_filed)
    def forward(self, input,mask=None):
        """Standard forward."""

        out = self.model(input)
        if mask is not None:
            receptive_mask = self.receptive_patch(mask)
            return out,receptive_mask

        return out



class Forgery_aware_v2(nn.Module):
    def __init__(self,channels=[],outchannel=1):
        super().__init__()
        def norm_block(in_planes, out_planes):
            block = nn.Sequential(
                conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
                NormLayer(out_planes * 2), GLU())
            return block

        self.down = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        channels.reverse()
        self.ups = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.external_blocks = nn.ModuleList()
        for i in range(len(channels)):
            if (i == len(channels)-1):
                self.ups.append(UpBlockSmall(channels[i], int(channels[i]//2)))
                self.norms.append(norm_block(channels[i], int(channels[i]//2)))
                self.external_blocks.append(External_attention(channels[i]))
                # self.ups.append(upBlock(channels[i], channels[i]))

            else:
                # self.ups.append(upBlock(channels[i],channels[i+1]))
                self.ups.append(UpBlockSmall(channels[i], int(channels[i+1]//2)))
                self.norms.append(norm_block(channels[i], int(channels[i+1]//2)))
                self.external_blocks.append(External_attention(channels[i+1]))


        self.main = nn.Sequential(
            UpBlockSmall(channels[-1], int(channels[-1])),
            conv2d(int(channels[-1]), int(channels[-1]//2), 3, stride=1, padding=1, bias=False),
            NormLayer(int(channels[-1]//2)),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(int(channels[-1] // 2), int(channels[-1] // 4), 3, stride=1, padding=1, bias=False),
            NormLayer(int(channels[-1] // 4)),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(int(channels[-1] // 4), int(outchannel), 3, stride=1, padding=1, bias=False),
            # NormLayer(int(channels[-1] // 8)),
            # GLU(),
            # conv2d(int(channels[-1] // 16), int(outchannel), 3, stride=1, padding=1,bias=False),
        )

    def forward(self,inputs):
        inputs = [inputs[key] for key in inputs]

        in_features = inputs[-1]
        in_features = self.down(in_features)
        inputs.append(in_features)
        inputs.reverse()
        for i in range(len(inputs)-1):
            out = self.ups[i](in_features)
            norm_out = self.norms[i](inputs[i+1])
            in_features = torch.cat([out,norm_out],dim=1)
            in_features = self.external_blocks[i](in_features)
            # if (i != len(self.ups)-1):
            #     norm = self.norms[i](out)
            #     # in_features = torch.cat([out,inputs[i+1]],dim=1)
            #     in_features = out+inputs[i+1]

        # del in_features
        out = in_features
        out = self.main(out)
        return out


class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=1,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        separable=False,
        patch=False,
        patchout = False,
        CAM="two",
        **kwargs,
    ):
        super().__init__()
        assert num_discs in [1, 2, 3, 4]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc
        Disc = SingleDisc_patchOut if patchout else SingleDisc
        self.patchout = patchout
        self.CAM = CAM
        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            if self.patchout:
                mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch,CAM=CAM)],
            else:
                mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch)],

        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, CAM="two"):
        all_logits = []
        fake_cams = []
        real_cams = []
        for k, disc in self.mini_discs.items():
            if CAM == "one":
                logits, real_cam = disc(features[k], CAM)
                all_logits.append(logits)
                real_cams.append(real_cam)
            elif CAM == "two":
                logits,fake_cam,real_cam = disc(features[k], CAM)
                all_logits.append(logits)
                fake_cams.append(fake_cam)
                real_cams.append(real_cam)
            else:
                all_logits.append(disc(features[k], CAM).view(features[k].size(0), -1))

        if self.patchout == True:
            all_logits = torch.cat(all_logits, dim=0)
        else:
            all_logits = torch.cat(all_logits, dim=1)

        if CAM == "one":
            #out class activate maps
            real_cams = torch.cat((real_cams), dim=1)
            return all_logits,real_cams
        elif CAM == "two":
            fake_cams = torch.cat((fake_cams), dim=0).unsqueeze(dim=1)
            real_cams = torch.cat((real_cams), dim=0).unsqueeze(dim=1)
            return all_logits, fake_cams, real_cams

        return all_logits



class ProjectedDiscriminator(torch.nn.Module):
    def __init__(
        self,
        diffaug=True,
        interp224=True,
        backbone_kwargs={},
        checkpoint_path =None,
        forgary_aware_tag = False,
        num_discs = 4,
        patchout= False,
        CAM = "tow",
        **kwargs
    ):
        super().__init__()
        self.diffaug = diffaug
        self.interp224 = interp224
        self.feature_network = F_RandomProj(checkpoint_path=checkpoint_path,**backbone_kwargs)
        #freeze the feature network
        self.feature_network.requires_grad_(False).eval()
        self.discriminator = MultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,num_discs=num_discs,
            patchout = patchout,
            CAM = CAM,
            **backbone_kwargs,
        )
        self.CAM = CAM
        self.forgaryNet = None
        if forgary_aware_tag == True:
            # self.forgaryNet = Forgery_aware(inchannel=self.feature_network.CHANNELS[0],outchannel=1)
            self.forgaryNet = Forgery_aware_v2(channels=self.feature_network.Pretrain_CHANNELS,outchannel=1)
    def train(self, mode=True):
        self.feature_network = self.feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)


    def forward(self, x, CAM,foragry_tag=False,pooling=False):
        #if we use the foragry aware, the augmentation will misled the seg mask
        if self.diffaug and foragry_tag == False:
            x = DiffAugment(x, policy='color,translation,cutout')

        if self.interp224:
            x = F.interpolate(x, 224, mode='bilinear', align_corners=False)

        features,pre_features = self.feature_network(x)

        if CAM == "one":
            #输出张量为b,1,h,w
            logits, real_cams = self.discriminator(features, CAM)
            return logits,real_cams
        elif CAM == "two":
            #输出张量为b,1,h,w  分别包含了两者的 tensor ==> fake and real
            logits,fake_cams,real_cams = self.discriminator(features, CAM)
            return logits,fake_cams,real_cams
        else:
            logits = self.discriminator(features, CAM)


        if self.forgaryNet !=None:
            # seg_out = self.forgaryNet(features['0'])
            seg_out = self.forgaryNet(pre_features)
            return logits,seg_out

        del pre_features

        if pooling ==True:
            logits = logits.mean(-1)

        return logits



