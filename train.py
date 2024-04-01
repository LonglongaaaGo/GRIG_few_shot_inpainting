"""
Created by Wanglong Lu on 2022
the training code for GRIG
"""

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from torch.utils import data
import argparse
from tqdm import tqdm
from models.models import weights_init,GRIG_G

from util.utils import print_networks
from util.operation import copy_dirs,copy_Dir2Dir
from pg_modules.discriminator import ProjectedDiscriminator,ForgeryPatchDiscriminator
from util.operation import copy_G_params, load_params, get_dir,get_mask,get_completion
from util.utils import co_mod_mask
from util.operation import ImageFolder,ImageFolder_CenterCrop
from diffaug import DiffAugment,DiffAugment_withsame_trans
import shutil
import gc
import os
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
policy = 'color,translation'
color_policy = 'color'

import Lpips
from Logger.Logger import Logger
from Logger.Scorer import ScoreManager
from test_util import get_metric_score,Reinference_v2

percept = Lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


def Patch_hinge_segmentation_loss_d(mask_01,rec_out,mask_ratio=1):
    """
    forgery patch loss for discriminator
    :param mask_01: input mask tensor
    :param rec_out: the predicted score map
    :param mask_ratio: the weight of mask loss
    :return: loss vaule for discriminator
    """
    mask_10 = 1 - mask_01
    #max real part, min fake part
    loss = F.relu((1 - rec_out) * mask_10).mean()*mask_ratio + F.relu( (1+ rec_out) * mask_01).mean()
    return loss



def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def accumulate(g_module, avg_param_G, decay=0.999):
    for p, avg_p in zip(g_module.parameters(), avg_param_G):
        avg_p.mul_(decay).add_((1-decay) * p.data)


"""
Iterative residual training for GRIG
"""

def train(args):
    #data root for training
    data_root = args.path
    #data root for testing
    test_root = args.test_path
    #reload checkpoint
    checkpoint = args.ckpt
    #
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nc=4
    nlr = 0.0002  #learning rate
    nbeta1 = 0.5
    use_cuda = True
    current_iteration = 0
    saved_model_folder, saved_image_folder = get_dir(args)

    metricsForSave = ["fid","U_IDS_score","P_IDS_score"]
    metrics_dic = {"mae": [999, -1], "psnr": [-999, 1], "ssim": [-999, 1], "fid": [999, -1],
                   "U_IDS_score": [-999, 1], "P_IDS_score": [-999, 1]}

    # frequency for saving model parameters
    save_interval = 500
    save_interval2 = 3333
    # frequency for showing imgs
    show_interval = 1000
    # frequency for testing models
    eval_interval = args.eval_interval
    total_iterations = args.iter

    if args.debug == True:  # if debug
        show_interval = 10
        save_interval = 10
        eval_interval = 20
        total_iterations = 80000

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    # if use multi gpus
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed_gpu = n_gpu>1
    if distributed_gpu:
        print("distributed_gpu!!!")
        if args.local_rank != -1:  # for torch.distributed.launch
            args.local_rank = args.local_rank
            args.current_device = args.local_rank
        elif 'SLURM_LOCALID' in os.environ:  # for sulrm scheduler
            # ngpus_per_node how many gpu can be used in a node
            # ngpus_per_node = torch.cuda.device_count()
            available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',', ""))
            print("available_gpus:", available_gpus)
            # ngpus_per_node how many gpu can be used in a node
            ngpus_per_node = torch.cuda.device_count()
            # local_rank  Which process in a node, local_rank is independent in each node
            args.local_rank = int(os.environ.get("SLURM_LOCALID"))
            # What is the rank among all processes?
            args.rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + args.local_rank

            available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',', ""))
            print("available_gpus:", available_gpus)
            args.current_device = int(available_gpus[args.local_rank])

        import datetime
        torch.cuda.set_device(args.current_device)
        torch.distributed.init_process_group(backend="nccl", init_method="env://",world_size=n_gpu,rank=args.rank,timeout=datetime.timedelta(0,24400))
        synchronize()

    print("now the rank is :",get_rank()) # model init
    # GRIG generator
    netG = GRIG_G(ngf=ngf, nc=nc, nz=nz,im_size=im_size).train()
    netG.apply(weights_init)
    netG.to(device)

    print("augmentation during training aug_train:",args.aug_train)
    print("augmentation during samples aug:",args.aug)
    # projected discriminator
    netD = ProjectedDiscriminator(diffaug=args.aug_train,interp224=(im_size < 224),
                cout = 64,expand = True,proj_type = 2,checkpoint_path=args.efficient_net,
        forgary_aware_tag=False,num_discs = 1,use_separable_discs=False,cond =False).train()
    netD.apply(weights_init)
    netD.to(device)

    #forgery patch discriminator
    patch_D = ForgeryPatchDiscriminator(input_nc=3, ndf=64, n_layers=3)
    patch_D.apply(weights_init)
    patch_D.to(device)
    #
    print_networks(netG,name="netG")
    print_networks(netD,name="netD")
    print_networks(patch_D,name="patch_D")
    #optimizer for each model
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerPatchD = optim.Adam(patch_D.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    avg_param_G = copy_G_params(netG)

    if checkpoint != 'None':
        # reload model checkpoint if there are pre-trained weights
        print("reload from ",checkpoint)
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        patch_D.load_state_dict(ckpt['patch_d'])

        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        optimizerPatchD.load_state_dict(ckpt['opt_patch_d'])

        if args.resume == True:
            current_iteration = int(ckpt['iter'])
            metrics_dic = ckpt['metrics']
            print("resume from checkpoint, the current_iteration is ",current_iteration)

        else:
            current_iteration = 0
        del ckpt

    # if use multi-gpus
    if distributed_gpu:
        netG = nn.parallel.DistributedDataParallel(
            netG,
            device_ids=[args.current_device],
            output_device=args.current_device,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        netD = nn.parallel.DistributedDataParallel(
            netD,
            device_ids=[args.current_device],
            output_device=args.current_device,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        patch_D = nn.parallel.DistributedDataParallel(
            patch_D,
            device_ids=[args.current_device],
            output_device=args.current_device,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    transform_list = transforms.Compose([
        transforms.Resize((int(im_size), int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = ImageFolder(root=data_root, transform=transform_list,im_size=(im_size,im_size))
    test_dataset = ImageFolder_CenterCrop(root=test_root, transform=test_transform,im_size=(im_size,im_size))

    dataloader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=data_sampler(dataset, shuffle=True, distributed=distributed_gpu),
                                 num_workers=args.num_workers, drop_last=True, pin_memory=True)
    dataloader = sample_data(dataloader)
    test_loader = DataLoader(
        test_dataset,batch_size=batch_size,
        sampler=data_sampler(test_dataset, shuffle=False, distributed=False),
        num_workers=args.num_workers, drop_last=False,)

    if get_rank() == 0:
        logger = Logger(path=args.logger_path, continue_=True)
        score_manager = ScoreManager(metrics_dic)

    if distributed_gpu:
        g_module = netG.module
        d_module = netD.module
        p_d_module = patch_D.module
    else:
        g_module = netG
        d_module = netD
        p_d_module = patch_D

    loss_dict = {}

    count =0
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        # if args.debug == True and count>10:break
        count+=1
        ##get data
        real_image_ = next(dataloader)
        real_image_ = real_image_.cuda(non_blocking=True)

        if args.aug == True:
            ##data augmentation  # A new object is generated after amplification, but the gradient still exists
            real_image = DiffAugment(real_image_, policy=policy)
        else:
            real_image = real_image_

        ##get mask
        gin, mask_01, im_in = co_mod_mask(real_image, im_size=im_size)

        residual_input = im_in.detach() #the first input
        out_list = []  #saving outputs from each residual inpainting iteration

        seg_list = []  # predictions from forgery-patch discriminator
        mask_label_list = []  #corresponding mask labels
        for kk in range(args.reinpaint_iter):

            ##get fake data
            residual_out = netG(residual_input.detach(),mask_01.detach())
            pre_imgs = residual_out+residual_input
            # completed_img
            completed_img = get_completion(pred=pre_imgs, gt=real_image, mask_01=mask_01)
            out_list.append(completed_img.detach().clone())

            if args.aug_train == True:  # if use augmentation for adversarial training
                aug_completed_img, aug_mask_01 = DiffAugment_withsame_trans(completed_img,mask_01,policy=policy)
                aug_real_img = DiffAugment(real_image.detach(),policy=policy)
            else:
                aug_completed_img = completed_img
                aug_mask_01 = mask_01
                aug_real_img = real_image

            ## 2. train projected Discriminator
            netD.zero_grad()
            real_pred = netD(aug_real_img.detach(), torch.empty(1).to(device), foragry_tag=False)
            d_loss = F.relu(torch.ones_like(real_pred) - real_pred).mean()

            fake_pred = netD(aug_completed_img.detach(), torch.empty(1).to(device),foragry_tag=False)
            fake_err = F.relu(torch.ones_like(fake_pred) + fake_pred).mean()

            d_loss += fake_err
            loss_dict["D_loss"] = d_loss

            d_loss.backward()
            optimizerD.step()

            ## 3. train forgery-patch Discriminator
            patch_D.zero_grad()
            real_pred = patch_D(aug_real_img.detach())
            p_d_loss = F.relu(torch.ones_like(real_pred) - real_pred).mean()
            fake_pred,mask_label = patch_D(aug_completed_img.detach(),aug_mask_01.detach())
            # forgery patch
            mask_label[mask_label > 0] = 1
            p_d_fake_err = Patch_hinge_segmentation_loss_d(mask_label.detach(),fake_pred,mask_ratio=1)

            seg_list.append(fake_pred.detach())
            mask_label_list.append(mask_label.detach())
            p_d_loss += p_d_fake_err
            loss_dict["forgery_patchD_loss"] = p_d_loss
            p_d_loss.backward()
            optimizerPatchD.step()

            ## 4. train Generator
            netG.zero_grad()
            pred_g = netD(aug_completed_img, torch.empty(1), foragry_tag=False)
            err_g = -pred_g.mean()

            patch_pred_g = patch_D(aug_completed_img)
            patch_err_g = -patch_pred_g.mean()*args.forgery_patch_loss_weight

            err_g+=patch_err_g
            loss_dict["G_loss"] = err_g

            #should calculate every time, otherwiese, the performance is not good
            g_percept_loss = percept(completed_img, real_image.detach()).sum() * 1.5
            loss_dict["g_percept_loss"] = g_percept_loss
            err_g += g_percept_loss

            err_g.backward()
            optimizerG.step()
            accumulate(g_module,avg_param_G)

            residual_input = completed_img.detach()

        # reduced is mast incorprated into the code, otherwise
        #Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
        #will happen
        loss_reduced = reduce_loss_dict(loss_dict)
        D_loss_val = loss_reduced["D_loss"].mean().item()
        G_loss_val = loss_reduced["G_loss"].mean().item()
        g_percept_loss_val = loss_reduced["g_percept_loss"].mean().item()
        forgery_patchD_loss_val = loss_reduced["forgery_patchD_loss"].mean().item()

        if get_rank() == 0 and iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f"%(D_loss_val, -G_loss_val))
            print("g_percept_loss loss d: %.5f   "%(g_percept_loss_val))
            print("  forgery_patchD_loss_val D: %.5f"%( forgery_patchD_loss_val))


        if get_rank() == 0 and (iteration+1) % (show_interval) == 0:
            show_img_in = im_in*(1-mask_01)+mask_01
            vutils.save_image(torch.cat([show_img_in, completed_img,real_image,*out_list]).add(1).mul(0.5),
                              saved_image_folder + '/%d.jpg' % iteration, nrow=batch_size)

            seg_list = [F.interpolate(seg_,size=show_img_in.shape[2],mode="nearest").repeat([1,3,1,1]) for seg_ in seg_list]
            mask_label_list = [F.interpolate(mask_label_,size=show_img_in.shape[2],mode="nearest").repeat([1,3,1,1]) for mask_label_ in mask_label_list]
            vutils.save_image(torch.cat([*seg_list,*mask_label_list]),
                              saved_image_folder + '/seg_%d.jpg' % iteration, nrow=batch_size,range=(0, 1),)


        if get_rank() == 0 and ((iteration + 1) % (save_interval2 + 1) == 0):
            print("saving  current model")
            backup_para = copy_G_params(g_module)
            load_params(g_module, avg_param_G)
            torch.save({
                'g': g_module.state_dict(), 'd': d_module.state_dict(), 'patch_d': p_d_module.state_dict(),
                'g_ema': avg_param_G,
                'opt_g': optimizerG.state_dict(), 'opt_d': optimizerD.state_dict(),
                'opt_patch_d': optimizerPatchD.state_dict(),
                'iter': iteration, 'metrics': score_manager.get_all_dic()},
                saved_model_folder + '/a_recent_model2.pth')
            load_params(g_module, backup_para)



        if get_rank()==0 and (iteration+1)  % (save_interval+1) == 0:
            print("saving  current model")
            backup_para = copy_G_params(g_module)
            load_params(g_module, avg_param_G)
            torch.save({
                        'g':g_module.state_dict(),'d':d_module.state_dict(),'patch_d':p_d_module.state_dict(),'g_ema': avg_param_G,
                'opt_g': optimizerG.state_dict(),'opt_d': optimizerD.state_dict(),'opt_patch_d':optimizerPatchD.state_dict(),
                'iter':iteration,'metrics':score_manager.get_all_dic()},
                saved_model_folder+'/a_recent_model.pth')

            load_params(g_module, backup_para)

        del completed_img,real_image,out_list,seg_list,mask_label_list
        gc.collect()

        if (iteration+1)  % (eval_interval) == 0 :
            print("inference!")
            synchronize()
            if get_rank() == 0:
                backup_para = copy_G_params(g_module)
                load_params(g_module, avg_param_G)
                # test the performance
                Reinference_v2(args, netG, test_loader, args.eval_dir)
                load_params(g_module, backup_para)

            synchronize()
            if get_rank() == 0:
                out_dics = get_metric_score(args, iteration, args.eval_dir, logger, device="cuda")

                tp_dic = {}
                for name in metricsForSave:
                    if score_manager.compare(name,out_dics):
                        tp_dic[name] =1
                    else:
                        tp_dic[name] = 0
                score_manager.update(out_dics)
                for key in tp_dic:
                    if tp_dic[key] == 1:
                        torch.save({
                            'g': g_module.state_dict(), 'd': d_module.state_dict(),'patch_d':p_d_module.state_dict(), 'g_ema': avg_param_G,
                            'opt_g': optimizerG.state_dict(), 'opt_d': optimizerD.state_dict(),'opt_patch_d':optimizerPatchD.state_dict(),
                            'iter': iteration,'metrics':score_manager.get_all_dic()},
                            saved_model_folder + '/%s_best_model.pth'%key)
                        shutil.copy(saved_model_folder + '/%s_best_model.pth'%key,saved_model_folder+'/a_recent_model.pth')
                        copy_Dir2Dir(args.eval_dir,args.eval_best_dir,num=120)
            synchronize()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRIG trainer')
    ### train
    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=2000000, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local rank for distributed training")
    parser.add_argument("--efficient_net", type=str, default="/home/onelong/Longlongaaago/pre-train/tf_efficientnet_lite0-0aa007d2.pth", help="psp model pretrained model")

    parser.add_argument("--logger_path", type=str, default="./logger.txt", help="path to the output the generated images")
    parser.add_argument("--num_workers",type =int, default = 2,help = "numworkers for dataloader")
    parser.add_argument("--resume",type=bool,default=False,help="reload => False, resume = > True ",)

    parser.add_argument("--aug",type=bool,default=True,help="augmentation in data sample ",)
    parser.add_argument("--aug_train",type=bool,default=False,help="augmentation for adversarial training ",)
    parser.add_argument("--reinpaint_iter",type=int,default=3,help="number of re inpaint ",)

    #loss
    parser.add_argument("--forgery_patch_loss_weight", type=float, default=1, help='forgery patch loss weight')
    parser.add_argument('--eval_interval', type=int, default=10000)

    ### test
    parser.add_argument("--debug",type=bool,default=False, help="Debug ")
    parser.add_argument('--test_path', type=str, default='../lmdbs/art_landscape_1k',
                        help='for test, path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument("--eval_dir", type=str, default="./eval_dir", help="path to the output the generated images")
    parser.add_argument("--eval_best_dir", type=str, default="./eval_best_dir", help="path to the output the generated images")


    ###fid
    args = parser.parse_args()
    print(args)

    #train and test
    train(args)
