import torch
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from util.operation import ImageFolder_with_mask,ImageFolder_test
import os
import argparse
from tqdm import tqdm
from models.models import GRIG_G
from torchvision import transforms
from util.operation import get_completion
import random
import numpy as np
import shutil
from test_util import get_metrics_with_lpips
from util.extract2 import MoveTotheSingalDir
import subprocess
import util.utils_train as ut
from util.utils import print_networks
import cv2
from Xlsx_save.xlsx_saver import Xlsx_saver


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def delete_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path,ignore_errors=True)
    return


def select_model(name,im_size):
    ngf = 64
    nz = 256
    nc = 4
    netG = GRIG_G(ngf=ngf, nc=nc, nz=nz,im_size=im_size).train()
    print(netG)
    print_networks(netG,name=name)
    return netG




if __name__ == "__main__":
    """
    GRIG  test 
    show intermediate results for better evaluation
    """

    parser = argparse.ArgumentParser(
        description='GRIG test procedure'
    )
    parser.add_argument('--ckpt_path', type=str, help='the path of the best checkpoint')
    parser.add_argument('--model_name', type=str, default='GRIG')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--reinpaint_iter', type=int, default=3)

    parser.add_argument('--test_path', type=str, default='../lmdbs/art_landscape_1k', help='for test, path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--mask_file_root', type=str, default='/home/k/Data/mask/mask', help='number of iterations')
    parser.add_argument('--mask_root', type=str, default='/home/k/Data/mask/mask/testing_mask_dataset', help='number of iterations')

    parser.add_argument('--eval_dict', type=str, default='./eval', help='path for eval')
    parser.add_argument('--view_dict', type=str, default='./view', help='path for eval')

    parser.add_argument("--debug", type=bool, default=False,)
    parser.add_argument("--view_inter", type=bool, default=False, help="show intermediate results if turn to True, else show the best results")
    parser.add_argument('--view_number', type=int, default=1000, help="how many images will be copied ")
    parser.add_argument('--device', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--im_size', type=int, default=256)
    parser.add_argument('--batch', default=1, type=int, help='batch size')

    parser.add_argument('--mask_type', default="Center", type=str, help='mask type for inference you can use ["Center","test_2.txt","test_3.txt","test_4.txt","test_5.txt","test_6.txt"]')
    parser.add_argument('--show_final', action='store_true', help='if or not show the intermediate results')

    args = parser.parse_args()
    if args.mask_type == "all":
        mask_types = ["Center","test_2.txt","test_3.txt","test_4.txt","test_5.txt","test_6.txt",]
    else:
        mask_types = [args.mask_type,]

    transform_list =  transforms.Compose([
        transforms.Resize((int(args.im_size), int(args.im_size))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    metrics_saver = Xlsx_saver("metrics_all")

    delete_dirs(args.eval_dict)
    os.makedirs(args.eval_dict,exist_ok=True)
    for jjj,mask_type in enumerate(mask_types):
        set_random_seed(1)
        print("#####################################seed:", np.random.rand(1))
        print("#######" * 10)
        print("for mask :", mask_type)

        # set data loader
        if mask_type == "Center":
            dataloader = DataLoader(
                ImageFolder_test(args.test_path, transform=transform_list,
                                 im_size=(args.im_size, args.im_size)),
                batch_size=args.batch, shuffle=False,drop_last=False, num_workers=args.num_workers)
        else:
            args.mask_file = os.path.join(args.mask_file_root, mask_type)
            data = ImageFolder_with_mask(args.test_path, args.mask_root, args.mask_file, transform=transform_list,
                                         train_tag=False, im_size=(args.im_size, args.im_size))
            dataloader = DataLoader(data, batch_size=args.batch, shuffle=False,drop_last=False, num_workers=args.num_workers)

        mask_type_name = str(mask_type).strip().split(".")[0]  # get mask type name
        # create sub dir
        sub_outdir = os.path.join(args.eval_dict, mask_type_name)
        os.makedirs(sub_outdir,exist_ok=True)

        print('load checkpoint from:',args.ckpt_path)
        netG = select_model(args.model_name,args.im_size).to(args.device)
        checkpoint = torch.load(args.ckpt_path)
        netG.load_state_dict(checkpoint['g'])
        load_params(netG, checkpoint['g_ema'])
        netG.eval()

        with torch.no_grad():
            for i,datas in tqdm(enumerate(dataloader)):
                # if i >10 and args.debug ==True: break

                real_image,mask_01 = datas
                real_image = real_image.to(args.device)
                mask_01 = mask_01.to(args.device).float()

                im_in = real_image * (1 - mask_01)

                residual_input = im_in
                completed_list = []

                # create sub dir store ground_truth images
                gt_folder =os.path.join(sub_outdir, "gt_folder")
                os.makedirs(gt_folder, exist_ok=True)

                for kk in tqdm(range(args.reinpaint_iter)):
                    residual_out = netG(residual_input, mask_01)
                    g_imgs = residual_input + residual_out
                    g_imgs = get_completion(pred=g_imgs, gt=real_image, mask_01=mask_01)
                    residual_input = g_imgs
                    completed_list.append(residual_input.detach())

                    iter_sub_folder = os.path.join(sub_outdir,f"iter_{kk+1}")
                    os.makedirs(iter_sub_folder,exist_ok = True)


                    if args.show_final == True:
                        abs_residual_out = torch.abs(residual_out)*mask_01
                        heat_residual_out = abs_residual_out[:,0,:,:] + abs_residual_out[:,1,:,:] + abs_residual_out[:,2,:,:]

                        Gray = abs_residual_out[:,0,:,:] * 0.299 + abs_residual_out[:,1,:,:] * 0.587 + abs_residual_out[:,2,:,:] * 0.114
                        Gray = Gray.repeat([1,3,1,1])

                        Red = Gray.clone()
                        Red[:,2,:,:] = 0
                        Red[:,1,:,:] = 0

                    mask_01_show = mask_01.repeat([1, 3, 1, 1])
                    torch.cuda.empty_cache()
                    for j, g_img in enumerate(g_imgs):
                        real_image_ = real_image[j].squeeze()
                        im_in_ = im_in[j].squeeze()
                        mask_01_ = mask_01_show[j].squeeze()

                        if args.show_final == True:
                            residual_out_ = abs_residual_out[j].squeeze()
                            Gray_ = Gray[j].squeeze()
                            Red_ = Red[j].squeeze()

                            heat_residual_out_ = heat_residual_out[j]
                            heat_residual_out_ = heat_residual_out_ / torch.max(heat_residual_out_)
                            heat_residual_out_ = np.uint8(255.0 * heat_residual_out_.cpu())
                            heat_residual_out_ = cv2.applyColorMap(heat_residual_out_, cv2.COLORMAP_JET)
                            heat_residual_out_ = np.clip(heat_residual_out_, 0, 255)
                            vutils.save_image(
                                residual_out_,
                                f"{str(iter_sub_folder)}/{str(i * args.batch + j).zfill(6)}_residual_out.png",
                                nrow=int(1),
                                normalize=True, range=(0, 1),
                            )

                            vutils.save_image(
                                g_img.add(1).mul(0.5),
                                f"{str(iter_sub_folder)}/{str(i * args.batch + j).zfill(6)}_inpaint.png",
                                nrow=int(1),
                                # normalize=True, range=(-1, 1),
                            )
                        elif j == len(g_imgs)-1:
                            vutils.save_image(
                                g_img.add(1).mul(0.5),
                                f"{str(iter_sub_folder)}/{str(i * args.batch + j).zfill(6)}_inpaint.png",
                                nrow=int(1),
                                # normalize=True, range=(-1, 1),
                            )

                        # vutils.save_image(
                        #     Gray_,
                        #     f"{str(iter_sub_folder)}/{str(i * args.batch + j).zfill(6)}_gray_residual_out.png",
                        #     nrow=int(1), normalize=True, range=(0, 1), )
                        # vutils.save_image(
                        #     heat_residual_out_,
                        #     f"{str(iter_sub_folder)}/{str(i * args.batch + j).zfill(6)}_heat_residual_out_.png",
                        #     nrow=int(1), normalize=True, range=(0, 1), )

                        # heat_residual_out_ = Image.fromarray(heat_residual_out_)
                        # heat_residual_out_.save(f"{str(iter_sub_folder)}/{str(i * args.batch + j).zfill(6)}_heat_residual_out_.png")

                        #
                        if args.show_final == True:
                            cv2.imwrite(f"{str(iter_sub_folder)}/{str(i * args.batch + j).zfill(6)}_heat_residual_out_.png", heat_residual_out_)

                        if kk == 0: # only store at first iter
                            # if jjj == 0: # only store gt images once!
                            vutils.save_image(
                                real_image_.add(1).mul(0.5),
                                f"{str(gt_folder)}/{str(i * args.batch + j).zfill(6)}_gt.png",
                                nrow=int(1),
                                # normalize=True, range=(-1, 1),
                            )
                                # if gt_folder = None
                                # gt_outdir = str(gt_folder)  #save the first gt image folder

                            vutils.save_image(
                                im_in_.add(1).mul(0.5),
                                f"{str(gt_folder)}/{str(i * args.batch + j).zfill(6)}_masked.png",
                                nrow=int(1),
                                # normalize=True, range=(-1, 1),
                            )

                            vutils.save_image(
                                mask_01_,
                                f"{str(gt_folder)}/{str(i * args.batch + j).zfill(6)}_mask.png",
                                nrow=int(1),
                                # normalize=True, range=(0, 1),
                            )


        torch.cuda.empty_cache()

        best_dic = {}
        best_metric_ = {}
        for kk in range(args.reinpaint_iter):
            print(f"the iter {kk+1}:============--------------============")
            iter_sub_folder = os.path.join(sub_outdir, f"iter_{kk + 1}")
            fid_value, U_IDS_score, P_IDS_score, mae, psnr, ssim,lpips_val = get_metrics_with_lpips(
                    gt_folder,iter_sub_folder, postfix1="_gt.png",
                postfix2="_inpaint.png", batch_size=1)
            dict_ = {"fid_value": fid_value, "U_IDS_score": U_IDS_score, "P_IDS_score": P_IDS_score,
                     "mae": mae, "psnr": psnr, "ssim": ssim,"lpips":lpips_val.item()}

            if kk ==0:
                gt_view_folder = os.path.join(args.view_dict, f"{mask_type}/gt_folder")
                os.makedirs(gt_view_folder, exist_ok=True)
                delete_dirs(gt_view_folder)
                print(f"{3*args.view_number}  start extract from " + gt_folder + " to " + gt_view_folder)
                MoveTotheSingalDir(gt_folder, gt_view_folder, 3*args.view_number, shuffle=False, function_=ut.copyfile2Dir)
                #
                best_dic["fid"] = fid_value
                best_dic["iter"] = kk
                best_metric_ = dict_

            if fid_value<best_dic["fid"]:  #compare to show the best fid
                best_dic["fid"] = fid_value
                best_dic["iter"] = kk
                best_metric_ = dict_

            if args.view_inter == True:  # save all the intermediate results
                print("saving the each iter")
                view_folder = os.path.join(args.view_dict, f"{mask_type}/iter_{kk + 1}")
                delete_dirs(view_folder)
                os.makedirs(view_folder, exist_ok=True)
                print(f"{args.view_number}  start extract from " + iter_sub_folder + " to " + view_folder)
                MoveTotheSingalDir(iter_sub_folder, view_folder, args.view_number, shuffle=False, function_=ut.copyfile2Dir)

            elif args.view_inter == False and kk == (args.reinpaint_iter - 1): #only shows the best iter
                iter_ = best_dic["iter"]
                print(f"only saving the best iter: {iter_}")
                best_sub_folder = os.path.join(sub_outdir, f"iter_{iter_ + 1}")
                view_folder = os.path.join(args.view_dict, f"{mask_type}/iter_{iter_ + 1}")
                delete_dirs(view_folder)
                os.makedirs(view_folder, exist_ok=True)
                print(f"{args.view_number}  start extract from " + best_sub_folder + " to " + view_folder)
                MoveTotheSingalDir(best_sub_folder, view_folder, args.view_number, shuffle=False, function_=ut.copyfile2Dir)
        metrics_saver.append(best_metric_,col_name=mask_type)

    metrics_saver.save()

    # _, out_print = subprocess.getstatusoutput(f'tar -zcf view.tar.gz  {args.view_dict}')
    p = subprocess.Popen(f'tar -zcf view.tar.gz  {args.view_dict}', shell=True)
    return_code = p.wait()
    _, out_print = subprocess.getstatusoutput(f'rm -rf ./view')

