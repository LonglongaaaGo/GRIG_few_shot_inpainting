import torch
import os
from distributed import (
    get_rank,synchronize
)
from torchvision import utils
from fid_eval import test_matrix,lpip_matrix
from pytorch_fid import fid_score
from tqdm import tqdm
from util.operation import dic_2_str,get_mask,get_completion,delete_dirs


def calculate_all_PIDS(args,iter,eval_dict,logger,best_evel_batch=10,device="cuda"):
    fid_value = fid_score.calculate_fid_given_paths_postfix(path1=eval_dict, postfix1="_gt.png",
                                                          path2=eval_dict, postfix2="_inpaint.png",
                                                          batch_size=best_evel_batch, device=device,
                                                          dims=2048, num_workers=args.num_workers)
    print("fid_score_:%g" % fid_value)

    print("fid_score.calculate_P_IDS_U_IDS_given_paths_postfix_no_fid")
    U_IDS_score, P_IDS_score = fid_score.calculate_P_IDS_U_IDS_given_paths_postfix_no_fid(path1=eval_dict,
                                                                                              postfix1="_gt.png",
                                                                                              path2=eval_dict,
                                                                                              postfix2="_inpaint.png",
                                                                                              batch_size=best_evel_batch,
                                                                                              device=device,
                                                                                              dims=2048,
                                                                                              num_workers=args.num_workers)

    test_name = ['mae', 'psnr', 'ssim']
    out_dic = test_matrix(path1=eval_dict, postfix1="_gt.png"
                          , path2=eval_dict, postfix2="_inpaint.png", test_name=test_name)

    logger.update(iter=iter, mae=out_dic['mae'], psnr=out_dic['psnr'],
                 ssim=out_dic['ssim'], fid=fid_value,
                 U_IDS_score=U_IDS_score, P_IDS_score=P_IDS_score)

    print("mae:%g, psnr:%g, ssim:%g,fid:%g" % (out_dic['mae'], out_dic['psnr'], out_dic['ssim'], fid_value))

    print("fid_value:%g, U_IDS_score:%g, P_IDS_score:%g" % (fid_value, U_IDS_score, P_IDS_score))

    out_dics = {}
    out_dics['mae'] = out_dic['mae']
    out_dics['psnr'] = out_dic['psnr']
    out_dics['ssim'] = out_dic['ssim']
    out_dics['fid'] = fid_value
    out_dics['U_IDS_score'] = U_IDS_score
    out_dics['P_IDS_score'] = P_IDS_score
    out_dics['tmp_fid'] = fid_value

    return  out_dics



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag




def Reinference_v2(args,netG,test_loader,eval_dict):
    torch.cuda.empty_cache()
    print("inference on testing dataset!!!")

    if get_rank() == 0:
        delete_dirs(eval_dict)
        if not os.path.exists(eval_dict):
            os.makedirs(eval_dict,exist_ok=True)
    # synchronize()
        with torch.no_grad():
            netG.eval()
            count = 0
            for i, data in tqdm(enumerate(test_loader)):
                if args.debug == True and count > 10: break
                count += 1
                real_image = data
                real_image = real_image.cuda()

                gin, gt_local, mask, mask_01, im_in = get_mask(real_image, mask_type='center', im_size= real_image.shape[2],
                                                mask_shapes= [real_image.shape[2]//2,real_image.shape[2]//2])

                residual_input = im_in
                for kk in range(args.reinpaint_iter):
                    residual_out = netG(residual_input,mask_01)

                    g_imgs = residual_input+residual_out
                    g_imgs = get_completion(pred=g_imgs, gt=real_image, mask_01=mask_01)
                    residual_input = g_imgs

                for j, g_img in enumerate(g_imgs):
                    real_image_ = real_image[j].squeeze()
                    im_in_ = im_in[j].squeeze()

                    utils.save_image(
                        g_img.add(1).mul(0.5),
                        f"{str(eval_dict)}/{str(i * args.batch_size + j).zfill(6)}_{str(get_rank())}_inpaint.png",
                        nrow=int(1), normalize=True)
                    utils.save_image(
                        real_image_.add(1).mul(0.5),
                        f"{str(eval_dict)}/{str(i * args.batch_size + j).zfill(6)}_{str(get_rank())}_gt.png",
                        nrow=int(1), normalize=True )

                    utils.save_image(
                        im_in_.add(1).mul(0.5),
                        f"{str(eval_dict)}/{str(i * args.batch_size + j).zfill(6)}_{str(get_rank())}_mask.png",
                        nrow=int(1), normalize=True )
            netG.train()
    # synchronize()
    print("inference down!")



def get_metric_score(args, iter, eval_dict, logger, device="cuda"):
    out_dics = calculate_all_PIDS(args, iter, eval_dict, logger, device=device)
    outstr_ = dic_2_str(out_dics)
    print(outstr_)
    # delete_dirs(eval_dict)
    return out_dics



def get_metrics(gt_path,pre_path,postfix1=".jpg",postfix2=".png",batch_size=10,device="cuda",num_workers=8):
    test_name = ['mae', 'psnr', 'ssim']
    out_dic = test_matrix(path1=gt_path, postfix1=postfix1
                          ,path2=pre_path, postfix2=postfix2, test_name=test_name, batch_size=batch_size)
    print("mae:%g, psnr:%g, ssim:%g" % (out_dic['mae'], out_dic['psnr'], out_dic['ssim']))
    fid_value, U_IDS_score, P_IDS_score = fid_score.calculate_P_IDS_U_IDS_given_paths_postfix(path1=gt_path,
                                                                                              postfix1=postfix1,
                                                                                              path2=pre_path,
                                                                                              postfix2=postfix2,
                                                                                              batch_size=batch_size,
                                                                                              device=device,
                                                                                              dims=2048,
                                                                                              num_workers=num_workers)



    print("fid_value:%g, U_IDS_score:%g, P_IDS_score:%g" % (fid_value, U_IDS_score, P_IDS_score))

    return fid_value, U_IDS_score, P_IDS_score,out_dic['mae'], out_dic['psnr'], out_dic['ssim']



def get_metrics_with_lpips(gt_path,pre_path,postfix1=".jpg",postfix2=".png",batch_size=10,device="cuda",num_workers=8):
    test_name = ['mae', 'psnr', 'ssim']
    out_dic = test_matrix(path1=gt_path, postfix1=postfix1
                          ,path2=pre_path, postfix2=postfix2, test_name=test_name, batch_size=batch_size)
    print("mae:%g, psnr:%g, ssim:%g" % (out_dic['mae'], out_dic['psnr'], out_dic['ssim']))

    lpip_val = lpip_matrix(path1=gt_path, postfix1=postfix1
                          ,path2=pre_path, postfix2=postfix2,device=device)
    print("lpips: %g "% (lpip_val))

    fid_value, U_IDS_score, P_IDS_score = fid_score.calculate_P_IDS_U_IDS_given_paths_postfix(path1=gt_path,
                                                                                              postfix1=postfix1,
                                                                                              path2=pre_path,
                                                                                              postfix2=postfix2,
                                                                                              batch_size=batch_size,
                                                                                              device=device,
                                                                                              dims=2048,
                                                                                              num_workers=num_workers)



    print("fid_value:%g, U_IDS_score:%g, P_IDS_score:%g" % (fid_value, U_IDS_score, P_IDS_score))

    return fid_value, U_IDS_score, P_IDS_score,out_dic['mae'], out_dic['psnr'], out_dic['ssim'],lpip_val


if __name__ == '__main__':
    gt_path = "/home/k/Data/place2_256_256/val_256"
    pre_path = "/home/k/Workspace/Lama/bin/outputs/2022-06-30/09-05-34/eval"
    get_metrics(gt_path,pre_path,postfix1=".jpg",postfix2=".png",batch_size=10)


    # gt_path = "/home/k/tmp_dir"
    # pre_path = "/home/k/tmp_dir"
    # get_metrics(gt_path,pre_path,postfix1=".png",postfix2=".jpg",
