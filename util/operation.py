import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
from copy import deepcopy
import shutil
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
# from scipy.misc import imread


import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import matplotlib.pyplot as plt
import imageio


def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

def dic_2_str(dics):
    out_str = "\n"
    for key in dics.keys():
        out_str += str(key)+":"+str(dics[key])+ " "
    out_str+="\n"
    return out_str

def delete_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path,ignore_errors=True)
    return

def copy_dirs(patha,pathb):
    delete_dirs(pathb)
    #copytree 包含了mkdir
    shutil.copytree(patha,pathb)


def listdir(path, list_name):  # 传入存储的list
    '''
    递归得获取对应文件夹下的所有文件名的全路径
    存在list_name 中
    :param path: input the dir which want to get the file list
    :param list_name:  the file list got from path
	no return
    '''
    list_dirs = os.listdir(path)
    list_dirs.sort()
    for file in list_dirs:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    list_name.sort()
    # print(list_name)

def copy_Dir2Dir(srcDir, dstDir,num=100):
    '''
    copy dir files into target dir and save the original file name
    srcfile = 目标文件夹
    dstDir = 目标文件夹
    将目标文件拷贝至目标文件夹
    '''
    if not os.path.exists(srcDir):
        print("%s not exist!" % (srcDir))
    else:
        file_list = []
        listdir(srcDir, file_list)
        if not os.path.exists(dstDir):
            os.makedirs(dstDir)  # 创建路径
        for i, file in enumerate(file_list):
            if i>num: break
            fpath, fname = os.path.split(file)  # 分离文件名和路径
            shutil.copyfile(file, os.path.join(dstDir, fname))  # 拷贝文件
            print("copy %s -> %s" % (file, os.path.join(dstDir, fname)))



class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def copy_G_params(model,multi_gpu =False):
    if multi_gpu == True:
        flatten = deepcopy(list(p.data for p in model.module.parameters()))
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))

    return flatten
    

def load_params(model, new_param,multi_gpu =False):
    if multi_gpu == True:
        for p, new_p in zip(model.module.parameters(), new_param):
            p.data.copy_(new_p)
    else:
        for p, new_p in zip(model.parameters(), new_param):
            # print("p_shape:", p.shape)
            # print("new_p_shape:",new_p.shape)
            p.data.copy_(new_p)


def get_dir(args):
    task_name = 'train_results/' + args.name
    saved_model_folder = os.path.join( task_name, 'models')
    saved_image_folder = os.path.join( task_name, 'images')
    
    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    for f in os.listdir('./'):
        if '.py' in f:
            shutil.copy(f, task_name+'/'+f)
    
    # with open( os.path.join(saved_model_folder, '../args.txt'), 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder

from util.utils import generate_mask

def get_mask(real_image,mask_type, im_size,mask_shapes):
    current_batch_size = real_image.shape[0]
    mask, rect = generate_mask(mask_type, [im_size, im_size], mask_shapes)
    mask_01 = torch.from_numpy(mask).cuda().repeat([current_batch_size, 1, 1, 1])
    if mask_type == 'rect':
        rect = [rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3]]
        gt_local = real_image[:, :, rect[0]:rect[0] + rect[1],
                   rect[2]:rect[2] + rect[3]]
    else:
        gt_local = real_image
    im_in = real_image * (1 - mask_01)
    #[data,mask]
    gin = torch.cat((im_in, mask_01), 1)
    # real_image = torch.cat((real_image, mask_01), 1)
    return gin, gt_local,mask,mask_01,im_in


def get_completion(pred,gt,mask_01):
    gt = F.interpolate(gt,(pred.shape[2],pred.shape[3]))
    mask_01 = F.interpolate(mask_01,(pred.shape[2],pred.shape[3]))
    completion = pred * mask_01 + gt * (1 - mask_01)
    return completion




class  ImageFolder_test(torch.utils.data.Dataset):
    """docstring for ArtDataset"""
    # with center crop for test

    def __init__(self, root, transform=None,im_size=(256,256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( ImageFolder_test, self).__init__()
        self.root = root
        self.frame = self._parse_frame()
        self.transform = transform
        self.im_size = im_size
        self.h = self.im_size[0]
        self.w = self.im_size[1]
        self.mask = np.zeros((self.h, self.w)).astype(np.uint8)
        self.mask[self.h // 4:self.h // 4 * 3, self.w // 4:self.w // 4 * 3] = 1.0


    def _parse_frame(self):
        # img_names = os.listdir(self.root)
        img_names = []
        listdir(self.root,img_names)
        img_names.sort()
        frame = []
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def resize_center_crop(self,w,h):

        return img
    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w,h = img.size

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang > 0: h_idx = h_rang // 2
            if w_rang > 0: w_idx = w_rang // 2
            img = img_scaled.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))


        if self.transform:
            img = self.transform(img)

        mask = self.mask.copy()
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img,mask



class ImageFolder_with_mask(torch.utils.data.Dataset):
    # with center crop for test
    """docstring for ArtDataset"""
    def __init__(self, root, mask_root,mask_file,transform=None,train_tag = True,im_size=(256,256)):
        """
        :param root: root for the images
        :param mask_root: root for the masks
        :param transform:
        :param im_size:  (h,w)
        """
        super(ImageFolder_with_mask, self).__init__()
        self.root = root
        self.mask_root = mask_root
        self.mask_file = mask_file
        self._get_mask_list()
        self.frame = self._parse_frame()

        self.transform = transform
        self.im_size = im_size
        self.train_tag = train_tag

    def _get_mask_list(self):
        mask_list = []

        file = open(self.mask_file)
        lines = file.readlines()
        for line in lines:
            mask_path = os.path.join(self.mask_root,line.strip())
            mask_list.append(mask_path)
        file.close()
        mask_list.sort()
        self.mask_list = mask_list

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w,h = img.size
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.Resampling.LANCZOS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = h_rang//2
            if w_rang > 0: w_idx = w_rang//2
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))

        # mask
        if not self.train_tag:
            mask_idx = idx% len(self.mask_list)
        else:
            mask_idx = np.random.randint(0,len(self.mask_list))
        mask_path = self.mask_list[mask_idx]
        mask_img = Image.open(mask_path).convert('P')

        mask_img = mask_img.resize(self.im_size,Image.NEAREST)

        mask_img = np.array(mask_img)
        if mask_img.ndim == 2:
            mask = np.expand_dims(mask_img, axis=0)
        else:
            mask = mask_img[0:1, :, :]
        mask[mask > 0] = 1.0


        if self.transform:
            img = self.transform(img)

        return img,mask




class  Few_shot_ImageFolder(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None,im_size=(256,256),shot=-1):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        :param shot:  how many samples will be used for training?   shot=-1 : all.  shot=5: 5 imgs will be used for training
        """
        super(Few_shot_ImageFolder, self).__init__()
        self.root = root

        self.frame = self._parse_frame()

        shot = min(shot, len(self.frame))
        self.shot = shot
        if shot != -1:
            self.frame = self.frame[:shot]

        self.transform = transform
        self.im_size = im_size
    def _parse_frame(self):
        # img_names = os.listdir(self.root)
        img_names = []
        listdir(self.root,img_names)
        img_names.sort()
        frame = []
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w,h = img.size
        # plt.imshow(img)
        # plt.show()

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.Resampling.LANCZOS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        if self.transform:
            img = self.transform(img)

        return img




class eye_blink_ImageFolder(Dataset):
    # eye blink exemple
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None,im_size=(256,256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( eye_blink_ImageFolder, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform
        self.im_size = im_size

        # folder_list = []
        # for folder in os.listdir(root):
        #     img_names = []
        #     listdir(self.root, img_names)
        #     img_names.sort()
        #     folder_list.append(img_names)

    def _parse_frame(self):
        # img_names = os.listdir(self.root)
        img_names = []
        listdir(self.root,img_names)
        img_names.sort()
        frame = []
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)
        # return len(self.folder_list)


    def __getitem__(self, idx):

        # img_list = self.folder_list[idx]
        # feame_idx = random.randint(0,len(img_list)-1-k)
        # frames = img_list[feame_idx:feame_idx+k]

        # frame_tensor = []
        # label_tensor = torch.zeros((len(frames)))
        # for i in range(len(frames)):
        #     name = os.path.basename(frames[i])
        #     tmp_label = int(str(name).strip().split("_")[-2])
        #
        #     img = Image.open(frames[i]).convert('RGB')
        #     if self.transform:
        #         img = self.transform(img)
        #     frame_tensor.append(img)
        #     label_tensor[i] = tmp_label
        # frame_tensor = torch.cat(frame_tensor,dim=0)

        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w,h = img.size
        # plt.imshow(img)
        # plt.show()

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.Resampling.LANCZOS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        if self.transform:
            img = self.transform(img) 

        return img


class ImageFolder(Dataset):
    """docstring for ArtDataset"""

    def __init__(self, root, transform=None, im_size=(256, 256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super(ImageFolder, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform
        self.im_size = im_size


    def _parse_frame(self):
        img_names = []
        listdir(self.root, img_names)
        img_names.sort()
        frame = []
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[
                                                                                                      -5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w, h = img.size
        # plt.imshow(img)
        # plt.show()

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang > 0: h_idx = random.randint(0, h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        if self.transform:
            img = self.transform(img)

        return img


class  ImageFolder_crop_only(Dataset):
    """docstring for ArtDataset
        only having the corp operation
    """
    def __init__(self, root, transform=None,im_size=(256,256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( ImageFolder_crop_only, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform
        self.im_size = im_size
    def _parse_frame(self):
        # img_names = os.listdir(self.root)
        img_names = []
        listdir(self.root,img_names)
        img_names.sort()
        frame = []
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w,h = img.size
        # plt.imshow(img)
        # plt.show()

        # if h != self.im_size[0] or w != self.im_size[1]:
        #     ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
        #     new_w = int(ratio * w)
        #     new_h = int(ratio * h)
        #     img_scaled = img.resize((new_w,new_h),Image.Resampling.LANCZOS)
        h_rang = h - self.im_size[0]
        w_rang = w - self.im_size[1]
        h_idx = 0
        w_idx = 0
        if h_rang>0: h_idx = random.randint(0,h_rang)
        if w_rang > 0: w_idx = random.randint(0, w_rang)
        img = img.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        if self.transform:
            img = self.transform(img)

        return img


class  ImageFolder_CenterCrop(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None,im_size=(256,256),shot=-1):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( ImageFolder_CenterCrop, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform
        self.im_size = im_size

        if shot != -1:
            self.shot = shot
            shot = min(shot, len(self.frame))
            self.frame = self.frame[:shot]

    def _parse_frame(self):
        # img_names = os.listdir(self.root)
        img_names = []
        listdir(self.root,img_names)
        img_names.sort()
        frame = []
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w,h = img.size
        # plt.imshow(img)
        # plt.show()

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.Resampling.LANCZOS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = h_rang//2
            if w_rang > 0: w_idx = w_rang//2
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        if self.transform:
            img = self.transform(img)

        return img



class ImageFolderWithEdge(Dataset):
    """docstring for ArtDataset"""

    def __init__(self, root, transform=None, im_size=(256, 256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super(ImageFolderWithEdge, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform
        self.im_size = im_size

    def _parse_frame(self):
        # img_names = os.listdir(self.root)
        img_names = []
        listdir(self.root, img_names)
        img_names.sort()
        frame = []
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[
                                                                                                      -5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def load_edge(self, img, mask=None):
        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        # mask = None if self.training else (1 - mask / 255).astype(np.bool)
        if mask is not None:
            mask =  (1 - mask / 255).astype(np.bool)

        sigma = random.randint(1, 4)

        return canny(img, sigma=sigma, mask=mask).astype(np.float)


    def __getitem__(self, idx):
        file = self.frame[idx]
        # pil_img = Image.open(file).convert('RGB')
        # img_gray = Image.open(file).convert('L')

        img =imageio.imread(file)
        img_gray = rgb2gray(img)

        edge = self.load_edge(img_gray, mask=None)

        # augment data
        if random.randint(0,1) == 1:
            # img = img[:, ::-1, ...].copy()
            # edge = edge[:, ::-1, ...].copy()
            img = np.flip(img, axis=1).copy()
            edge = np.flip(edge, axis=1).copy()


        if edge.ndim == 2:
            edge = np.expand_dims(edge, axis=0)
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img,edge

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif','gif']

def make_multi_dataset(dir, class_to_idx, extensions):
    """
    :param dir: data root
    :param class_to_idx: {xx:0,aaaa:b}  mapping table
    :param extensions: img_extensons
    :return:  a list of the dataset list for each dataset [cls_0: [aaa.jpg, ...],cls_1: [aaa.jpg, ...]]
    """
    datasets = [len(class_to_idx.keys())*[]]

    #把 ~/ 目录 转换成 /home/xxx/xx/ 的目录
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        #遍历这个类别下的所有数据
        for root, dirs, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                #如果满足后缀名,就加入对应的类别
                if has_file_allowed_extension(fname, extensions):

                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    datasets[class_to_idx[target]].append(item)

    return datasets

def find_classes(dir):
    classes = [d for d in os.listdir(dir)]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class  Multi_domain_ImageFolder(Dataset):
    """
    最终这个类被废弃了，因为这种方式不能保证一个batch里使用相同的类别
    multidomain_ dataset

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    """
    def __init__(self, root, transform=None,im_size=(256,256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( Multi_domain_ImageFolder, self).__init__()
        self.root = root
        classes, class_to_idx = find_classes(root)
        self.data_tables = make_multi_dataset(root,class_to_idx,extensions=IMG_EXTENSIONS)
        self.domain_len = len(self.data_tables)
        self.transform = transform
        self.im_size = im_size

        self.idx_list = range(0,self.domain_len -1)
        self.len_list = [len(self.data_tables[i]) for i in range(self.domain_len)]

    def __len__(self):
        """
        :return:求出所有数据集中数量最大的那个数
        """
        max_num = -1
        for i in range(len(self.data_tables)):
            if len(self.data_tables[i])>max_num: max_num = len(self.data_tables[i])

        return max_num

    def re_size_img(self,img):
        w,h = img.size
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang > 0: h_idx = random.randint(0, h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))

        return img
    def __getitem__(self, idx):

        idx_list = random.sample(self.idx_list, 2)
        #先随机获得两个不相关的域
        idx_a = idx_list[0]
        idx_b = idx_list[1]

        #数据数量不一样，所以用取余的方式来取样本
        file_a,target_a= self.data_tables[idx_a][idx % self.len_list[idx_a]]
        file_b,target_b = self.data_tables[idx_b][idx % self.len_list[idx_b]]

        img_a = Image.open(file_a).convert('RGB')
        img_b = Image.open(file_b).convert('RGB')

        img_a = self.re_size_img(img_a)
        img_b = self.re_size_img(img_b)

        # img.show()
        # plt.imshow(img)
        # plt.show()

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)


        return img_a,img_b


class ImageFolder_with_maskpair(Dataset):
    """docstring for ArtDataset"""

    def __init__(self, root, mask_root, transform=None, im_size=(256, 256)):
        """
        :param root: root for the images
        :param mask_root: root for the masks
        :param transform:
        :param im_size:  (h,w)
        """
        super(ImageFolder_with_maskpair, self).__init__()
        self.root = root
        self.mask_root = mask_root
        self.mask_list = self._get_mask_list()
        self.frame = self._parse_frame()

        self.transform = transform
        self.im_size = im_size

    def _get_mask_list(self):
        mask_list = []
        listdir(self.mask_root, mask_list)
        mask_list.sort()
        return mask_list

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[
                                                                                                      -5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w, h = img.size

        # mask
        rand_ = random.randint(0,1)
        mask_path = self.mask_list[idx*2+rand_]
        mask_img = Image.open(mask_path).convert('L')

        # resize and corp img and mask
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            mask_img = mask_img.resize((new_w, new_h), Image.NEAREST)

            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang > 0: h_idx = random.randint(0, h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))
            mask_img = mask_img.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))

        # mask dim and value
        mask_img = np.array(mask_img)
        if mask_img.ndim == 2:
            mask = np.expand_dims(mask_img, axis=0)
        else:
            mask_img = np.transpose(mask_img, (2, 0, 1))
            mask = mask_img[0:1, :, :]

        mask[mask <= 20] = 0
        mask[mask > 20] = 1.0

        if self.transform:
            img = self.transform(img)

        return img, mask


class ImageFolder_with_maskpair_test(Dataset):
    #center crop
    """docstring for ArtDataset"""

    def __init__(self, root, mask_root, transform=None, im_size=(256, 256)):
        """
        :param root: root for the images
        :param mask_root: root for the masks
        :param transform:
        :param im_size:  (h,w)
        """
        super(ImageFolder_with_maskpair_test, self).__init__()
        self.root = root
        self.mask_root = mask_root
        self.mask_list = self._get_mask_list()
        self.frame = self._parse_frame()

        self.transform = transform
        self.im_size = im_size

    def _get_mask_list(self):
        mask_list = []
        listdir(self.mask_root, mask_list)
        mask_list.sort()
        return mask_list

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[
                                                                                                      -5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w, h = img.size

        # mask
        rand_ = random.randint(0, 1)
        mask_path = self.mask_list[idx * 2 + rand_]
        mask_img = Image.open(mask_path).convert('L')

        # resize and corp img and mask
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            mask_img = mask_img.resize((new_w, new_h), Image.NEAREST)

            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang > 0: h_idx = h_rang // 2
            if w_rang > 0: w_idx = w_rang // 2
            img = img_scaled.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))
            mask_img = mask_img.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))

        # mask dim and value
        mask_img = np.array(mask_img)
        if mask_img.ndim == 2:
            mask = np.expand_dims(mask_img, axis=0)
        else:
            mask_img = np.transpose(mask_img, (2, 0, 1))
            mask = mask_img[0:1, :, :]

        mask[mask <= 20] = 0
        mask[mask > 20] = 1.0

        if self.transform:
            img = self.transform(img)

        return img, mask


class Multi_domainDatasets():
    def __init__(self,root,batch_size=1,dataloader_workers=4,im_size=256,type="train"):

        self.im_size = im_size
        self.type = type
        if type == "train":
            transform_list = [
                transforms.Resize((int(im_size), int(im_size))),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        elif type == "test":
            transform_list = [
                transforms.Resize((int(im_size), int(im_size))),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]

        trans = transforms.Compose(transform_list)

        #得到所有类别下的 类别路径
        self.dir_list = self.get_dirs(root=root)

        self.data_name_table = [ str(self.dir_list[i]).strip().split("/")[-1] for i in range(len(self.dir_list))]
        self.domain_len = len(self.dir_list)
        self.idx_list = range(0,self.domain_len - 1)

        self.dataset_len = []
        self.dataset_tables = []
        for i in range(self.domain_len):
            dataset = ImageFolder(root=self.dir_list[i], transform=trans)

            if self.type == "train":
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers,drop_last=True,
                                             pin_memory=False)
            elif self.type == "test":
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=dataloader_workers,drop_last=True,
                                        pin_memory=False)
            self.dataset_len.append(len(dataset))
            self.dataset_tables.append(iter(dataloader))

    def get_domian_num(self):
        return self.domain_len

    def get_dataset_len(self,idx):
        return self.dataset_len[idx]

    def get_dirs(self,root):
        dir_list = []
        for target in sorted(os.listdir(root)):
            d = os.path.join(root, target)
            if not os.path.isdir(d):
                continue
            dir_list.append(d)
        return dir_list

    def get_all_dataloaders_name(self,idx):
        return self.data_name_table[idx]

    def get_all_dataloaders_root(self,idx):
        return self.dir_list[idx]

    def get_all_dataloaders(self,idx):
        return self.dataset_tables[idx]

    def get_datalodars(self):
        if self.domain_len == 2:
            idx_a = 0
            idx_b = 1
        else:
            idx_list = random.sample(self.idx_list, 2)
            # 先随机获得两个不相关的域
            idx_a = idx_list[0]
            idx_b = idx_list[1]

        return (self.dataset_tables[idx_a],idx_a),(self.dataset_tables[idx_b],idx_b)


from io import BytesIO
import lmdb
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            #key_asp = f'aspect_ratio-{str(index).zfill(5)}'.encode('utf-8')
            #aspect_ratio = float(txn.get(key_asp).decode())

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class MultiResolutionDatasetWithEdge(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def load_edge(self, img, mask=None):
        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        # mask = None if self.training else (1 - mask / 255).astype(np.bool)
        if mask is not None:
            mask =  (1 - mask / 255).astype(np.bool)

        sigma = random.randint(1, 4)

        return canny(img, sigma=sigma, mask=mask).astype(np.float)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            #key_asp = f'aspect_ratio-{str(index).zfill(5)}'.encode('utf-8')
            #aspect_ratio = float(txn.get(key_asp).decode())

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        # edge = self.load_edge(img, mask=None)
        # # augment data
        # if self.augment and np.random.binomial(1, 0.5) > 0:
        #     img = img[:, ::-1, ...]
        #     edge = edge[:, ::-1, ...]
        #
        # img = self.transform(img)

        # img = imageio.imread(file)
        img = np.array(img)
        img_gray = rgb2gray(img)
        edge = self.load_edge(img_gray, mask=None)

        # augment data
        if random.randint(0, 1) == 1:
            # img = img[:, ::-1, ...].copy()
            # edge = edge[:, ::-1, ...].copy()
            img = np.flip(img, axis=1).copy()
            edge = np.flip(edge, axis=1).copy()

        if edge.ndim == 2:
            edge = np.expand_dims(edge, axis=0)
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img,edge

