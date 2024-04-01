#-*-coding:utf-8 -*-
#!/usr/bin/python
#test_copyfile.py

import os,shutil
import random
import util.utils_train as ut

def MoveTotheSingalDir(srcDir,dstDir,num,shuffle = True,function_ = None):
    '''
	srcDir = 源文件夹
	dstdir = 目标文件夹
    num = 抽取数量
    '''
    ut.mkdir(dstDir)
    file_list =[]
    ut.listdir(srcDir, file_list)

    if(shuffle == True):
        random.shuffle(file_list)
    for file_signal in file_list[0:num]:
        print("from ."+file_signal+" to "+dstDir)
        function_(file_signal,dstDir)

def MoveTothe2Stage_Dir(srcDir,dstDir,num,ratio=0,shuffle = True,function_ = None):
    '''
    对应的把里面一层的文件夹保留 两层目录
	srcDir = 源文件夹
	dstdir = 目标文件夹
    num = 抽取数量
    '''
    #get the class of the dir


    for file in os.listdir(srcDir):
        file_path = os.path.join(srcDir, file)
        #for each dir get the all img of the dir
        if os.path.isdir(file_path):

            file_list =[]
            ut.listdir(file_path, file_list)
            #get all img paths and shuffle
            file_list.sort()
            if(shuffle == True):
                random.shuffle(file_list)

            if ratio > 0:
                num = int (ratio*float(len(file_list)))
            #move the first num th img
            for file_signal in file_list[0:num]:
                # print("move ."+file_signal+" to "+dstDir)
                # temp_sp = file_signal.split("/")
                final_dir = os.path.join(dstDir,file);
                print (final_dir)
                # movefile(file_signal,final_dir)
                function_(file_signal,final_dir)


def extra_2_folder(type ="move"):
    """
    move or copy file for class dataset
    in the details, move or copy files but save the sub folder for one level
    保存两层目录  即 /A/B  ==> /C/B   依然保存了下属一层文件夹
    """
    # 根据比率或根据数量进行抽取
    # 最新的 文件转移方法，利用随机的方式，方法更好

    srcDir = "/root/workspace/Workspace/Data/VLD-45-B_class_30000/train"
    dstDir = "/root/workspace/Workspace/Data/VLD-45-B_class_30000/test"
    num = -1
    ratio= 0.5
    fun = None
    if type == "move":
        fun = ut.movefile2Dir
    elif type == "copy":
        fun = ut.copyfile2Dir


    print(str(num) + " start extract from " + srcDir + " to " + dstDir)
    # MoveTotheSingalDir(srcDir,dstDir,num,shuffle = True)
    # function_ 选择拷贝还是移动
    MoveTothe2Stage_Dir(srcDir, dstDir, num, ratio=ratio, shuffle=True, function_= fun)
    print("over .")




def extra_1_folder(type="copy"):
    """
    move or copy file directly
    保存一层目录   依然保存了下属一层文件夹
    """
    # 根据比率或根据数量进行抽取
    # 最新的 文件转移方法，利用随机的方式，方法更好

    srcDir = "/media/k/Longlongaaago4/Dataset/FFHQ_wm/FFHQ/FFHQ"
    dstDir = "/media/k/Longlongaaago4/Dataset/FFHQ_wm/FFHQ/test"
    num = 10000
    ####
    fun = None
    if type == "move":
        fun = ut.movefile2Dir
    elif type == "copy":
        fun = ut.copyfile2Dir

    print(str(num) + " start extract from " + srcDir + " to " + dstDir)
    # MoveTotheSingalDir(srcDir,dstDir,num,shuffle = True)
    # function_ 选择拷贝还是移动

    print(type+"!!!!")
    MoveTotheSingalDir(srcDir, dstDir, num, shuffle=True, function_=fun)
    print(type+"over .")


if __name__ == "__main__":
    extra_1_folder(type="move")
    # extra_2_folder(type="move")







