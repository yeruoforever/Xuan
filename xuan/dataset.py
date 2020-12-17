from typing import *
import warnings
import os.path as path
import itertools


class SegmentationSet(object):
    def __init__(self,data_dir:str,description=""):
        self.description=description
        self.data_dir=data_dir
        self.ids=self._get_all_indentitys()
        self.pairs=[]
        for i in self.ids:
            img=self._id_to_img(i)
            img=path.join(self.data_dir,img)
            if path.isfile(img):
                seg=self._id_to_seg(i)
                seg=path.join(self.data_dir,seg)
                seg=seg if path.isfile(seg) else None
                self.pairs.append((img,seg))
            
        if len(self.pairs)!=len(self.ids):
            warnings.warn("样本标识个数和数据集中病例样本不匹配，请检查数据集目录及文件完整。")

    def _get_all_indentitys(self):
        '用于实现获取数据集中各病例唯一标识'
        raise NotImplementedError()

    def _id_to_img(self,id):
        '用于实现根据病例唯一标识获取图像数据路径,对于多模态数据,返回一个路径元组'
        raise NotImplementedError()

    def _id_to_seg(self,id):
        '用于实现根据病例唯一标识获取分割标注数据路径'
        raise NotImplementedError()


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self,index:int):
        return self.pairs[index]

    def num_images(self):
        '''数据集中病例个数,包括不含分割标注的病例(通常仅用于训练)'''
        return self.__len__()

    def num_trainable(self):
        '''数据集中可用于训练的病例个数'''
        return sum(map(lambda x:1 if x[1] else 0,self.pairs))

    def num_segmentations(self):
        '''数据集中可用的分割标注个数'''
        return self.num_trainable()