import os
import jittor as jt
import numpy as np
import jittor.nn as nn
from jittor import transform
from jittor.dataset import Dataset
from PIL import Image
import json
from model_dense import Net1
from model import Net2, Net3, Net4
jt.flags.use_cuda=1

class bragging(object):
    def __init__(self, model_list, path_list, loader,  model_number = 1, class_num = 130):
        self.model_list = model_list
        self.path_list = path_list
        self.loader = loader
        self.model_number = model_number
        self.class_num = class_num
        self.model_weight = [1.01, 1.03, 1.0, 1.02] 
        self.class_dic = [0.0] * self.class_num

    def brag_eval(self):
        test_dic = {}
        result_path = './result2.json'
        for images, image_name  in self.loader:
            brag_res = self.vote(images)
            self.class_dic = [0] * self.class_num
            #print(brag_res)
            output = self.model_list[2](images)
            y = np.argpartition(output.data, -5, axis=1)
            thelist =[]
            for i in range(5):
                temp = str(y[0][999-i]+1)
                temp = eval(temp)
                thelist.append(temp)
            #print('original list: ', thelist) 
            thelist[0] = brag_res + 1
            while thelist[1] == thelist[0]:
                if thelist[1] == 130:
                    thelist[1] = 1
                else: 
                    thelist[1] += 1
            while thelist[2] == thelist[0] or thelist[2] == thelist[1]:
                if thelist[2] == 130:
                    thelist[2] = 1
                else: 
                    thelist[2] += 1
            while thelist[3] == thelist[0] or thelist[3] == thelist[1] or thelist[3] == thelist[2]:
                if thelist[3] == 130:
                    thelist[3] = 1
                else: 
                    thelist[3] += 1
            while thelist[4] == thelist[0] or thelist[4] == thelist[1] or thelist[4] == thelist[2] or thelist[4] == thelist[3]:
                if thelist[4] == 130:
                    thelist[4] = 1
                else:
                    thelist[4] += 1

            #print(thelist)
            test_dic[image_name[0]] = thelist
            with open(result_path, 'w') as f:
                json.dump(test_dic, f)

    def vote(self, image):
        for i in range(self.model_number):
            self.model_list[i].load(self.path_list[i])
            self.model_list[i].eval()
            output = self.model_list[i](image)
            pred = np.argmax(output.data, axis=1)
            #print(pred[0])
            self.class_dic[pred[0]] += self.model_weight[i]
        final_pred = self.class_dic.index(max(self.class_dic))
        return final_pred

class DogLoader(Dataset):
    def __init__(self, root_dir, batch_size, shuffle=False, transform=None, num_workers=1):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_classes = 130
        self.shuffle = shuffle
        self.image_list = []
        self.id_list = []
        self.transform = transform

        self.root_dir = os.path.join(self.root_dir, 'TEST_A/')
        self.root_dir_list = os.listdir(self.root_dir)
        for i in self.root_dir_list:
            self.image_list.append(i)

        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.image_list),
            shuffle=self.shuffle
        )

    def __getitem__(self, idx):
        print(idx)
        image_name = self.image_list[idx]
        print("image_name: ", image_name)
        image_path = self.root_dir + image_name
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        img = np.asarray(image)
        return img, image_name

def main():
    model_list = []
    model_list.append(Net1(130))
    model_list.append(Net2(130))
    model_list.append(Net3(130))
    model_list.append(Net4(130))

    path_list = []
    path_list.append('/home/rao/jittor/TsinghuaDoge/best_model_densenet.pkl')
    path_list.append('/home/rao/jittor/TsinghuaDoge/best_model_googlenet.pkl')
    path_list.append('/home/rao/jittor/TsinghuaDoge/best_model_resnet.pkl')
    path_list.append('/home/rao/jittor/TsinghuaDoge/best_model_vgg.pkl')

    transform_test = transform.Compose([
        transform.Resize((512, 512)),
        transform.CenterCrop(448),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    root_dir = '/home/rao/jittor/data/TsinghuaDogs'
    test_loader = DogLoader(root_dir, batch_size=1, shuffle=False, transform=transform_test)
    brag_instance = bragging(model_list, path_list, test_loader, 4, 130)
    brag_instance.brag_eval()

if __name__ == '__main__':
    main()
