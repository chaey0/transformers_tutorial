import glob
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def print_data_count(label_list):
    count = []
    for i in range(5):
        count.append(label_list.count(i))
    count.append(len(label_list))
    return count

class DatasetSerial(data.Dataset):
    """get image by index
    """

    def __init__(self, pair_list, img_transform=None, target_transform=None, two_crop=False):
        self.pair_list = pair_list

        self.img_transform = img_transform
        self.target_transform = target_transform
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.pair_list[index]
        image = pil_loader(path)

        # # image
        if self.img_transform is not None:
            img = self.img_transform(image)
        else:
            img = image

        return img, target

    def __len__(self):
        return len(self.pair_list)

def prepare_colon_tma_data():
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        cancer_test = False
        if cancer_test:
            file_list_bn = glob.glob(pathname.replace('*.jpg', '*0.jpg'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0])-1 for file_path in file_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    data_root_dir = './KBSMC_colon_tma_cancer_grading_1024/'

    set_1010711 = load_data_info('%s/1010711/*.jpg' % data_root_dir)
    set_1010712 = load_data_info('%s/1010712/*.jpg' % data_root_dir)
    set_1010713 = load_data_info('%s/1010713/*.jpg' % data_root_dir)
    set_1010714 = load_data_info('%s/1010714/*.jpg' % data_root_dir)
    set_1010715 = load_data_info('%s/1010715/*.jpg' % data_root_dir)
    set_1010716 = load_data_info('%s/1010716/*.jpg' % data_root_dir)
    wsi_00016 = load_data_info('%s/wsi_00016/*.jpg' % data_root_dir, parse_label=True,
                               label_value=0)  # benign exclusively
    wsi_00017 = load_data_info('%s/wsi_00017/*.jpg' % data_root_dir, parse_label=True,
                               label_value=0)  # benign exclusively
    wsi_00018 = load_data_info('%s/wsi_00018/*.jpg' % data_root_dir, parse_label=True,
                               label_value=0)  # benign exclusively

    train_set = set_1010711 + set_1010712 + set_1010713 + set_1010715 + wsi_00016
    valid_set = set_1010716 + wsi_00018
    test_set = set_1010714 + wsi_00017

    # print dataset detail
    train_label = [train_set[i][1] for i in range(len(train_set))]
    val_label = [valid_set[i][1] for i in range(len(valid_set))]
    test_label = [test_set[i][1] for i in range(len(test_set))]

    print(print_data_count(train_label))
    print(print_data_count(val_label))
    print(print_data_count(test_label))

    return train_set, valid_set, test_set


def visualize(ds, batch_size, nr_steps=5):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, batch_size + 1):
            sample = ds[data_idx + j]

            if isinstance(sample[0], str):
                img = pil_loader(sample[0])
                img = np.array(img)
            else:
                img = sample[0]

            if len(sample) > 2:
                aux = np.squeeze(sample[-1])
                aux = cmap(aux)[..., :3]
                aux = (aux * 255).astype('unint8')
                img = np.concatenate([img, aux], axis=0)
                img = cv2.resize(img, (40, 80), interpolation=cv2.INTER_CUBIC)

            plt.subplot(1, batch_size, j)
            plt.title(str(sample[1]))
            plt.imshow(img)

        plt.show()
        data_idx += batch_size


if __name__ == '__main__':
    train_set, valid_set, test_set = prepare_colon_tma_data()
    visualize(train_set, batch_size=4)



