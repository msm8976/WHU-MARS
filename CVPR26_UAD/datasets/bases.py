from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        
        pids, cams, tracks = [], [], []
        num_imgs = 0

        for k in data.keys():
            num_imgs+=len(data[k])
            for _, pid, camid, trackid in data[k]:
                pids += [pid]
                cams += [camid]
                tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

class ImageDataset(Dataset):
    def __init__(self, dataset, modalities, transform=None):
        # dataset: dict(modality -> list of (img_path, pid, camid, modality_id))
        self.dataset = dataset
        self.modalities = list(modalities)
        self.transform = transform

    def __len__(self):
        return sum(len(self.dataset[m]) for m in self.modalities)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            imgs = []
            camids = []
            pid = None
            for m_idx, modality in enumerate(self.modalities):
                img_path, cur_pid, camid, _ = self.dataset[modality][index[m_idx]]
                if pid is None:
                    pid = cur_pid
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)
                camids.append(camid)
            return imgs, pid, camids

class ImageDatasetTest(Dataset):
    def __init__(self, dataset, modality, transform=None):
        # dataset: list of (img_path, pid, camid, modality)
        self.dataset = dataset
        self.modality = modality
        self.transform = transform

    def __len__(self):
        return len(self.dataset[self.modality])
    
    def __getitem__(self, index):
        if isinstance(index, int):
            img_path, pid, camid, modality = self.dataset[self.modality][index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            filename = osp.basename(img_path)
            return img, pid, camid, modality, filename
