import os
import numpy as np 
import pandas as pd 
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from utils import get_augmentations
import pytorch_lightning as pl

def get_dataloader(
    path_to_csv: list,
    phase: str,
    batch_size: int = 1,
    num_workers: int = 0,
):
    '''Returns: dataloader for the model training'''
    for path in path_to_csv:
        if 'val' in path:
            val_df = pd.read_csv(path)
        else:
            df = pd.read_csv(path)

    dataset = BratsDataset(df, phase)    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )
    if phase == 'train':
        val_dataset = BratsDataset(val_df, phase)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,   
        )
        return [dataloader, val_dataloader]
    else:
        return [dataloader]
        
class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str="test", is_resize: bool=True):
        super().__init__()
        self.df = df
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.is_resize = is_resize
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        # load all modalities
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)#.transpose(2, 0, 1)
            
            if self.is_resize:
                img = self.resize(img)
            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        # if self.phase != "test":
        mask_path =  os.path.join(root_path, id_ + "_seg.nii")
        mask = self.load_img(mask_path)
        
        if self.is_resize:
            mask = self.resize(mask)
            mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
            mask = np.clip(mask, 0, 1)
        mask = self.preprocess_mask_labels(mask)
        augmented = self.augmentations(image=img.astype(np.float32), 
                                        mask=mask.astype(np.float32))
        
        img = augmented['image']
        mask = augmented['mask']
        return {
            "Id": id_,
            "image": img,
            "mask": mask,
        }
        
        # return {
        #     "Id": id_,
        #     "image": img,
        # }
    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data
    
    def normalize(self, data: np.ndarray):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def resize(self, data: np.ndarray):
        return resize(data, (128, 128, 128), preserve_range=True)
    
    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask


class BratsDataModule(pl.LightningDataModule):

    def __init__(self, train_df, test_df, is_resize=True, batch_size=1, num_workers=4):
        super(BratsDataModule, self).__init__() 
        self.train_df = train_df 
        self.test_df = test_df 
        self.is_resize = is_resize 
        self.batch_size = batch_size 
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = BratsDataset(self.train_df, 'train') 
        self.test_dataset = BratsDataset(self.test_df, 'test')  

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,   
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,   
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,   
        )

if __name__ == '__main__':
    train_df = pd.read_csv('/u/akommala/brain-tumor-segmentation/notebooks/train_data.csv')
    test_df = pd.read_csv('/u/akommala/brain-tumor-segmentation/notebooks/test_data.csv')

    dm = BratsDataModule(train_df, test_df)
    dm.prepare_data()
    dm.setup()

    for batch in dm.train_dataloader():
        break 

    batch_id, images, masks = batch['Id'], batch['image'], batch['mask']
    print('ID: ', batch_id)
    print('image batch shape: ', images.shape)
    print('masks batch shape: ', masks.shape)