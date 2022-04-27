
import pandas as pd
import monai
import torch
import numpy as np
from monai.transforms import Compose,NormalizeIntensityd,ToDeviced,LoadImaged,AddChanneld,ScaleIntensityd,Resized,ConcatItemsd,RandRotate90d,RandRotated,RandZoomd,RandFlipd,RandFlipd,  ToTensord,  Compose
import config
from monai.data import (
    CacheDataset,
)

def dataloader(
        annotated_dataset,
        dataset_folder,
        scans,
        features,
        batch_size,
        ):
    
        df = pd.read_excel(annotated_dataset)
        df=df[['#','FNH','HCC','MET','Other','Dataset',
                'Early enhancement',
                'Washout', 
                'Delayed phase enhancement', 
                'Peripheral enhancement',
                'Central scar',
                'Capsule',
                'T2 hyperintensity', 
                'Iso- or hyperintensity on venous phase',
                'Hypoenhancing core', 
                'Hemorrhage/siderosis',
                'Diameter'
                ]]

        df=df.dropna()
        df = df.astype({'#': str,'FNH': int,'HCC': int,'MET': int,'Other': int,
                        'Early enhancement': int,
                        'Washout': int ,
                        'Delayed phase enhancement':int,
                        'Dataset':int,
                        'Peripheral enhancement':int,
                        'Iso- or hyperintensity on venous phase':int,
                        'Central scar':int,
                        'Capsule':int,
                        'Hypoenhancing core':int,
                        'T2 hyperintensity':int, 
                        'Hemorrhage/siderosis':int,
                        'Diameter':str,
                        })

        for index, row in df.iterrows():   
            if row['FNH']==0 and row['HCC']==0 and row['MET']==0 and row['Other']==0:
                df.drop(index, inplace=True)
        
        df.isnull().sum()
        df.head()
        df.describe()
        pd.set_option('display.max_columns', None)


        val_dataframe = df[(df['Dataset']==1)]
        train_dataframe = df[(df['Dataset']==0)]


        print(
            "Using %d samples for training and %d for validation"
            % (len(train_dataframe), len(val_dataframe))
        )

        mypath=dataset_folder
        
        trainY = []
        valY = []
        
        df_tr=train_dataframe[features]
        for index, row in df_tr.iterrows():     
            trainY.append(row.values)
            
        df_val=val_dataframe[features]
        for index, row in df_val.iterrows():     
            valY.append(row.values)

        if 'HBP' in scans:
            train_df_HBP = mypath +  train_dataframe['#'].astype(str) + '_HBP' + '.nii.gz'
        if 'T2W' in scans:
            train_df_T2W = mypath +  train_dataframe['#'].astype(str) + '_T2W' + '.nii.gz'
        if 'ART' in scans:
            train_df_ART = mypath +  train_dataframe['#'].astype(str) + '_ART' + '.nii.gz'
        if 'NAT' in scans:
            train_df_NAT = mypath +  train_dataframe['#'].astype(str) + '_NAT' + '.nii.gz'
        if 'PVP' in scans:
            train_df_PVP = mypath +  train_dataframe['#'].astype(str) + '_PVP' + '.nii.gz'
        if 'VEN' in scans:
            train_df_VEN = mypath +  train_dataframe['#'].astype(str) + '_VEN' + '.nii.gz'


        if 'HBP' in scans:
            val_df_HBP = mypath + val_dataframe['#'].astype(str) + '_HBP' + '.nii.gz'
        if 'T2W' in scans:
            val_df_T2W = mypath + val_dataframe['#'].astype(str) + '_T2W' + '.nii.gz'
        if 'ART' in scans:
            val_df_ART = mypath + val_dataframe['#'].astype(str) + '_ART' + '.nii.gz'
        if 'NAT' in scans:
            val_df_NAT = mypath + val_dataframe['#'].astype(str) + '_NAT' + '.nii.gz'
        if 'PVP' in scans:
            val_df_PVP = mypath + val_dataframe['#'].astype(str) + '_PVP' + '.nii.gz'
        if 'VEN' in scans:
            val_df_VEN = mypath + val_dataframe['#'].astype(str) + '_VEN' + '.nii.gz'

        
        train_files = [{
                       'T2W': image_nameA,
                       'NAT': image_nameB, 
                       'ART': image_nameC, 
                       'PVP': image_nameD,  
                       'VEN': image_nameE,
                       'HBP': image_nameF,
                       'label': label_name}
                       
        for image_nameA,
            image_nameB,
            image_nameC,
            image_nameD,
            image_nameE,
            image_nameF,
            label_name in zip(
                                                                                                       train_df_T2W,
                                                                                                       train_df_NAT,
                                                                                                       train_df_ART, 
                                                                                                       train_df_PVP,
                                                                                                       train_df_VEN,
                                                                                                       train_df_HBP,
                                                                                                       trainY)]
    
        val_files = [{
                       'T2W': image_nameA,
                       'NAT': image_nameB, 
                       'ART': image_nameC, 
                       'PVP': image_nameD,  
                       'VEN': image_nameE,
                       'HBP': image_nameF,
                       'label': label_name}
                       
        for image_nameA,
            image_nameB,
            image_nameC,
            image_nameD,
            image_nameE,
            image_nameF,
            label_name in zip(
                                                                                                       val_df_T2W,
                                                                                                       val_df_NAT,
                                                                                                       val_df_ART, 
                                                                                                       val_df_PVP,
                                                                                                       val_df_VEN,
                                                                                                       val_df_HBP,
                                                                                                       valY)]

        #Train transform probability:
        train_prob=config.config()['augmentation_prob']

        if config.config()['dimensions']=='2d':
                    train_transforms = Compose(
                            [
                                LoadImaged(keys=scans),
                                AddChanneld(keys=scans),
                                NormalizeIntensityd(keys=scans, channel_wise=True),
                                ScaleIntensityd(keys=scans,minv=-1.0,maxv=1.0),
                                Resized(keys=scans, spatial_size=(32, 32)),
                                ConcatItemsd(keys=scans,name='conc_image', dim=0),
                                RandRotated(keys=['conc_image'], prob=train_prob, range_x=60.0, range_y=60.0,  keep_size=True, mode='nearest', padding_mode='border', align_corners=False),
                                RandZoomd(keys=['conc_image'], prob=train_prob, min_zoom=0.8, max_zoom=1.35, mode='nearest', keep_size=True),
                                RandFlipd(keys=['conc_image'], prob=train_prob,spatial_axis=1),
                                ToTensord(keys=['conc_image'])
                            ]
                    )
                    val_transforms = Compose(
                            [
                                LoadImaged(keys=scans),
                                AddChanneld(keys=scans),
                                NormalizeIntensityd(keys=scans, channel_wise=True),
                                ScaleIntensityd(keys=scans,minv=-1.0,maxv=1.0),
                                Resized(keys=scans, spatial_size=(32, 32,)),
                                ConcatItemsd(keys=scans,name='conc_image', dim=0),
                                ToTensord(keys=['conc_image'])
                            ]
                    )






        elif config.config()['dimensions']=='3d':
                            train_transforms = Compose(
                            [
                                LoadImaged(keys=scans),
                                AddChanneld(keys=scans),
                                NormalizeIntensityd(keys=scans, channel_wise=True),
                                ScaleIntensityd(keys=scans,minv=-1.0,maxv=1.0),
                                Resized(keys=scans, spatial_size=(32, 32, 32)),
                                ConcatItemsd(keys=scans,name='conc_image', dim=0),
                                RandRotated(keys=['conc_image'], prob=train_prob, range_x=60.0, range_y=60.0,  keep_size=True, mode='nearest', padding_mode='border', align_corners=False),
                                RandZoomd(keys=['conc_image'], prob=train_prob, min_zoom=0.8, max_zoom=1.35, mode='nearest', keep_size=True),
                                RandFlipd(keys=['conc_image'], prob=train_prob,spatial_axis=1),
                                ToTensord(keys=['conc_image']),
                            ]
                        )
                            val_transforms = Compose(
                                [
                                    LoadImaged(keys=scans),
                                    AddChanneld(keys=scans),
                                    NormalizeIntensityd(keys=scans, channel_wise=True),
                                    ScaleIntensityd(keys=scans,minv=-1.0,maxv=1.0),
                                    Resized(keys=scans, spatial_size=(32, 32, 32)),
                                    ConcatItemsd(keys=scans,name='conc_image', dim=0),
                                    ToTensord(keys=['conc_image']),
                                ]
                            )


        train_transforms.set_random_state(config.config()['experiment_seed'])
      
        def _init_fn(worker_id):
            if config.config()['experiment_seed']==None:
                return(False)
            else:
                np.random.seed(int(config.config()['experiment_seed']))

        train_ds = CacheDataset( data=train_files,transform=train_transforms,cache_rate=1.0, num_workers=8, copy_cache=False, )
        val_ds = CacheDataset( data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=5, copy_cache=False,)

            # create a training data loader
        #train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        train_loader = monai.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available(), worker_init_fn=_init_fn)
            
            # create a validation data loader
        #val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
        val_loader = monai.data.DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available(), worker_init_fn=_init_fn)
            
        return train_ds, train_loader, val_ds, val_loader