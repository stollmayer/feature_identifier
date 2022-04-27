
import pandas as pd
import monai
import torch
from monai.transforms import NormalizeIntensityd,LoadImaged,AddChanneld,ScaleIntensityd,Resized,ConcatItemsd,  ToTensord,  Compose
import config

def dataloader(
        annotated_dataset,
        dataset_folder,
        scans,
        features,
        excel_coding
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
                'Hemorrhage/siderosis'
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

        test_dataframe = df[(df['Dataset']==excel_coding)]

        print(
            "Using %d testing"
            % (len(test_dataframe))
        )

        mypath=dataset_folder
 
        testY = []

        df_test=test_dataframe[features]
        for index, row in df_test.iterrows():     
            testY.append(row.values)

        if 'HBP' in scans:
            test_df_HBP = mypath + test_dataframe['#'].astype(str) + '_HBP' + '.nii.gz'
        if 'T2W' in scans:
            test_df_T2W = mypath + test_dataframe['#'].astype(str) + '_T2W' + '.nii.gz'
        if 'ART' in scans:
            test_df_ART = mypath + test_dataframe['#'].astype(str) + '_ART' + '.nii.gz'
        if 'NAT' in scans:
            test_df_NAT = mypath + test_dataframe['#'].astype(str) + '_NAT' + '.nii.gz'
        if 'PVP' in scans:
            test_df_PVP = mypath + test_dataframe['#'].astype(str) + '_PVP' + '.nii.gz'
        if 'VEN' in scans:
            test_df_VEN = mypath + test_dataframe['#'].astype(str) + '_VEN' + '.nii.gz'

        test_files = [{'T2W': image_nameA,
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
                                                                                                       test_df_T2W,
                                                                                                       test_df_NAT,
                                                                                                       test_df_ART, 
                                                                                                       test_df_PVP,
                                                                                                       test_df_VEN,
                                                                                                       test_df_HBP,
                                                                                                       testY)]

        
        if config.config()['dimensions']=='2d':

            test_transforms = Compose(
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
                    test_transforms = Compose(
                        [
                            LoadImaged(keys=scans),
                            AddChanneld(keys=scans),
                            NormalizeIntensityd(keys=scans, channel_wise=True),
                            ScaleIntensityd(keys=scans,minv=-1.0,maxv=1.0),
                            Resized(keys=scans, spatial_size=(32, 32, 32)),
                            ConcatItemsd(keys=scans,name='conc_image', dim=0),
                            ToTensord(keys=['conc_image'])
                        ]
                )

        # create a validation data loader

        inference_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
        inference_loader = monai.data.DataLoader(inference_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
                
        return inference_ds, inference_loader
