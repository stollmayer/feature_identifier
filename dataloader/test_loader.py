
import monai
import pandas as pd
from monai.transforms import NormalizeIntensityd,LoadImaged,AddChanneld,ScaleIntensityd,Resized,ConcatItemsd,ToTensord,Compose
import torch
import config

def dataloader(
        annotated_dataset,
        annotated_dataset_novice,
        dataset_folder,
        scans,
        features,
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

        df_novice = pd.read_excel(annotated_dataset_novice)
        df_novice=df_novice[['#','FNH','HCC','MET','Other','Dataset',
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
                ]]

        df_novice=df_novice.dropna()
        df_novice = df_novice.astype({'#': str,'FNH': int,'HCC': int,'MET': int,'Other': int,
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
                        })

        for index, row in df_novice.iterrows():   
            if row['FNH']==0 and row['HCC']==0 and row['MET']==0 and row['Other']==0:
                df_novice.drop(index, inplace=True)
        
        df_novice.isnull().sum()
        df_novice.head()
        df_novice.describe()
        pd.set_option('display.max_columns', None)

        val_dataframe = df[(df['Dataset']==2)]

        if config.config()['lesion_class']=='FNH':
            for index, row in df.iterrows():   
                if  row['FNH']==0:
                    df.drop(index, inplace=True)
        if config.config()['lesion_class']=='HCC':
            for index, row in df.iterrows():   
                if  row['HCC']==0:
                    df.drop(index, inplace=True)               
        if config.config()['lesion_class']=='MET':
            for index, row in df.iterrows():   
                if  row['MET']==0:
                    df.drop(index, inplace=True)
        if config.config()['lesion_class']=='Other':
            for index, row in df.iterrows():   
                if  row['Other']==0:
                    df.drop(index, inplace=True)

        test_dataframe = df[(df['Dataset']==2)]
        print(
            "Using %d lesions for testing (expert)"
            % (len(test_dataframe))
        )

        val_dataframe_novice = df_novice[(df_novice['Dataset']==2)]

        if config.config()['lesion_class']=='FNH':
            for index, row in df_novice.iterrows():   
                if  row['FNH']==0:
                    df_novice.drop(index, inplace=True)
        if config.config()['lesion_class']=='HCC':
            for index, row in df_novice.iterrows():   
                if  row['HCC']==0:
                    df_novice.drop(index, inplace=True)               
        if config.config()['lesion_class']=='MET':
            for index, row in df_novice.iterrows():   
                if  row['MET']==0:
                    df_novice.drop(index, inplace=True)
        if config.config()['lesion_class']=='Other':
            for index, row in df_novice.iterrows():   
                if  row['Other']==0:
                    df_novice.drop(index, inplace=True)

        test_dataframe_novice = df_novice[(df_novice['Dataset']==2)]
        print(
            "Using %d lesions for testing (novice)"
            % (len(test_dataframe_novice))
        )

        mypath=dataset_folder

        valY = []
        testY = []

        df_val=val_dataframe[features]
        for index, row in df_val.iterrows():     
            valY.append(row.values)

        df_test=test_dataframe[features]
        for index, row in df_test.iterrows():     
            testY.append(row.values)

        classes=['FNH','HCC','MET','Other']
        classesY=[]
        df_test_classes=test_dataframe[classes]
        for index, row in df_test_classes.iterrows():     
            classesY.append(row.values)

        valY_novice = []
        testY_novice = []

        df_val_novice=val_dataframe_novice[features]
        for index, row in df_val_novice.iterrows():     
            valY_novice.append(row.values)

        df_test_novice=test_dataframe_novice[features]
        for index, row in df_test_novice.iterrows():     
            testY_novice.append(row.values)

        classesY_novice=[]
        df_test_classes_novice=test_dataframe_novice[classes]
        for index, row in df_test_classes_novice.iterrows():     
            classesY_novice.append(row.values)

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

        val_files = [{ 'HBP': image_nameA,
                       'T2W': image_nameB,
                       'NAT': image_nameC, 
                       'ART': image_nameD, 
                       'PVP': image_nameE,  
                       'VEN': image_nameF,
                       'label': label_name}
                     
        for image_nameA,
        image_nameB,
        image_nameC,
        image_nameD,
        image_nameE,
        image_nameF,
        label_name in zip(
                                                                                                       val_df_HBP,
                                                                                                       val_df_T2W,
                                                                                                       val_df_NAT,
                                                                                                       val_df_ART, 
                                                                                                       val_df_PVP,
                                                                                                       val_df_VEN,
                                                                                                       valY)]


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
                       'label': label_name,
                       'label_novice':label_name_novice,
                       'lesion_class':lesion_class,
                       'lesion_class_novice':lesion_class_novice,
                       }
                     
        for image_nameA,
        image_nameB,
        image_nameC,
        image_nameD,
        image_nameE,
        image_nameF,
        label_name,
        label_name_novice,
        lesion_class,
        lesion_class_novice in zip(                                                                                            
                                                                                                       test_df_T2W,
                                                                                                       test_df_NAT,
                                                                                                       test_df_ART, 
                                                                                                       test_df_PVP,
                                                                                                       test_df_VEN,
                                                                                                       test_df_HBP,
                                                                                                       testY,
                                                                                                       testY_novice,
                                                                                                       classesY,
                                                                                                       classesY_novice
                                                                                                       )]

        
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
        val_ds = monai.data.Dataset(data=val_files, transform=test_transforms)
        val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
        test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
        test_loader = monai.data.DataLoader(test_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())           
        return val_ds, val_loader, test_ds, test_loader
