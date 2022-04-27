
import os
import glob
import pandas as pd
import json
import SimpleITK as sitk
import config

def largestNumber(in_str):
    in_str=in_str.replace( "*" , "x")
    in_str=in_str.replace( "-" , "x")
    in_str=in_str.replace( "m" , " ")
    in_str=in_str.replace( " " , "")
    l=[int(x) for x in in_str.split('x') if x.isdigit()]
    return max(l) if l else None

def hyphen_split(a):
    if a.count("_") == 1:
        return a.split("_")[0]
    return "_".join(a.split("_", 2)[:2])

def crop_3d(
        sequences,
        annotations,
        scan_paths,  
        scans_folder,
        fiducials,
        ziehl_folder,
        radius_multiplier,
        ):

    padding=config.config()['crop_padding']
    df = pd.read_excel(annotations)
    df.dropna(subset = ["#"], inplace=True)
    df.dropna(subset = ["Dataset"], inplace=True)
    df_scans = pd.read_excel(scan_paths)
    folder=fiducials
    img_folder=scans_folder
    ziehl_folder=ziehl_folder
    if not os.path.exists(ziehl_folder):
        os.makedirs(ziehl_folder)
    marker=glob.glob(folder+'/*.json')
    coef=radius_multiplier
    for i in df['#'].index:
        for j in marker:
            if j==folder+str(df['#'][i])+'.mrk.json':
              with open(j) as f:
                 data = json.load(f)
              crop_origin=data['markups'][0]['controlPoints'][0]['position']
              for sequence in sequences:
                    if sequence == 'HBP_non_reg':
                            img=img_folder+'reoriented_resampled_HBP_isovolumetric/'+hyphen_split(df['#'][i])+'*.nii.gz'
                            img=glob.glob(img)[0]
                            image = sitk.ReadImage(img, sitk.sitkInt16)
                            image.GetSize()
                            crop_origin=[
                                 crop_origin[0],
                                 abs(-crop_origin[1]),
                                 abs(crop_origin[2])
                                ]
                            diameter=df['Diameter'][i]                     
                            if str('*') in str(diameter):
                                largest_radius=largestNumber(diameter)/2
                            elif str('-') in str(diameter):
                                largest_radius=largestNumber(diameter)/2     
                            elif str('x') in str(diameter):
                                largest_radius=largestNumber(diameter)/2                   
                            else:
                                largest_radius=int(diameter)/2

                            if config.config()['large_crop_for_small_lesions']==True:
                                if largest_radius<20:
                                            a_up=int(crop_origin[0]+coef*largest_radius+20)
                                            a_down=int(crop_origin[0]-coef*largest_radius-20)
                                            b_up=int(crop_origin[1]+coef*largest_radius+20)
                                            b_down=int(crop_origin[1]-coef*largest_radius-20)
                                            c_up=int(crop_origin[2]+coef*largest_radius+20)
                                            c_down=int(crop_origin[2]-coef*largest_radius-20)                    
                
                                else:
                                    a_up=int(crop_origin[0]+coef*largest_radius+5)
                                    a_down=int(crop_origin[0]-coef*largest_radius-5)
                                    b_up=int(crop_origin[1]+coef*largest_radius+5)
                                    b_down=int(crop_origin[1]-coef*largest_radius-5)
                                    c_up=int(crop_origin[2]+coef*largest_radius+5)
                                    c_down=int(crop_origin[2]-coef*largest_radius-5)
                            else:
                                    a_up=int(crop_origin[0]+coef*largest_radius+padding)
                                    a_down=int(crop_origin[0]-coef*largest_radius-padding)
                                    b_up=int(crop_origin[1]+coef*largest_radius+padding)
                                    b_down=int(crop_origin[1]-coef*largest_radius-padding)
                                    c_up=int(crop_origin[2]+coef*largest_radius+padding)
                                    c_down=int(crop_origin[2]-coef*largest_radius-padding)
                                
                            image_cropped = image[
                                    a_down:a_up,
                                    b_down:b_up, 
                                    c_down:c_up,
                                                ]

                            print( ziehl_folder+df['#'][i]+'_'+sequence[0:3]+".nii.gz")    
                            sitk.WriteImage(image_cropped, ziehl_folder+df['#'][i]+'_'+sequence[0:3]+".nii.gz")
 
                    else:
                      for u, row in df_scans.iterrows():
                        if str(str(row['Patient ID'])+'_'+str(row['Scan ID'])) == str(hyphen_split(df['#'][i])):                                                   

                          if str(row[sequence])=='ERR':
                              pass
                          elif str(row[sequence])=='nan':
                              pass
                          
                          else:
                            img=img_folder+'registered_elastix_isovol/'+row['Patient ID']+'_'+row['Scan ID']+'_ELASTIX_isovol_'+str(row[sequence])
                            image = sitk.ReadImage(img, sitk.sitkInt16)
                            image.GetSize()[1]
                            crop_origin=[
                                 crop_origin[0],
                                 abs(-crop_origin[1]),
                                 abs(crop_origin[2])
                                ]
                            diameter=df['Diameter'][i]

                            if str('*') in str(diameter):
                                largest_radius=largestNumber(diameter)/2
                            elif str('-') in str(diameter):
                                largest_radius=largestNumber(diameter)/2     
                            elif str('x') in str(diameter):
                                largest_radius=largestNumber(diameter)/2                   
                            else:
                                largest_radius=int(diameter)/2
                            
                            if config.config()['large_crop_for_small_lesions']==True:
                                if largest_radius<20:
                                            a_up=int(crop_origin[0]+coef*largest_radius+20)
                                            a_down=int(crop_origin[0]-coef*largest_radius-20)
                                            b_up=int(crop_origin[1]+coef*largest_radius+20)
                                            b_down=int(crop_origin[1]-coef*largest_radius-20)
                                            c_up=int(crop_origin[2]+coef*largest_radius+20)
                                            c_down=int(crop_origin[2]-coef*largest_radius-20)
                                else:
                                    a_up=int(crop_origin[0]+coef*largest_radius+5)
                                    a_down=int(crop_origin[0]-coef*largest_radius-5)
                                    b_up=int(crop_origin[1]+coef*largest_radius+5)
                                    b_down=int(crop_origin[1]-coef*largest_radius-5)
                                    c_up=int(crop_origin[2]+coef*largest_radius+5)
                                    c_down=int(crop_origin[2]-coef*largest_radius-5)

                            else:
                                    a_up=int(crop_origin[0]+coef*largest_radius+padding)
                                    a_down=int(crop_origin[0]-coef*largest_radius-padding)
                                    b_up=int(crop_origin[1]+coef*largest_radius+padding)
                                    b_down=int(crop_origin[1]-coef*largest_radius-padding)
                                    c_up=int(crop_origin[2]+coef*largest_radius+padding)
                                    c_down=int(crop_origin[2]-coef*largest_radius-padding)

                            image_cropped = image[
                                    a_down:a_up,
                                    b_down:b_up, 
                                    c_down:c_up,]

                            print( ziehl_folder+df['#'][i]+'_'+sequence[0:3]+".nii.gz")
                            sitk.WriteImage(image_cropped, ziehl_folder+df['#'][i]+'_'+sequence[0:3]+".nii.gz")
    return()

 

def crop_2d(
        sequences,
        annotations,
        scan_paths,    
        scans_folder,
        fiducials,
        ziehl_folder,
        radius_multiplier,

        ):

    padding=config.config()['crop_padding']
    df = pd.read_excel(annotations)
    df.dropna(subset = ["#"], inplace=True)
    df.dropna(subset = ["Dataset"], inplace=True)
    df_scans = pd.read_excel(scan_paths)
    folder=fiducials
    img_folder=scans_folder
    ziehl_folder=ziehl_folder
    if not os.path.exists(ziehl_folder):
        os.makedirs(ziehl_folder)
    marker=glob.glob(folder+'/*.json')
    coef=radius_multiplier
    for i in df['#'].index:
        for j in marker:
            if j==folder+str(df['#'][i])+'.mrk.json':
                with open(j) as f:
                    data = json.load(f)
                crop_origin=data['markups'][0]['controlPoints'][0]['position']
                
                for sequence in sequences:
                          if sequence == 'HBP_non_reg':
                            img=img_folder+'reoriented_resampled_HBP_isovolumetric/'+hyphen_split(df['#'][i])+'*.nii.gz'
                            img=glob.glob(img)[0]
                            
                            image = sitk.ReadImage(img, sitk.sitkInt16)
                            image.GetSize()
                            crop_origin=[
                                 crop_origin[0],
                                 abs(-crop_origin[1]),
                                 abs(crop_origin[2])
                                ]
                            diameter=df['Diameter'][i]




                            if str('*') in str(diameter):
 
                                largest_radius=largestNumber(diameter)/2
                            elif str('-') in str(diameter):
   
                                largest_radius=largestNumber(diameter)/2     
                            elif str('x') in str(diameter):
  
                                largest_radius=largestNumber(diameter)/2                   
                            else:
                                largest_radius=int(diameter)/2

                            if config.config()['large_crop_for_small_lesions']==True:
                                        if largest_radius<20:
                                            middle=int(crop_origin[2])
                                            a_up=int(crop_origin[0]+coef*largest_radius+20)
                                            a_down=int(crop_origin[0]-coef*largest_radius-20)
                                            b_up=int(crop_origin[1]+coef*largest_radius+20)
                                            b_down=int(crop_origin[1]-coef*largest_radius-20)
                                            c_up=int(crop_origin[2]+coef*largest_radius+20)
                                            c_down=int(crop_origin[2]-coef*largest_radius-20)
                                        else:
                                                        middle=int(crop_origin[2])
                                                        a_up=int(crop_origin[0]+coef*largest_radius+5)
                                                        a_down=int(crop_origin[0]-coef*largest_radius-5)
                                                        b_up=int(crop_origin[1]+coef*largest_radius+5)
                                                        b_down=int(crop_origin[1]-coef*largest_radius-5)
                                                        c_up=int(crop_origin[2]+coef*largest_radius+5)
                                                        c_down=int(crop_origin[2]-coef*largest_radius-5)
                            else:
                                            middle=int(crop_origin[2])
                                            a_up=int(crop_origin[0]+coef*largest_radius+padding)
                                            a_down=int(crop_origin[0]-coef*largest_radius-padding)
                                            b_up=int(crop_origin[1]+coef*largest_radius+padding)
                                            b_down=int(crop_origin[1]-coef*largest_radius-padding)
                                            c_up=int(crop_origin[2]+coef*largest_radius+padding)
                                            c_down=int(crop_origin[2]-coef*largest_radius-padding)
            
                            image_cropped = image[
                                    a_down:a_up,
                                    b_down:b_up, 
                                    middle
                                                ]
                            


                                
                            print( ziehl_folder+df['#'][i]+'_'+sequence[0:3]+".nii.gz")
                            
                            sitk.WriteImage(image_cropped, ziehl_folder+df['#'][i]+'_'+sequence[0:3]+".nii.gz")
 
                          else:
                            for u, row in df_scans.iterrows():
                                if str(str(row['Patient ID'])+'_'+str(row['Scan ID'])) == str(hyphen_split(df['#'][i])):                                                   
        
                                  if str(row[sequence])=='ERR':
                                      pass
                                  elif str(row[sequence])=='nan':
                                      pass
                                  
                                  else:
                                    img=img_folder+'registered_elastix_isovol/'+row['Patient ID']+'_'+row['Scan ID']+'_ELASTIX_isovol_'+str(row[sequence])
                                    
        
                        
                                    image = sitk.ReadImage(img, sitk.sitkInt16)
                                    image.GetSize()[1]
                                    crop_origin=[
                                         crop_origin[0],
                                         abs(-crop_origin[1]),
                                         abs(crop_origin[2])
                                        ]
                                    diameter=df['Diameter'][i]
                                    

                                    if str('*') in str(diameter):
                                        largest_radius=largestNumber(diameter)/2
                                    elif str('-') in str(diameter):
                                        largest_radius=largestNumber(diameter)/2     
                                    elif str('x') in str(diameter):
                                        largest_radius=largestNumber(diameter)/2                   
                                    else:
                                        largest_radius=int(diameter)/2
                                 
                                    if config.config()['large_crop_for_small_lesions']==True:
                                        if largest_radius<20:
                                            middle=int(crop_origin[2])
                                            a_up=int(crop_origin[0]+coef*largest_radius+20)
                                            a_down=int(crop_origin[0]-coef*largest_radius+20)
                                            b_up=int(crop_origin[1]+coef*largest_radius+20)
                                            b_down=int(crop_origin[1]-coef*largest_radius+20)
                                            c_up=int(crop_origin[2]+coef*largest_radius+20)
                                            c_down=int(crop_origin[2]-coef*largest_radius+20)
                                        else:
                                                middle=int(crop_origin[2])
                                                a_up=int(crop_origin[0]+coef*largest_radius+5)
                                                a_down=int(crop_origin[0]-coef*largest_radius-5)
                                                b_up=int(crop_origin[1]+coef*largest_radius+5)
                                                b_down=int(crop_origin[1]-coef*largest_radius-5)
                                                c_up=int(crop_origin[2]+coef*largest_radius+5)
                                                c_down=int(crop_origin[2]-coef*largest_radius-5)
                                    else:
                                            middle=int(crop_origin[2])
                                            a_up=int(crop_origin[0]+coef*largest_radius+padding)
                                            a_down=int(crop_origin[0]-coef*largest_radius-padding)
                                            b_up=int(crop_origin[1]+coef*largest_radius+padding)
                                            b_down=int(crop_origin[1]-coef*largest_radius-padding)
                                            c_up=int(crop_origin[2]+coef*largest_radius+padding)
                                            c_down=int(crop_origin[2]-coef*largest_radius-padding)
            
                                    image_cropped = image[
                                            a_down:a_up,
                                            b_down:b_up, 
                                            middle
                                                        ]

                                    print( ziehl_folder+df['#'][i]+'_'+sequence[0:3]+".nii.gz")
                                    sitk.WriteImage(image_cropped, ziehl_folder+df['#'][i]+'_'+sequence[0:3]+".nii.gz")
    
    return()
