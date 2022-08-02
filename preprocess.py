import config
import crop.select_crop

def preprocess_crop():
        if config.config()['dimensions']=='2d':
            print('Lesions will be cropped from each scan at the level of the fiducial marker!')
            crop.select_crop.crop_2d(
                sequences=config.config()['sequences'],
                annotations=config.config()['annotations'],                         
                scan_paths=config.config()['scan_paths'],
                scans_folder=config.config()['scans_folder'],                           
                fiducials=config.config()['fiducials'],
                goal_folder=config.config()['goal_folder'], 
                radius_multiplier=config.config()['radius_multiplier'],    
                )
        elif config.config()['dimensions']=='3d':   
            
            crop.select_crop.crop_3d(
                sequences=config.config()['sequences'],
                annotations=config.config()['annotations'],                         
                scan_paths=config.config()['scan_paths'],
                scans_folder=config.config()['scans_folder'],                           
                fiducials=config.config()['fiducials'],
                goal_folder=config.config()['goal_folder'], 
                radius_multiplier=config.config()['radius_multiplier'],    
      
                                    )
        return()

if config.config()['crop']==True:
    preprocess_crop()
else:
   print( 'Cropping is set to False. Modify config file to enable cropping.' )
