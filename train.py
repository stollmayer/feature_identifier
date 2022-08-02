import dataloader.dataloader
import train.train_process as train
import config
import preprocess
from models.models import model_selector

import torch
import torch.nn as nn
import time
import random
import numpy as np
import monai
from monai.utils import set_determinism


start = time.time()

def random_seeding():
        seed=config.config()['experiment_seed']
        if seed == None:
                pass
        else:
                set_determinism(seed)
        return

if config.config()['crop']==True:
    preprocess.preprocess_crop()
    preprocess_time=time.time()
    print("Cropping took:", preprocess_time)
    
for model_architecture in config.config()["model_architecture"]:
    for lr_init in config.config()["lr_init"]:
        for dropout_prob in config.config()['dropout_prob']:
            for batch_size in config.config()["batch_size"]:
                random_seeding()

                wandb_init=config.config()['wandb']
                wandb_project=config.config()['wandb_project']
                wandb_entity=config.config()['wandb_entity']
                print(config.config())
                val_interval=config.config()['val_interval']
                root=config.config()['root']
                annotated_dataset=config.config()['annotations']

                dataset_folder=config.config()['goal_folder']
                scans=config.config()["scans"]

                print("The following scans will be used:" + str(scans))
                print()

                features=config.config()["features"]

                # setting device on GPU if available, else CPU
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print('Using device:', device)
                print()

                #Additional Info when using cuda
                if device.type == 'cuda':
                    print(torch.cuda.get_device_name(0))
                    print('Memory Usage:')
                    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
                    print()

                epoch_num=config.config()["epoch_num"]

                in_channels=len(scans)
                out_channels=len(features)
                dimensions=config.config()['dimensions'] 
                pretrained=config.config()['pretrained'] 


                model_sig = model_selector(in_channels=in_channels, 
                        out_channels=out_channels, 
                        model_architecture=model_architecture, 
                        dimensions=dimensions,
                        pretrained=pretrained,
                        dropout_prob=dropout_prob,
                        device=device,
                        )


                if config.config()['crop']==True:
                        print('Selected scans will be used for cropping!')
                        preprocess.crop()
                else: 
                        print('Previously processed images will be used as input.')

                train_ds, train_loader, val_ds, val_loader = dataloader.dataloader.dataloader(
                        annotated_dataset=annotated_dataset,
                        dataset_folder=dataset_folder,
                        scans=scans,
                        features=features,
                        batch_size=batch_size,
                        )

                train.train(  
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        model_sig=model_sig,
                        epoch_num=epoch_num,
                        lr_init=lr_init,
                        batch_size=batch_size,
                        wandb_init=wandb_init,
                        wandb_project=wandb_project,
                        wandb_entity=wandb_entity,
                        val_interval=val_interval,
                        features=features,
                        dropout_prob=dropout_prob,
                        model_architecture=model_architecture, 
                        model_weights_path=config.config()["model_weights_path"]
                        )

end = time.time()
print("Whole process took:",end)
