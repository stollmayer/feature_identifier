import dataloader.test_loader
import config
import preprocess
from models.models import model_selector
import torch
import torch.nn as nn
import test.results_calculator

if config.config()['crop']==True:
    preprocess.preprocess_crop()
    preprocess_time=time.time()
    print("Cropping took:", preprocess_time)

annotated_dataset=config.config()['annotations']
annotated_dataset_novice=config.config()['annotations_novice']

dataset_folder=config.config()['ziehl_folder']
scans=config.config()["scans"]

print("The following scans will be used:" + str(scans))
print()

features=config.config()["features"]
results_folder=config.config()['results_folder']

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

epoch_num=config.config()["epoch_num"]

in_channels=len(scans)
out_channels=len(features)
dimensions=config.config()['dimensions'] 
pretrained=config.config()['pretrained'] 
model_architecture=config.config()['model_architecture'] 

model_sig = model_selector(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        model_architecture=model_architecture, 
                        dimensions=dimensions,
                        pretrained=pretrained,
                        dropout_prob=0,
                        device=device,
                        )

model_sig.load_state_dict(torch.load(config.config()['model_weights']))
model_sig.eval()

if config.config()['crop']==True:
                        print('Selected scans will be used for cropping!')
                        preprocess.crop()
else: 
                        print('Previously processed images will be used as input.')

sample_ds, sample_loader, test_ds, test_loader = dataloader.test_loader.dataloader(
                        annotated_dataset=annotated_dataset,
                        annotated_dataset_novice=annotated_dataset_novice,
                        dataset_folder=dataset_folder,
                        scans=scans,
                        features=features,
                        )

### Predict case-wise and visualize the results:

test.results_calculator.test_results_calculator(
device,
sample_loader,
model_sig,
features,
test_loader,
results_folder,
)

test.results_calculator.case_wise_results_plots(
    model_sig=model_sig,
    device=device,
    results_folder=results_folder,
    sample_loader=sample_loader,
    features=features,
    test_loader=test_loader,
    scans=scans,
    create_occlusion_map_plots=config.config()['create_occlusion_map_plots'],
    
)