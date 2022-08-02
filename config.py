def config():
    config = {
        # Perform cropping: if True, pick dimension and list of sequence codes to perform on. Modify 'scan_paths' to change scan paths data sheet, 'fiducials' to change marker paths.
        "crop": False,  # True, False
        "dimensions": "3d",  # 2d, 3d,
        "sequences": [
            "T2W_non_reg",
            "NAT_non_reg",
            "ART_non_reg",
            "PVP_non_reg",
            "VEN_non_reg",
            "HBP_non_reg",
        ],
        "scans_folder": "",
        "scan_paths": "",
        "fiducials": "",
        "large_crop_for_small_lesions": False,  # crop radius extended by 2cm for lesions smaller than 40 mm --- set to False
        "root": "./",
        "annotations": "./Anonym MRI liver lesions_expert.xlsx",
        "annotations_novice": "./Anonym MRI liver lesions_novice.xlsx",

        "radius_multiplier": 1.00,
        "crop_padding": 2,
        "goal_folder": "./fll_cropped/", # folder where the cropped cubic volumes are located (or where they sould be cropped
        # scan names for calculations and plotting
        "scans": [
            "T2W",
            "NAT",
            "ART",
            "PVP",
            "VEN",
            "HBP",
        ],
        # feature names for calculations and plotting
        "features": [
            "Early enhancement",
            "Washout",
            "Delayed phase enhancement",
            "Peripheral enhancement",
            "Central scar",
            "Capsule",
            "T2 hyperintensity",
            "Iso- or hyperintensity on venous phase",
            "Hypoenhancing core",
            "Hemorrhage/siderosis",
        ],
        # Training configurations:
        "wandb": False,
        "wandb_project": "",
        "wandb_entity": "",
        "experiment_seed": 58736530,  # seed number or None
        "augmentation_prob": 0.8,
        "epoch_num": int(500),
        "lr_init": [
            1e-4,
        ],
        "pretrained": False,
        "dropout_prob": [0, 0.25, 0.5, 0.75],  # for DenseNets
        "dropout_prob": "_",  # for EfficientNets
        "batch_size": [32,], #32, 64, 128
        "weight_decay": 0.2,

        "val_interval": 20,
        "model_weights_path": "./results_folder/model_weights",
        "model_architecture": [
            "EfficientNetB0",
            "EfficientNetB1",
            "EfficientNetB2",
            "EfficientNetB3",
            "EfficientNetB4",
            "EfficientNetB5",
            "EfficientNetB6",
            "EfficientNetB7",           
        ],

        "model_architecture": [
            "DenseNet121",
            "DenseNet169",
            "DenseNet201",
            "DenseNet264",
        ],  # Leave this in to only train DenseNets.

  
        # For testing a single model name:
        "model_architecture": "EfficientNetB0",

        # Paremeters for test results reproduction:
        "model_weights": "./results_folder/model_weights/EfficientNetB0_0.0001_32___500_weights.pth",
        
        "results_folder": "./results_folder/",
        "lesion_class": "all",  # FNH, HCC, MET, Other, all
        "create_occlusion_map_plots": False, 

        "plot": False,
        "plot_ROC": True,  # plot cutoff sample ROCs
    }

    return config
