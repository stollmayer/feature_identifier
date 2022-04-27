
from torch import nn
from monai.networks.nets import(
            densenet264,
            densenet201,
            densenet169,
            densenet121,
            EfficientNetBN,
                            )

def model_selector(in_channels, 
          out_channels, 
          model_architecture, 
          dimensions,
          pretrained,
          dropout_prob,
          device,
          
          ):
        if dimensions=="3d":
            dimensions=3
        elif dimensions=="2d":
            dimensions=2

        class model_class(nn.Module):
                    def __init__(self, out_channels=out_channels):
                        super().__init__()
                        if model_architecture == "DenseNet264":
                            model = densenet264(spatial_dims=dimensions, 
                                                                       in_channels=in_channels, 
                                                                       out_channels=out_channels, 
                                                                       dropout_prob=dropout_prob,
                                                                       pretrained=pretrained
                                                                       ).to(device) 
                            
                        elif model_architecture == "DenseNet201":
                            model = densenet201(spatial_dims=dimensions, 
                                                                       in_channels=in_channels, 
                                                                       out_channels=out_channels, 
                                                                       dropout_prob=dropout_prob,
                                                                       pretrained=pretrained
                                                                       ).to(device) 
                            
                        elif model_architecture == "DenseNet169":
                            model = densenet169(spatial_dims=dimensions, 
                                                                       in_channels=in_channels, 
                                                                       out_channels=out_channels, 
                                                                       dropout_prob=dropout_prob,
                                                                       pretrained=pretrained
                                                                       ).to(device) 
                            
                        elif model_architecture == "DenseNet121":
                            model = densenet121(  spatial_dims=dimensions, 
                                                                       in_channels=in_channels, 
                                                                       out_channels=out_channels, 
                                                                       dropout_prob=dropout_prob,
                                                                       pretrained=pretrained
                                                                       ).to(device) 
                            
                        elif model_architecture == "EfficientNetB7":
                            model = EfficientNetBN("efficientnet-b7",
                                                                     pretrained=pretrained, 
                                                                     progress=pretrained, 
                                                                     spatial_dims=dimensions, 
                                                                     in_channels=in_channels, 
                                                                     num_classes=out_channels,
                                                                       ).to(device)   
                        elif model_architecture == "EfficientNetB6":
                            model = EfficientNetBN("efficientnet-b6",
                                                                     pretrained=pretrained, 
                                                                     progress=pretrained, 
                                                                     spatial_dims=dimensions, 
                                                                     in_channels=in_channels, 
                                                                     num_classes=out_channels,
                                                                       ).to(device)   
                                                       
                        elif model_architecture == "EfficientNetB5":
                            model = EfficientNetBN("efficientnet-b5",
                                                                     pretrained=pretrained, 
                                                                     progress=pretrained, 
                                                                     spatial_dims=dimensions, 
                                                                     in_channels=in_channels, 
                                                                     num_classes=out_channels,
                                                                       ).to(device)
                        elif model_architecture == "EfficientNetB4":
                            model = EfficientNetBN("efficientnet-b4",
                                                                     pretrained=pretrained, 
                                                                     progress=pretrained, 
                                                                     spatial_dims=dimensions, 
                                                                     in_channels=in_channels, 
                                                                     num_classes=out_channels,
                                                                       ).to(device)   
                        elif model_architecture == "EfficientNetB3":
                            model = EfficientNetBN("efficientnet-b3",
                                                                     pretrained=pretrained, 
                                                                     progress=pretrained, 
                                                                     spatial_dims=dimensions, 
                                                                     in_channels=in_channels, 
                                                                     num_classes=out_channels,
                                                                       ).to(device) 
                            
                        elif model_architecture == "EfficientNetB2":
                            model = EfficientNetBN("efficientnet-b2",

                                                                     spatial_dims=dimensions, 
                                                                     in_channels=in_channels, 
                                                                     num_classes=out_channels,
                                                                       ).to(device)   
                        elif model_architecture == "EfficientNetB1":
                            model = EfficientNetBN("efficientnet-b1",
                                                                     pretrained=pretrained, 
                                                                     progress=pretrained, 
                                                                     spatial_dims=dimensions, 
                                                                     in_channels=in_channels, 
                                                                     num_classes=out_channels,
                                                                       ).to(device)   
                            
                        elif model_architecture == "EfficientNetB0":
                            model = EfficientNetBN("efficientnet-b0",
                                                                     pretrained=pretrained, 
                                                                     progress=pretrained, 
                                                                     spatial_dims=dimensions, 
                                                                     in_channels=in_channels, 
                                                                     num_classes=out_channels,
                                                                       ).to(device)                                                                                                                                                                 

                        self.base_model = model 
                        self.sigm = nn.Sigmoid()
                
                    def forward(self, x):
                            return self.sigm(self.base_model(x))
                            
        return(model_class())