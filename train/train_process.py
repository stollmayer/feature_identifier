
import os
import torch
import wandb
import numpy as np
import torch.nn as nn
from monai.utils import set_determinism
from monai.optimizers import Novograd
import random
from metrics.feature_metrics import feature_specific_metrics
import config

def train(
        train_loader,
        val_loader,
        device,
        model_sig,
        model_architecture,
   
        epoch_num,
        lr_init,
        dropout_prob,

        wandb_init,
        wandb_project,
        wandb_entity,
        batch_size,

        val_interval,
        features,
        model_weights_path,
        ):
        weight_decay = config.config()['weight_decay']
        device = torch.device(device)
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(model_sig.parameters(), lr_init, weight_decay=weight_decay)

        val_interval = val_interval
        best_metric = -1
        epoch_loss_values = []

        seed=config.config()['experiment_seed']
        if seed == None:
                        pass
        else:
                        set_determinism(seed)

        epoch_num = epoch_num
        name=str(model_architecture+"_"+str(lr_init)+"_"+str(batch_size)+"_"+str(dropout_prob)+"_"+str(epoch_num))

        if wandb_init==True:
            run = wandb.init(
                    project=wandb_project, 
                    entity=wandb_entity, 
                    reinit=True,
                    name=name
            )
            wandb.log({"model_architecture": model_architecture,})           
            wandb.log({"lr_init": lr_init,})
            wandb.log({"batch_size": batch_size,})
            wandb.log({"dropout_prob": dropout_prob,})

        best_metric = -1
        epoch_loss_values = []
        epoch_loss_val_values = []

        for epoch in range(epoch_num):
                print("-" * 10)
                print(f"epoch {epoch + 1}/{epoch_num}")
                
                epoch_loss = 0
                epoch_loss_val= 0
                step = 0
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)

                for batch_data in train_loader:
                    model_sig.train()
                    step += 1
                    inputs, labels = batch_data["conc_image"].to(device), batch_data["label"].to(device)                  
                    optimizer.zero_grad()

                    outputs = model_sig(inputs)
                    loss = loss_function((outputs),  labels.type(torch.float))
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    model_sig.eval()
                    with torch.no_grad():
                            y_pred = torch.cat([y_pred, model_sig(inputs)], dim=0)
                            y = torch.cat([y, labels], dim=0)

                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                if wandb_init==True:
                    wandb.log({"Training loss": loss,'epoch': epoch + 1,})
                    wandb.log({"Training loss /step": epoch_loss,'epoch': epoch + 1,})

                model_sig_results=[]
                ref_labels = []
                model_sig_results.extend(y_pred.cpu().numpy())
                ref_labels.extend(y.cpu().numpy())
                feature_metrics, avg_feature_rocauc = feature_specific_metrics(np.array(model_sig_results), np.array(ref_labels), features)
                
                if wandb_init==True:
                    wandb.log({"train_feature_metrics": feature_metrics,'epoch': epoch + 1,})
                    wandb.log({"train_avg_feature_rocauc": avg_feature_rocauc,'epoch': epoch + 1,})

                if (epoch + 1) % val_interval == 0:
                    model_sig.eval()
                    with torch.no_grad():
                        y_pred = torch.tensor([], dtype=torch.float32, device=device)
                        y = torch.tensor([], dtype=torch.long, device=device)
                        step = 0
                        for val_data in val_loader:
                            step += 1
                            val_images, val_labels = val_data["conc_image"].to(device), val_data["label"].to(device)
                            
                            y_pred = torch.cat([y_pred, model_sig(val_images)], dim=0)
                            y = torch.cat([y, val_labels], dim=0)
                            outputs_val=model_sig(val_images)
                            loss_val = loss_function(outputs_val,val_labels.type(torch.float))
                            epoch_loss_val += loss_val.item()

                        epoch_loss_val /= step
                        if wandb_init==True:
                            wandb.log({"Validation loss": loss_val,'epoch': epoch + 1,}) 
                            wandb.log({"Validation loss /step": epoch_loss_val,'epoch': epoch + 1,}) 
                        epoch_loss_val_values.append(epoch_loss_val)

                        model_sig_results=[]
                        ref_labels = []
                        model_sig_results.extend(y_pred.cpu().numpy())
                        ref_labels.extend(y.cpu().numpy())
                        val_feature_metrics,val_avg_feature_rocauc = feature_specific_metrics(np.array(model_sig_results), np.array(ref_labels), features)
                       
                        if wandb_init==True:
                            wandb.log({"val_feature_metrics": val_feature_metrics,'epoch': epoch + 1,})
                            wandb.log({"val_avg_feature_rocauc": val_avg_feature_rocauc,'epoch': epoch + 1,})

                        if val_avg_feature_rocauc > best_metric:
                            best_metric = val_avg_feature_rocauc
                            if wandb_init==True:
                                wandb.log({"best_metric": val_avg_feature_rocauc,'epoch': epoch + 1,})
                            torch.save(model_sig.state_dict(), model_weights_path+"/"+name+"_weights.pth")
                            if wandb_init==True:
                                torch.save(model_sig.state_dict(),os.path.join(wandb.run.dir, wandb.run.name +"_weights.pth"))
       
        if wandb_init==True:
            run.finish()
        else:
            return()