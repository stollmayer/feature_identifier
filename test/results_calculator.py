
import monai
import matplotlib.pyplot as plt
import ntpath
import pandas as pd
import numpy as np
import torch
import metrics.feature_metrics
import config
import matplotlib.cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def test_results_calculator(
device,
sample_loader,
model_sig,
features,
test_loader,
results_folder,
):

    youden_dictionary = dict.fromkeys(features, "nan")
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        step = 0

        if config.config()['lesion_class']=='all':
            for val_data in test_loader:
                step += 1
                val_images, val_labels = val_data["conc_image"].to(device), val_data["label"].to(device)
                y_pred = torch.cat([y_pred, model_sig(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)

        if config.config()['lesion_class']!='all':
            for val_data in sample_loader:
                step += 1
                val_images, val_labels = val_data["conc_image"].to(device), val_data["label"].to(device)
                y_pred = torch.cat([y_pred, model_sig(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)

        model_sig_results=[]
        ref_labels = []
        model_sig_results.extend(y_pred.cpu().numpy())
        ref_labels.extend(y.cpu().numpy())
        val_feature_metrics,val_avg_feature_rocauc = metrics.feature_metrics.feature_specific_metrics(np.array(model_sig_results), np.array(ref_labels), features,)
        
        # Create a dictionary for the cutoff values
        for f in features:               
            youden_dictionary[f] = {
                'Youden-corrected threshold': val_feature_metrics[f]['Youden-corrected threshold'],
                'Validation cutoff point':  val_feature_metrics[f]['cutoff_index']
            }
        
        # Plot ROC curves for the cutoff sample 
        if config.config()['plot_ROC']==True:
                        plt.figure()
                        colors=['forestgreen',
                        'lightcoral',
                        'brown',
                        'yellowgreen',
                        'skyblue',
                        'deeppink',
                        'peru',
                        'blue',
                        'gold',
                        'indigo',
                        ]
                        lw = 2
                        o=0
                        for f in features:
                            tpr=val_feature_metrics[f]['tpr_dict'] 
                            fpr=val_feature_metrics[f]['fpr_dict'] 
                            roc_auc=val_feature_metrics[f]['ROC_AUC'] 
                            idx=val_feature_metrics[f]['cutoff_index']
                            plt.plot(
                                fpr,
                                tpr,
                                color=colors[o],
                                lw=lw,
                                label=f+" (AUC = %0.2f)" % roc_auc,
                            )
                            plt.plot(fpr[idx], tpr[idx], marker="o", markersize=12, markeredgecolor='red', markerfacecolor=colors[o])

                            o+=1
                        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel("1 - Specificity")
                        plt.ylabel("Sensitivity")
                        plt.title('ROC curves for each feature')
                        plt.legend(loc="lower right")
                        plt.savefig(results_folder+''.join(e for e in str('auc_plot') if e.isalnum())+'_set_ROC_plot.svg', dpi=300) 
                        plt.show()

        # Calculate results on the test dataset  
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        y_novice = torch.tensor([], dtype=torch.long, device=device)

        step = 0
        file_names=[]
        lesion_class_list=[]
        for test_data in test_loader:
            step += 1
            test_images, test_labels, test_labels_novice = test_data["conc_image"].to(device), test_data["label"].to(device), test_data["label_novice"].to(device)
            lesion_class = test_data['lesion_class'].cpu().numpy()[0]
            lesion_class_list.append(lesion_class)
            y_pred = torch.cat([y_pred, model_sig(test_images)], dim=0)
            y = torch.cat([y, test_labels], dim=0)
            y_novice = torch.cat([y_novice, test_labels_novice], dim=0)

            name=ntpath.basename(test_data['T2W_meta_dict']['filename_or_obj'][0])[:-11]
            file_names.append(name)

        model_sig_results=[]
        ref_labels = []
        ref_labels_novice = []
        model_sig_results.extend(y_pred.cpu().numpy())
        ref_labels.extend(y.cpu().numpy())
        ref_labels_novice.extend(y_novice.cpu().numpy())

        test_feature_metrics,test_avg_feature_rocauc = metrics.feature_metrics.test_feature_specific_metrics(cases=file_names,
                                                                                                             pred=np.array(model_sig_results), 
                                                                                                             target=np.array(ref_labels),
                                                                                                             target_novice=np.array(ref_labels_novice),
                                                                                                             features=features,
                                                                                                             youden_dictionary=youden_dictionary,
                                                                                                             lesion_class_list=lesion_class_list
                                                                                                             )
        d = test_feature_metrics
        df = pd.DataFrame.from_dict(d, orient='index')
        df.to_csv(results_folder+'test_feature_metrics_'+str(config.config()['lesion_class']+'.csv'))
    return()


# create results plots for each lesion
def case_wise_results_plots(
    model_sig,
    device,
    results_folder,
    sample_loader,
    features,
    test_loader,
    scans,
    create_occlusion_map_plots
):
        youden_dictionary = dict.fromkeys(features, "nan")
        occ_sens = monai.visualize.OcclusionSensitivity(
         nn_module=model_sig, n_batch=10, stride=[ 4, 4, 8],mask_size=[ 16, 16, 32])
 
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)

        step = 0
        for val_data in test_loader:
                step += 1
                val_images, val_labels = val_data["conc_image"].to(device), val_data["label"].to(device)
                y_pred = torch.cat([y_pred, model_sig(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
        model_sig_results=[]
        ref_labels = []
        model_sig_results.extend(y_pred.detach().cpu().numpy())
        ref_labels.extend(y.detach().cpu().numpy())
        val_feature_metrics, val_avg_feature_rocauc = metrics.feature_metrics.feature_specific_metrics(np.array(model_sig_results), np.array(ref_labels), features)
        
         # Create a dictionary for the cutoff values:
        for f in features:               
            youden_dictionary[f] = {
                'Youden-corrected threshold': val_feature_metrics[f]['Youden-corrected threshold'],
                'Validation cutoff point':  val_feature_metrics[f]['cutoff_index']
            }

        # Calculate the test dataset results:
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        step = 0
        for test_data in test_loader:
            step += 1
            test_images, test_labels = test_data["conc_image"].to(device), test_data["label"].to(device)

            y_pred =  model_sig(test_images)
            y =  test_labels
            file_name=ntpath.basename(test_data['T2W_meta_dict']['filename_or_obj'][0])[:-11]
            
            model_sig_case_results=(y_pred.detach().cpu().numpy())
            ref_case_labels=(y.cpu().numpy())
            test_feature_metrics = metrics.feature_metrics.test_feature_specific_metrics_single_case(file_name,
                                                                                                                np.array(model_sig_case_results), 
                                                                                                                np.array(ref_case_labels), 
                                                                                                                features,
                                                                                                                youden_dictionary
                                                                                                                )
            
            # Plot occlusion sensitivity maps
            img = test_images

            if create_occlusion_map_plots ==True:
                occ_map, occ_most_prob = occ_sens(x=img)
                x=len(scans)
                y=len(features)
                cmap= plt.get_cmap('gnuplot2')
                for j in range(y):
                    fig, ax = plt.subplots(2, x, sharex='col', sharey='row',figsize=(36,10))
                    for i in range(x):

                            max_value = torch.max(occ_map[0][:,:,:,16,j]).cpu()
                            min_value = torch.min(occ_map[0][:,:,:,16,j]).cpu()
                            norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
                            ax[0,i].imshow(img[0,i,:,:,16].cpu(),cmap='gray',)
                            im1=ax[0,i].imshow(img[0,i,:,:,16].cpu(),cmap='gray',norm=None )
                            divider1 = make_axes_locatable(ax[0,i])
                            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                            ax[0,i].set_ylabel(scans[i], rotation=90, fontsize=12, labelpad=10)
                        
                            fig.colorbar(im1, cax=cax1, orientation='vertical',norm=norm )  

                            ax[0,i].set_title(features[j]+'\n'+'GT:'+str(test_feature_metrics[str(features[j])]['True labels'][0])
                                                +' '+', prob:'+str("%0.2f" % (test_feature_metrics[str(features[j])]['predictions'][0])
                                                +' '+', pred:'+str(int(test_feature_metrics[str(features[j])]['Youden-corrected predictions'][0]))),
                                fontsize = 12,)
                            ax[1,i].set_ylabel(scans[i], rotation=90, fontsize=12, labelpad=10)
                            ax[1,i].imshow(img[0,i,:,:,16].cpu(),cmap='gray',norm=None)
                            im=ax[1,i].imshow(occ_map[0][i,:,:,16,j].cpu(),cmap=cmap,alpha=0.4,norm=norm)
                            divider2 = make_axes_locatable(ax[1,i])
                            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                            fig.colorbar(im, cax=cax2, ax=ax[1,i],norm=norm)
                    plt.savefig(results_folder+"plots/"+file_name+'_'+''.join(char for char in str(features[j]) if char.isalnum())+'_occ_map.png', dpi=300)                 
                # Show plot while processing
                if config.config()['plot']==True:
                    plt.show()
                    