
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix,cohen_kappa_score
import numpy as np
import statistics  

def sensivity_specifity_cutoff(y_true, y_score):
    #Reused from: https://gist.github.com/twolodzko/4fae2980a1f15f8682d243808e5859bb
    '''Find data-driven cut-off for classification
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels.
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).    
    References
    ----------
    1. Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.
    2. Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    3. Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    '''
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx], fpr, tpr, idx         

def feature_specific_metrics(pred, target, features):
        results_dictionary = dict.fromkeys(features, "nan")
        n=0
        avg_feature_rocauc=[]
        for f in features:
            y_true = [el[n] for el in target]
            y_pred = [el[n] for el in pred]
            threshold, fpr, tpr, idx   = sensivity_specifity_cutoff(y_true, y_pred)
            y_pred_youden = np.array(y_pred > threshold, dtype=float)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_youden,labels=[0,1]).ravel()
            num_samples = tn + fp + fn + tp
            npv = tn / (tn + fn)
            specificity = tn / (tn + fp)
            f1 = 2*tp/(2*tp+fp+fn)
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
         
            cohen_kappa=cohen_kappa_score(y_true, y_pred_youden)

            try:
                roc_auc_sc=roc_auc_score(y_true, y_pred)

            except ValueError:
                print('Only one class present in y_true. ROC AUC score is not defined in that case.')
                roc_auc_sc=0
                pass

            avg_feature_rocauc.append(roc_auc_sc)
            results_dictionary[f] = {   'predictions': y_pred,
                                'Youden-corrected threshold': threshold, 
                                'Youden-corrected predictions': y_pred_youden,
                                'True labels': y_true,
                                'True positives': tp,
                                'True negatives': tn,                                
                                'False positives': fp,
                                'False negatives': fn,
                                'num_samples':num_samples,

                                'PPV': precision,
                                'NPV': npv,
                                'Sensitivity': recall,
                                'Specificity': specificity,
                                'f1': f1,

                                "Cohen's Kappa" : cohen_kappa,
                                'ROC_AUC': roc_auc_sc,
                                
                                'fpr_dict': fpr,
                                'tpr_dict': tpr,
                                'cutoff_index':idx, 
                                 }                  
            n+=1
        avg_feature_rocauc=sum(avg_feature_rocauc)/len(features)   
        return((results_dictionary,avg_feature_rocauc)) 

def test_feature_specific_metrics(cases, pred, target, target_novice, features, youden_dictionary, lesion_class_list): 
        test_results_dictionary = dict.fromkeys(features, "nan")

        avg_feature_rocauc=[]
        avg_feature_kappa=[]
        avg_feature_kappa2=[]
        avg_feature_kappa3=[]

        avg_feature_ppv=[]
        avg_feature_npv=[]  
        avg_feature_recall=[]
        avg_feature_specificity=[]
        avg_feature_f1=[]
        num_lesions_list=[]

        fnh = [el[0] for el in lesion_class_list]
        hcc = [el[1] for el in lesion_class_list]
        met = [el[2] for el in lesion_class_list]
        other = [el[3] for el in lesion_class_list]

        n=0
        for f in features:
              case_names = cases
                          
              y_true = [el[n] for el in target]
              y_pred = [el[n] for el in pred]

              threshold_test, fpr, tpr, idx_test   = sensivity_specifity_cutoff(y_true, y_pred)
              
              fnh_num=0
              hcc_num=0
              met_num=0
              other_num=0
              
              k=0
              for l in y_true:
                if l==1 and fnh[k]==1:
                  fnh_num+=1
                elif l==1 and hcc[k]==1:
                  hcc_num+=1
                elif l==1 and met[k]==1:
                  met_num+=1
                elif l==1 and other[k]==1:
                  other_num+=1
                k+=1


              threshold= youden_dictionary[f]['Youden-corrected threshold']
              idx= youden_dictionary[f]['Validation cutoff point']

              y_pred_youden = np.array(y_pred > threshold, dtype=float)
              
              tn, fp, fn, tp = confusion_matrix(y_true, y_pred_youden,labels=[0,1]).ravel()
              num_samples = tn + fp + fn + tp
              num_lesions = tp+fn

              npv = tn / (tn + fn)
            
              specificity = tn / (tn + fp)
              f1 = 2*tp/(2*tp+fp+fn)
              precision = tp/(tp+fp)
              recall = tp/(tp+fn)
              
              y_true_novice = [el[n] for el in target_novice]

              cohen_kappa_novice=cohen_kappa_score(y_true_novice, y_pred_youden)
              cohen_kappa_novice_v_expert=cohen_kappa_score(y_true_novice, y_true)
              cohen_kappa=cohen_kappa_score(y_true, y_pred_youden)
              
              try:
                roc_auc_sc=roc_auc_score(y_true, y_pred)
              except ValueError:
                print('Only one class present in y_true. ROC AUC score is not defined in that case.')
                roc_auc_sc=404
                pass

              avg_feature_rocauc.append(roc_auc_sc)
              avg_feature_kappa.append(cohen_kappa)
              avg_feature_kappa2.append(cohen_kappa_novice)
              avg_feature_kappa3.append(cohen_kappa_novice_v_expert)

              avg_feature_ppv.append(precision)
              avg_feature_npv.append(npv)
              avg_feature_recall.append(recall)
              avg_feature_specificity.append(specificity)  
              avg_feature_f1.append(f1)
              num_lesions_list.append(num_lesions)

              test_results_dictionary[f] = {   
                                  'cases': case_names,
                                  'predictions': y_pred,
                                  'Youden-corrected threshold': threshold, 
                                  'Youden-corrected predictions': y_pred_youden,
                                  'True labels': y_true,
                                  'True positives': tp,
                                  'True negatives': tn,                                
                                  'False positives': fp,
                                  'False negatives': fn,
                                  'n_cases': num_lesions,
                                  'n_controls':num_samples-num_lesions,
                                  'FNH (%)': 100*fnh_num/num_lesions,
                                  'HCC (%)': 100*hcc_num/num_lesions,
                                  'MET (%)': 100*met_num/num_lesions,
                                  'Other (%)': 100*other_num/num_lesions,

                                  'PPV': precision,
                                  'NPV': npv,
                                  'Sensitivity': recall,
                                  'Specificity': specificity,
                                  'f1': f1,
                                  
                                  "Cohen's Kappa: Model vs. Expert" : cohen_kappa,
                                  "Cohen's Kappa: Model vs. Novice" : cohen_kappa_novice,
                                  "Cohen's Kappa: Novice vs. Expert" : cohen_kappa_novice_v_expert,

                                  'ROC_AUC': roc_auc_sc,

                                  'fpr_dict': fpr,
                                  'tpr_dict': tpr, 
                           
                                 }                  
              n+=1
      

        test_results_dictionary['Mean values']={     
                                'Number of lesions': statistics.mean(num_lesions_list),                
                                'PPV': statistics.mean(avg_feature_ppv),
                                'NPV': statistics.mean(avg_feature_npv),
                                'Sensitivity': statistics.mean(avg_feature_recall),
                                'Specificity': statistics.mean(avg_feature_specificity),
                                'f1': statistics.mean(avg_feature_f1),
                                "Cohen's Kappa: Model vs. Expert" : statistics.mean(avg_feature_kappa),
                                "Cohen's Kappa: Model vs. Novice" : statistics.mean(avg_feature_kappa2),
                                "Cohen's Kappa: Novice vs. Expert" : statistics.mean(avg_feature_kappa3),

                                'ROC_AUC': statistics.mean(avg_feature_rocauc),
                                 } 
                      
        test_results_dictionary['SD values']={     
                                'Number of lesions': statistics.pstdev(num_lesions_list),      
                    
                                'PPV': statistics.pstdev(avg_feature_ppv),
                                'NPV': statistics.pstdev(avg_feature_npv),
                                'Sensitivity': statistics.pstdev(avg_feature_recall),
                                'Specificity': statistics.pstdev(avg_feature_specificity),
                                'f1': statistics.pstdev(avg_feature_f1),
                                "Cohen's Kappa: Model vs. Expert" : statistics.pstdev(avg_feature_kappa),
                                "Cohen's Kappa: Model vs. Novice" : statistics.pstdev(avg_feature_kappa2),
                                "Cohen's Kappa: Novice vs. Expert" : statistics.pstdev(avg_feature_kappa3),

                                'ROC_AUC': statistics.pstdev(avg_feature_rocauc),
                                 } 
        return((test_results_dictionary,avg_feature_rocauc)) 

def test_feature_specific_metrics_single_case(cases, pred, target, features, youden_dictionary):   
        test_results_dictionary = dict.fromkeys(features, "nan")
        n=0
        for f in features:
            case_names = cases
            y_true = [el[n] for el in target]
            y_pred = [el[n] for el in pred]
            threshold = youden_dictionary[f]['Youden-corrected threshold']
            y_pred_youden = np.array(y_pred > threshold, dtype=float)

            test_results_dictionary[f] = {   
                                'cases': case_names,
                                'predictions': y_pred,
                                'Youden-corrected threshold': threshold, 
                                'Youden-corrected predictions': y_pred_youden,
                                'True labels': y_true,
                                 }                  
            n+=1
        return(test_results_dictionary) 
