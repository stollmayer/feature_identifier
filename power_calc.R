# Sample size and power computation for ROC curves, based on:
# https://www.rdocumentation.org/packages/pROC/versions/1.18.0/topics/power.roc.test

library(irr)
library(pROC)

data <- read.csv(file = './results_folder/test_feature_metrics_all.csv')

# One ROC curve with a given AUC n_cases and power calculation:

# As a reminder: 
# significance level = P(Type I error) = probability of finding an effect that is not there
# power = 1 - P (Type II error) = probability of finding an effect that is there
# power above 80% is considered accepatble

features = data$X
cases = data$'n_cases'
controls = data$'n_controls'
aucs = data$'ROC_AUC'

n=1
range <- 1:10
for (i in range) {
    case = cases[n]
    control = controls[n]
    auc = aucs[n]
    roc_test = power.roc.test(ncases=case ,ncontrols=control, auc=auc, method='obuchowski')
    print(features[n])
    print(roc_test)
    n = n + 1
}
