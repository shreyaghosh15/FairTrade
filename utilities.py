
import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

def find_ate_score(protected_attr,labels,predictions):
        protected_pos = 0.
        protected_neg = 0.
        non_protected_pos = 0.
        non_protected_neg = 0.

        saValue = 0
        for i in range(len(protected_attr)):
            # protected population
            if protected_attr[i] == saValue:
                if predictions[i] == 1:
                    protected_pos += 1.
                else:
                    protected_neg += 1.
                # correctly classified
                

            else:
                if predictions[i] == 1:
                    non_protected_pos += 1.
                else:
                    non_protected_neg += 1.

            
        if((protected_pos + protected_neg) == 0):
            C_prot = 0
        else:
            C_prot = (protected_pos) / (protected_pos + protected_neg)
        if((non_protected_pos + non_protected_neg) == 0):
            C_non_prot =0
        else:
            C_non_prot = (non_protected_pos) / (non_protected_pos + non_protected_neg)

        stat_par = C_non_prot - C_prot
        return stat_par

def find_ate(predictions,protected_attr,saValue): #H is client_data
    protected_pos = 0.
    protected_neg = 0.
    non_protected_pos = 0.
    non_protected_neg = 0.


    for i in range(len(protected_attr)):
            # protected population
            if protected_attr[i] == saValue:
                if predictions[i] == 1:
                    protected_pos += 1.
                else:
                    protected_neg += 1.
                

            else:
                if predictions[i] == 1:
                    non_protected_pos += 1.
                else:
                    non_protected_neg += 1.
                    
    if((protected_pos + protected_neg) == 0):
            C_prot = 0
    else:
            C_prot = (protected_pos) / (protected_pos + protected_neg)
    if((non_protected_pos + non_protected_neg) == 0):
            C_non_prot =0
    else:
            C_non_prot = (non_protected_pos) / (non_protected_pos + non_protected_neg)

    stat_par = C_non_prot - C_prot
    return stat_par


def find_ate_3(predictions,potential_outcome,sex_list): #H is client_data
    diff = predictions-potential_outcome
    sex_list = torch.tensor(sex_list, dtype=torch.float32)
    ate_list = ((sex_list*diff)-(1-sex_list)*diff)
    ate = ate_list.mean(dim=0)
    return ate

def find_statistical_parity_score(protected_attr,labels,predictions):
        protected_pos = 0.
        protected_neg = 0.
        non_protected_pos = 0.
        non_protected_neg = 0.

        saValue = 0
        for i in range(len(protected_attr)):
            # protected population
            if protected_attr[i] == saValue:
                if predictions[i] == 1:
                    protected_pos += 1.
                else:
                    protected_neg += 1.
                # correctly classified
                

            else:
                if predictions[i] == 1:
                    non_protected_pos += 1.
                else:
                    non_protected_neg += 1.

            
        if((protected_pos + protected_neg) == 0):
            C_prot = 0
        else:
            C_prot = (protected_pos) / (protected_pos + protected_neg)
        if((non_protected_pos + non_protected_neg) == 0):
            C_non_prot =0
        else:
            C_non_prot = (non_protected_pos) / (non_protected_pos + non_protected_neg)

        stat_par = C_non_prot - C_prot
        return stat_par

# def positive_prediction_rate(predictions):
#     predictions = np.asarray(predictions, dtype=float)
#     if predictions.size == 0:
#         return np.nan
#     return float(predictions.mean())

def positive_prediction_rate(predictions, mask):
    values = np.asarray(predictions, dtype=float)[np.asarray(mask)]
    if values.size == 0:
        return np.nan
    return float(values.mean())

def spd_from_masks(predictions, protected_mask, reference_mask):
    protected_rate = positive_prediction_rate(predictions, protected_mask)
    reference_rate = positive_prediction_rate(predictions, reference_mask)
    return reference_rate - protected_rate

def build_intersectional_spd_rows(x_test_df, predictions, sex_mapping, race_mapping):
    predictions = np.asarray(predictions, dtype=int)

    sex = x_test_df['sex'].astype(int).to_numpy()
    race = x_test_df['race'].astype(int).to_numpy()

    female_value = sex_mapping['Female']
    male_value = sex_mapping['Male']
    white_value = race_mapping['White']

    female_mask = sex == female_value
    male_mask = sex == male_value
    white_mask = race == white_value
    non_white_mask = race != white_value
    white_male_mask = white_mask & male_mask
    white_female_mask = white_mask & female_mask
    non_white_male_mask = non_white_mask & male_mask
    non_white_female_mask = non_white_mask & female_mask

    rows = [
        {
            'group_type': 'attribute',
            'group_name': 'Gender: Female vs Male',
            'reference_group': 'Male',
            'group_positive_rate': positive_prediction_rate(predictions, female_mask),
            'reference_positive_rate': positive_prediction_rate(predictions, male_mask),
            'spd': spd_from_masks(predictions, female_mask, male_mask),
        },
        {
            'group_type': 'attribute',
            'group_name': 'Race: Non-White vs White',
            'reference_group': 'White',
            'group_positive_rate': positive_prediction_rate(predictions, non_white_mask),
            'reference_positive_rate': positive_prediction_rate(predictions, white_mask),
            'spd': spd_from_masks(predictions, non_white_mask, white_mask),
        },
        {
            'group_type': 'intersection',
            'group_name': 'White Male',
            'reference_group': 'White Male',
            'group_positive_rate': positive_prediction_rate(predictions, white_male_mask),
            'reference_positive_rate': positive_prediction_rate(predictions, white_male_mask),
            'spd': 0.0,
        },
        {
            'group_type': 'intersection',
            'group_name': 'White Female',
            'reference_group': 'White Male',
            'group_positive_rate': positive_prediction_rate(predictions, white_female_mask),
            'reference_positive_rate': positive_prediction_rate(predictions, white_male_mask),
            'spd': spd_from_masks(predictions, white_female_mask, white_male_mask),
        },
        {
            'group_type': 'intersection',
            'group_name': 'Non-White Male',
            'reference_group': 'White Male',
            'group_positive_rate': positive_prediction_rate(predictions, non_white_male_mask),
            'reference_positive_rate': positive_prediction_rate(predictions, white_male_mask),
            'spd': spd_from_masks(predictions, non_white_male_mask, white_male_mask),
        },
        {
            'group_type': 'intersection',
            'group_name': 'Non-White Female',
            'reference_group': 'White Male',
            'group_positive_rate': positive_prediction_rate(predictions, non_white_female_mask),
            'reference_positive_rate': positive_prediction_rate(predictions, white_male_mask),
            'spd': spd_from_masks(predictions, non_white_female_mask, white_male_mask),
        },
    ]

    return rows

    # def make_row(group_type, group_name, group_mask, reference_name, reference_mask):
    #     group_rate = positive_prediction_rate(predictions[group_mask])
    #     reference_rate = positive_prediction_rate(predictions[reference_mask])
    #     spd = group_rate - reference_rate

    #     return {
    #         'group_type': group_type,
    #         'group_name': group_name,
    #         'reference_group': reference_name,
    #         'group_positive_rate': group_rate,
    #         'reference_positive_rate': reference_rate,
    #         'spd': spd,
    #         'abs_spd': abs(spd),
    #     }

    # rows = []
    # rows.append(make_row('attribute', 'Female', female_mask, 'Male', male_mask))
    # rows.append(make_row('attribute', 'Non-White', nonwhite_mask, 'White', white_mask))
    # rows.append(make_row('intersection', 'White Male', white_male_mask, 'White Male', white_male_mask))
    # rows.append(make_row('intersection', 'White Female', white_mask & female_mask, 'White Male', white_male_mask))
    # rows.append(make_row('intersection', 'Non-White Male', nonwhite_mask & male_mask, 'White Male', white_male_mask))
    # rows.append(make_row('intersection', 'Non-White Female', nonwhite_mask & female_mask, 'White Male', white_male_mask))

    return rows

def find_eqop_score(self,data,labels,predictions):
        protected_pos = 0.
        protected_neg = 0.
        non_protected_pos = 0.
        non_protected_neg = 0.

        tp_protected = 0.
        tn_protected = 0.
        fp_protected = 0.
        fn_protected = 0.

        tp_non_protected = 0.
        tn_non_protected = 0.
        fp_non_protected = 0.
        fn_non_protected = 0.
        saIndex = 2
        saValue = 0
        for idx, val in enumerate(data):
            # protrcted population
            if val[saIndex] == saValue:
                if predictions[idx] == 1:
                    protected_pos += 1.
                else:
                    protected_neg += 1.

                # correctly classified
                if labels[idx] == predictions[idx]:
                    if labels[idx] == 1:
                        tp_protected += 1.
                    else:
                        tn_protected += 1.
                # misclassified
                else:
                    if labels[idx] == 1:
                        fn_protected += 1.
                    else:
                        fp_protected += 1.

            else:
                if predictions[idx] == 1:
                    non_protected_pos += 1.
                else:
                    non_protected_neg += 1.

                # correctly classified
                if labels[idx] == predictions[idx]:
                    if labels[idx] == 1:
                        tp_non_protected += 1.
                    else:
                        tn_non_protected += 1.
                # misclassified
                else:
                    if labels[idx] == 1:
                        fn_non_protected += 1.
                    else:
                        fp_non_protected += 1.

        if((tp_protected + fn_protected)==0):
            tpr_protected = 0
        else:
            tpr_protected = tp_protected / (tp_protected + fn_protected)
        #tnr_protected = tn_protected / (tn_protected + fp_protected)

        if((tp_non_protected + fn_non_protected) == 0):
            tpr_non_protected=0
        else:
            tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
        #tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)


        eqop = tpr_non_protected - tpr_protected
        return eqop
 


def find_ate_2(y_potential, predictions, protected_attr):

        tp_protected = 0.
        tn_protected = 0.
        fp_protected = 0.
        fn_protected = 0.

        tp_non_protected = 0.
        tn_non_protected = 0.
        fp_non_protected = 0.
        fn_non_protected = 0.
        
        saValue = 0
        for idx in range(len(protected_attr)):
            # protrcted population
            if protected_attr[idx] == saValue:

                # correctly classified
                if y_potential[idx] == predictions[idx]:
                    if y_potential[idx] == 1:
                        tp_protected += 1.
                    else:
                        tn_protected += 1.
      
                else:
                    if y_potential[idx] == 1:
                        fn_protected += 1.
                    else:
                        fp_protected += 1.

            else:

                # correctly classified
                if y_potential[idx] == predictions[idx]:
                    if y_potential[idx] == 1:
                        tp_non_protected += 1.
                    else:
                        tn_non_protected += 1.
                # misclassified
                else:
                    if y_potential[idx] == 1:
                        fn_non_protected += 1.
                    else:
                        fp_non_protected += 1.

        if((tp_protected + fn_protected)==0):
            tpr_protected = 0
        else:
            tpr_protected = tp_protected / (tp_protected + fn_protected)
        #tnr_protected = tn_protected / (tn_protected + fp_protected)

        if((tp_non_protected + fn_non_protected) == 0):
            tpr_non_protected=0
        else:
            tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
        #tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)


        eqop = tpr_non_protected - tpr_protected
        return eqop


# Evaluation of the global model on the test data
def all_metrics(y_true,y_pre):
        #print(y_true)
        #p#rint(y_pre)
        conf = (confusion_matrix(y_true,y_pre.round()))
        TN = conf[0][0]
        FP = conf[0][1]
        FN = conf[1][0]
        TP = conf[1][1]
        #print(TN)
        #print(FP)
        #print(FN)
        #print(TP)
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        BalanceACC = (sensitivity+specificity)/2
        G_mean = math.sqrt(sensitivity*specificity)
        FN_rate= FN/(FN+TP)
        FP_rate = FP/(FP+TN)
        Precision = TP/(TP+FP)
        f1_sc = 2*(sensitivity * Precision) / (sensitivity + Precision)
        acc = (TP+TN)/(TP+TN+FN+FP)
        auc = roc_auc_score(y_true, y_pre)
        return sensitivity,specificity,BalanceACC,G_mean,FN_rate,FP_rate,Precision,f1_sc,acc, auc
    
    
def find_class_weights(labels,majority_label=0,minority_label=1):
        unique, counts = np.unique(labels, return_counts=True)
        count_ap_dict = dict(zip(unique, counts))

        majority_class_weight = 1
        minority_class_weight = count_ap_dict.get(majority_label,0)/count_ap_dict.get(minority_label,1)
        #class_weights={majority_label:1,minority_label:minority_class_weight}
        #class_weights = [1,minority_class_weight]
        class_weights = []
        for i in range(len(labels)):
            if labels[i]==minority_label:
                class_weights.append(minority_class_weight)
            else:
                class_weights.append(1)
        t = torch.tensor(class_weights)
        return t
