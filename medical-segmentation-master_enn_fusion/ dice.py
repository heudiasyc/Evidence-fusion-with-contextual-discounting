#!/usr/bin/env python
# coding: utf-8

# In[]:

###########################  IMPORTS   ############################################# 


import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
from monai.networks.nets import UNet_DS_K_P20_t2,UNet_DS_K_P20_t1ce,UNet_DS_K_P20_t1,UNet_DS_K_P20_flair,UNet_DS_K_P20_fusion
from monai.metrics import compute_meandice
from monai.networks.utils import one_hot
from monai.transforms import (
    AsDiscrete,
    Activations,
    AddChanneld,
    AsChannelFirstd,
    Compose,
    LoadNiftid,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    ToTensord,
)

from monai.inferers import sliding_window_inference
#from monai.data import decollate_batch
from monai.visualize import plot_2d_or_3d_image
from monai.data.utils import list_data_collate, worker_init_fn
from monai.metrics import DiceMetric
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import csv
import time
import SimpleITK as sitk
from os.path import splitext,basename
import random
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from copy import copy
import os
import numpy as np
#from math import ceil
from torch.utils.tensorboard import SummaryWriter

from global_tools.tools import display_loading_bar
from class_modalities.transforms import LoadNifti, Roi2Mask, ResampleReshapeAlign, Sitk2Numpy, ConcatModality
from dataset.brats import get_datasets
#from deeplearning_models.model_exp import UNet3D


from dice import EDiceLoss_Val
###########################  RULES   #############################################



#os.environ["CUDA_VISIBLE_DEVICES"]="1"

PREDICTION_VALIDATION_SET = True
#MODALITY = Modality_TRAINING_PET_CT()
path_results = '/home/lab/hualing/brats_hl_medical-segmentation-master_enn_fusion/result'

# generates folders

if not os.path.exists(path_results):
    os.makedirs(path_results)

##################






full_train_dataset, l_val_dataset, bench_dataset = get_datasets(1234, fold_number=0)
train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=4, shuffle=True,
                                               num_workers=4, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=4)
bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=4)



device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(device)

model=UNet_DS_K_P20_fusion().to(device)

#print(model)
##############t1ce###############

#trained_model_path="best_metric_model_segmentation3d_dict_baseline.pth"
trained_model_path_1="/home/lab/hualing/brats_hl_medical-segmentation-master_enn_t1ce/best_metric_model_segmentation3d_dict_enn_t1ce_f.pth"
#trained_model_path_1="best_metric_model_segmentation3d_dict_enn_t1ce_fusion.pth"

model_dict_1 = model.t1ce_modality.state_dict()
pre_dict_1 = torch.load(trained_model_path_1)
pre_dict_1 = {k: v for k, v in pre_dict_1.items() if k in model_dict_1}
model_dict_1.update(pre_dict_1)
model.t1ce_modality.load_state_dict(model_dict_1)
##############t1###############
trained_model_path_2="/home/lab/hualing/brats_hl_medical-segmentation-master_enn_t1/best_metric_model_segmentation3d_dict_enn_t1_f.pth"
#trained_model_path_2="best_metric_model_segmentation3d_dict_enn_t1_fusion.pth"

model_dict_2 = model.t1_modality.state_dict()
pre_dict_2 = torch.load(trained_model_path_2)
pre_dict_2 = {k: v for k, v in pre_dict_2.items() if k in model_dict_2}
model_dict_2.update(pre_dict_2)
model.t1_modality.load_state_dict(model_dict_2)
##############flair###############
trained_model_path_3="/home/lab/hualing/brats_hl_medical-segmentation-master_enn_flair/best_metric_model_segmentation3d_dict_enn_flair_f.pth"
#trained_model_path_3="best_metric_model_segmentation3d_dict_enn_flair_fusion.pth"
model_dict_3 = model.flair_modality.state_dict()
pre_dict_3 = torch.load(trained_model_path_3)
pre_dict_3 = {k: v for k, v in pre_dict_3.items() if k in model_dict_3}
model_dict_3.update(pre_dict_3)
model.flair_modality.load_state_dict(model_dict_3)

###############
trained_model_path_4="/home/lab/hualing/brats_hl_medical-segmentation-master_enn_t2/best_metric_model_segmentation3d_dict_enn_t2.pth"
#trained_model_path_4="best_metric_model_segmentation3d_dict_enn_t2_fusion.pth"

model_dict_4 = model.t2_modality.state_dict()
pre_dict_4 = torch.load(trained_model_path_4)
pre_dict_4 = {k: v for k, v in pre_dict_4.items() if k in model_dict_4}
model_dict_4.update(pre_dict_4)
model.t2_modality.load_state_dict(model_dict_4)
'''


trained_model_path="best_metric_model_segmentation3d_dict_enn_fusion_discounting1.pth"

model_dict = model.state_dict()
pre_dict = torch.load(trained_model_path)
pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
model_dict.update(pre_dict)
model.load_state_dict(model_dict)
'''
params = filter(lambda p: p.requires_grad, model.parameters())
print(params)

for name, param in model.named_parameters():
    #print(name,param)
    if param.requires_grad==True:
        print(name)
#optimizer = torch.optim.Adam(model.parameters(), 1e-3)

# = monai.metrics.DiceMetric( include_background=False,reduction="mean")
optimizer = torch.optim.Adam(params, 1e-4)
#optimizer = torch.optim.Adam([{'params':params_1,},{'params':params_2},{'params':params_3},{'params':params_4}], 1e-4)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=10)
loss_function = monai.losses.DiceLoss(include_background=False,softmax=False,squared_pred=True,to_onehot_y=False)

val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter()
def save_metrics(epoch, metrics, writer, current_epoch, teacher=False,):
    metrics = list(zip(*metrics))
    # print(metrics)
    # TODO check if doing it directly to numpy work
    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    # print(metrics)
    #labels = ("ET", "TC", "WC")
    labels = ("other","ED", "ET", "NRC")
    #labels = ( "ED", "ET", "NRC")
    metrics = {key: value for key, value in zip(labels, metrics)}

    print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    for key, value in metrics.items():
        tag = f"val{'_teacher' if teacher else ''}{''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)

def save_metrics_c(epoch, metrics, writer, current_epoch, teacher=False,):
    metrics = list(zip(*metrics))
    # print(metrics)
    # TODO check if doing it directly to numpy work
    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    # print(metrics)
    labels = ("ET", "TC", "WC")
    #labels = ("other","ED", "ET", "NRC")
    #labels = ( "ED", "ET", "NRC")
    metrics = {key: value for key, value in zip(labels, metrics)}

    print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    for key, value in metrics.items():
        tag = f"val{'_teacher' if teacher else ''}{''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)


"""
for epoch in range(100):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{5}")
    model.train()
    epoch_loss = 0
    step = 0

    roi_size = (128, 128, 128)
    sw_batch_size = 1
    CE_L = torch.nn.BCELoss()
    
    for batch_data in train_loader:
        torch.cuda.empty_cache()
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        inputs,labels=inputs.float(),labels.float()
        input1=inputs[:,0].unsqueeze(1)
        input2 = inputs[:, 1].unsqueeze(1)
        input3 = inputs[:, 2].unsqueeze(1)
        input4 = inputs[:, 3].unsqueeze(1)
        optimizer.zero_grad()
        train_outputs= model(input1,input2,input3,input4)

        test1 = train_outputs[0, :, 60, 60, 60]
        test2 = labels[0, :, 60, 60, 60]

        outputs=train_outputs[:, :4, :, :, :]#+(1/4)*train_outputs[:, 4, :, :, :].unsqueeze(1)
        #bce_loss = CE_L(outputs[:, 1:, :, :, :], labels[:, 1:, :, :, :])
        dice_loss=loss_function(outputs, labels)
        loss = dice_loss#+bce_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(full_train_dataset) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    scheduler.step(epoch_loss)


    if (epoch + 1) % val_interval == 0:
        model.eval()
        metrics3 = []
        metrics3_c = []
        with torch.no_grad():
            dice_metric_c = DiceMetric(include_background=True, reduction="mean")
            ric_c = DiceMetric(include_background=True, reduction="mean")                   dice_metric = DiceMetric(include_background=False, mutually_exclusive=True, reduction="mean")
            criterian_val = EDiceLoss_Val()
            metric_3label = criterian_val.metric


            metric_sum = 0.0
            metric_count = 0
            val_images = None
            val_labels = None
            val_outputs = None
            #metrics3 = []
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_images, val_labels = val_images.float(), val_labels.float()

                outputs = model(val_images[:,0,].unsqueeze(1),val_images[:,1,].unsqueeze(1),val_images[:,2,].unsqueeze(1),val_images[:,3,].unsqueeze(1))

                val_outputs = outputs[:, :4, :, :, :]#+(1/4)*outputs[:, 4, :, :, :].unsqueeze(1)
                test1=val_outputs[0,:,60,60,60]
                test2 = val_labels[0, :, 60, 60, 60]

                
                pred_label = torch.argmax(val_outputs, dim=1, keepdim=True)
                patient_label = torch.argmax(val_labels, dim=1, keepdim=True)
                pred_label = pred_label.squeeze(1).cpu()
                patient_label = patient_label.squeeze(1).cpu()
                et = patient_label == 2
                tc = np.logical_or(patient_label == 2, patient_label == 3)
                wt = np.logical_or(tc, patient_label == 1)
                patient_label = np.stack([et, tc, wt],axis=1)

                et_ = pred_label == 2
                tc_ = np.logical_or(pred_label == 2, pred_label == 3)
                wt_ = np.logical_or(tc_, pred_label == 1)
                pred_label = np.stack([et_, tc_, wt_],axis=1)

                patient_label = torch.from_numpy(patient_label)
                pred_label = torch.from_numpy(pred_label)
                
                value_c = dice_metric_c(y_pred=pred_label, y=patient_label)

                metric_3_c=metric_3label(pred_label,patient_label,False)
                metrics3_c.extend(metric_3_c)
                


                value = dice_metric(y_pred=val_outputs, y=val_labels)

                metric_3=metric_3label(val_outputs,val_labels,True)
                metrics3.extend(metric_3)


                #print('value',value)
                #print('value_c',value)


                not_nans = dice_metric.not_nans.item()
                metric_count += not_nans
                metric_sum += value.item() * not_nans

            metric = metric_sum / metric_count
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_segmentation3d_dict_enn_fusion_discounting1_new.pth")
                print("saved new best metric model")
            print(
                    "current epoch: {} current mean dice: {:.4f}, best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric,best_metric, best_metric_epoch
                )
            )
            writer.add_scalar("val_mean_dice", metric, epoch + 1)

        save_metrics(epoch, metrics3, writer, epoch, False)
        save_metrics_c(epoch, metrics3_c, writer, epoch, False)


        torch.cuda.empty_cache()

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()
"""

#model.compile(loss=loss_object,optimizer=optimizer,metrics=metrics)

# LEARNING PROCEDURE

start_tt = time.time()
labels_names = ['Background','Lymphoma',]
labels_numbers = [0,1]

model.load_state_dict(torch.load("best_metric_model_segmentation3d_dict_enn_fusion_discounting1_new.pth"))
def PREDICT_MASK(data_set_ids, path_predictions, model):

    # generates folder
    if not os.path.exists(path_predictions):
        os.makedirs(path_predictions)

    filenames_predicted_masks = []

    #test_loader = DataLoader(data_set_ids, batch_size=1, num_workers=4, collate_fn=list_data_collate)
    test_loader=bench_loader
    had=monai.metrics.compute_hausdorff_distance()
    dice_metric_c = DiceMetric(include_background=True, reduction="mean")
    dice_metric = monai.metrics.DiceMetric(include_background=False,mutually_exclusive=True,to_onehot_y=False,reduction="mean")
    metric_sum = 0.0
    #loss=0.0
    metric_count=0.0
    # for i,data_set_id in enumerate(data_set_ids):
    metrics3_c = []
    criterian_val = EDiceLoss_Val()
    metric_3label = criterian_val.metric
    for i,test_data in enumerate(test_loader):
        #criterian_val = EDiceLoss_Val()
        #metric_3label = criterian_val.metric
        val_images, val_labels = test_data["image"].to(device), test_data["label"].to(device)
        val_images, val_labels = val_images.float(), val_labels.float()
        #val_labels=val_labels.unsqueeze(1)
        #prediction,feature = model(val_images)
        inputs=val_images
        input1=inputs[:,0].unsqueeze(1)
        input2 = inputs[:, 1].unsqueeze(1)
        input3 = inputs[:, 2].unsqueeze(1)
        input4 = inputs[:, 3].unsqueeze(1)
        #optimizer.zero_grad()
        prediction= model(input1,input2,input3,input4)
        #print(val_labels[0,:,0,0,0])
        #loss = loss_function(prediction, val_labels)
        #print('loss',loss)
        #val_outputs = prediction
        val_outputs = prediction[:, :4, :, :, :]#+(1/4)*prediction[:, 4, :, :, :].unsqueeze(1)
        pm=val_outputs
        value = dice_metric(y_pred=val_outputs, y=val_labels)
        #print('value',value)
        
        pred_label = torch.argmax(val_outputs, dim=1, keepdim=True)
        patient_label = torch.argmax(val_labels, dim=1, keepdim=True)
        pred_label = pred_label.squeeze(1).cpu()
        patient_label = patient_label.squeeze(1).cpu()
        et = patient_label == 2
        tc = np.logical_or(patient_label == 2, patient_label == 3)
        wt = np.logical_or(tc, patient_label == 1)
        patient_label = np.stack([et, tc, wt],axis=1)

        et_ = pred_label == 2
        tc_ = np.logical_or(pred_label == 2, pred_label == 3)
        wt_ = np.logical_or(tc_, pred_label == 1)
        pred_label = np.stack([et_, tc_, wt_],axis=1)

        patient_label = torch.from_numpy(patient_label)
        pred_label = torch.from_numpy(pred_label)
        #hausdorff=had(pred_label,patient_label)
        #print(hausdorff)

        value_c = dice_metric_c(y_pred=pred_label, y=patient_label)

        metric_3_c=metric_3label(pred_label,patient_label,False)
        metrics3_c.extend(metric_3_c)

        #val_outputs=torch.argmax(pm, axis=1)
        #val_outputs=val_outputs.unsqueeze(1)

        value=value_c 
        metric_count += len(value)
        metric_sum += value.item() * len(value)

        prediction = torch.argmax(prediction, axis=1)
        #prediction=prediction.permute(0,3,1,2)# output from a multiclass softmax
        prediction = prediction.squeeze().cpu().numpy()
        
        prediction =np.transpose(prediction,(2,1,0))
        # conversion in unsigned int 8 to store mask with less memory requirement
        mask = np.asarray(prediction, dtype=np.uint8)


        #new_filename = path_predictions + "/pred_" + splitext(basename(test_files[i]["mask_img"]))[0] + '.nii'
        ##filenames_predicted_masks.append(new_filename)
        #sitk.WriteImage(sitk.GetImageFromArray(mask), new_filename)


    metric_dice = metric_sum / metric_count
    save_metrics_c(1, metrics3_c, writer, 1, False)


    print("dice:", metric_dice)


    return filenames_predicted_masks

    ####################################################################################################

def VISUALISATION_MIP_PREDICTION(path_output, data_set_ids, pred_ids, filename=None):
        """
            Generate MIP projection of PET/CT files with its predicted mask

            data_set_ids : [(PET_id_1,CT_id_1),(PET_id_2,CT_id_2)...]
        """

        if filename is None:
            filename = path_output + "/PETCT_MIP_" + time.strftime("%m%d%H%M%S") + ".pdf"
        else:
            filename = path_output + '/' + filename

        # generates folder
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        n_patients = len(data_set_ids)

        transparency = 0.7

        color_CT = plt.cm.gray
        color_PET = plt.cm.plasma
        color_MASK = copy(plt.cm.Greys)
        color_MASK.set_bad('white', 0.0)

        with PdfPages(filename) as pdf:

            ############################ BOXPLOTS GENERATION #############################################

            # get boxplots
            num_class = len(labels_numbers)
            # initialize values board
            Conf = np.zeros((num_class, num_class), dtype=np.int32)
            TruePos = []
            GTsum = []
            PRsum = []

            TruePos, GTsum, PRsum = compute_metrics(data_set_ids, pred_ids)

            accuracy_TAB = TruePos / PRsum
            dice_TAB = (2 * TruePos + 0.1) / (GTsum + PRsum + 0.1)

            f = plt.figure(figsize=(15, 10))
            f.suptitle('Metrics evaluation', fontsize=15)

            plt.subplot(121)
            plt.boxplot(accuracy_TAB[:, :], sym='x', whis=5, labels=labels_names)
            accuracy_median_tumor = np.mean(accuracy_TAB[:, 1])
            plt.ylim(0.1)
            plt.title("Accuracy Boxplot : tumor=%5.3f" % accuracy_median_tumor, fontsize=15)

            plt.subplot(122)
            plt.boxplot(dice_TAB[:, :], sym='x', whis=5, labels=labels_names)
            dice_median_tumor = np.mean(dice_TAB[:, 1])
            plt.ylim(0.1)
            plt.title("Dice Boxplot : tumor=%5.3f" % dice_median_tumor, fontsize=15)

            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            ############################# MIP GENERATION ################################################

            # loop on files to get MIP visualisation
            for i, (DataSet_id, Pred_id) in enumerate(zip(data_set_ids, pred_ids)):
                display_loading_bar(iteration=i, length=n_patients, add_char=basename(DataSet_id["pet_img"]) + '    ')

                # load imgs
                PET_id= DataSet_id["pet_img"]
                CT_id = DataSet_id["pet_img"]
                Mask_id = DataSet_id["mask_img"]
                PET_scan = sitk.GetArrayFromImage(sitk.ReadImage(PET_id))
                CT_scan = sitk.GetArrayFromImage(sitk.ReadImage(CT_id))
                MASK = sitk.GetArrayFromImage(sitk.ReadImage(Mask_id))
                PRED = sitk.GetArrayFromImage(sitk.ReadImage(Pred_id))

                # for TEP visualisation
                PET_scan = np.where(PET_scan > 1.0, 1.0, PET_scan)
                PET_scan = np.where(PET_scan < 0.0, 0.0, PET_scan)

                # for CT visualisation
                CT_scan = np.where(CT_scan > 1.0, 1.0, CT_scan)
                CT_scan = np.where(CT_scan < 0.0, 0.0, CT_scan)

                # for correct visualisation
                PET_scan = np.flip(PET_scan, axis=0)
                CT_scan = np.flip(CT_scan, axis=0)
                MASK = np.flip(MASK, axis=0)
                PRED = np.flip(PRED, axis=0)

                # stacked projections
                PET_scan = np.hstack((np.amax(PET_scan, axis=1), np.amax(PET_scan, axis=2)))
                CT_scan = np.hstack((np.amax(CT_scan, axis=1), np.amax(CT_scan, axis=2)))
                MASK = np.hstack((np.amax(MASK, axis=1), np.amax(MASK, axis=2)))
                PRED = np.hstack((np.amax(PRED, axis=1), np.amax(PRED, axis=2)))

                ##################################################################
                f = plt.figure(figsize=(15, 10))
                f.suptitle(splitext(basename(PET_id))[0], fontsize=15)

                plt.subplot(121)
                plt.imshow(CT_scan, cmap=color_CT, origin='lower')
                plt.imshow(PET_scan, cmap=color_PET, alpha=transparency, origin='lower')
                plt.imshow(np.where(MASK, 0, np.nan), cmap=color_MASK, origin='lower')
                plt.axis('off')
                plt.title('Ground Truth', fontsize=20)

                plt.subplot(122)
                plt.imshow(CT_scan, cmap=color_CT, origin='lower')
                plt.imshow(PET_scan, cmap=color_PET, alpha=transparency, origin='lower')
                plt.imshow(np.where(PRED, 0, np.nan), cmap=color_MASK, origin='lower')
                plt.axis('off')
                plt.title('Prediction', fontsize=20)

                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                ##################################################################

def compute_metrics(data_set_ids, pred_ids):
        """ Called by VISUALISATION_MIP_PREDICTION """

        num_class = len(labels_names)

        TruePositive = []
        Sum_GroundTruth = []
        Sum_Prediction = []

        for data_set, pred_id in zip(data_set_ids, pred_ids):
            mask_id=data_set["mask_img"]
            ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(mask_id, sitk.sitkUInt8))
            prediction = sitk.GetArrayFromImage(sitk.ReadImage(pred_id, sitk.sitkUInt8))

            TP = []
            GT = []
            PR = []

            TP_mask = ground_truth == prediction

            for label in labels_numbers:
                label_mask = ground_truth == label

                TP.append(np.sum(np.logical_and(TP_mask, label_mask)))  # predict_true, label_true
                GT.append(np.sum(label_mask))  # label_true
                PR.append(np.sum(prediction == label))  # all_true(lymphona, background)

            TruePositive.append(TP)
            Sum_GroundTruth.append(GT)
            Sum_Prediction.append(PR)

        # final conversion of board of values
        TruePositive = np.stack(TruePositive)
        Sum_GroundTruth = np.stack(Sum_GroundTruth)
        Sum_Prediction = np.stack(Sum_Prediction)

        return TruePositive, Sum_GroundTruth, Sum_Prediction

###########################  VISUALISATION   #############################################

if PREDICTION_VALIDATION_SET:
    print("Prediction on validation set :")
    # use to fine tune and evaluate model performances

    filename = "/RESULTS_valid_set_" + time.strftime("%m%d%H%M%S") + ".pdf"

    print("Generating predictions :")
    test_prediction_ids = PREDICT_MASK(data_set_ids=l_val_dataset,
                                                 path_predictions=path_results + '/valid_predictions',
                                                 model=model)
    print("yes")


