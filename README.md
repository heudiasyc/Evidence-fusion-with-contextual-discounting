Code for  paper MICCAI2022 paper "Evidence fusion with contextual discounting for multi-modality medical image segmentation".

```bash
We propose a new deep framework allowing us to merge multi-MRI image segmentation results using the formalism of Dempster-Shafer theory while taking into account the reliability of different modalities relative to different classes.
```
###
Environment requirement: 
```bash
Before using the code, please install the required packages according to the instructions( refer to https://github.com/iWeisskohl/Evidential-neural-network-for-lymphoma-segmentation )
```
###
Models:
```bash
Copy the models from net  into ./monai/networks/nets
```
###
Pre-Trained weights of ES module for flair, t1, t1Gd and t2 are located in ./model_single_modality


 Training:  ./medical-segmentation-master_enn_fusion
 ```bash
python TRAINING_unet_enn.py
```

###########Citing this paper #############
```bash
@inproceedings{huang2022evidence,
  title={Evidence fusion with contextual discounting for multi-modality medical image segmentation},
  author={Huang, Ling and Denoeux, Thierry and Vera, Pierre and Ruan, Su},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={401--411},
  year={2022},
  organization={Springer}

}
```
