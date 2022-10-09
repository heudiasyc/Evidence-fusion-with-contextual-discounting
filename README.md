# MedicalSegmentation



Code for 

### Model
The model used during this project is a custom U-Net [1], adapted to handle 3D medical images, reduce overfitting and limit the RAM comsumption. The model V-Net [2] is also implemented but couldn't be used, as it is more elaborate and requires more RAM, which wasn't possible.

[1] O. Ronneberger, P. Fischer, and T. Brox, ‘U-Net: Convolutional Networks for Biomedical Image Segmentation’, arXiv:1505.04597 [cs], May 2015.

[2]	F. Milletari, N. Navab, and S.-A. Ahmadi, ‘V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation’, arXiv:1606.04797 [cs], Jun. 2016.

### Applications & Results
- **Tissue segmentation of CT scan** in 5 classes: ```Background / Fat / Soft tissues / Lungs / Bones```.

  **results**: no signs of overfitting, visually correct, median DSC > 0.9
  
- **Physiologic segmentation of IRM** in 9 classes: ```N/A / Spleen / Liver / 6 lymphatic nodes```.

  **results**: signs of overfitting, visually correct on Liver and Spleen, wrong for lymphatic nodes, median DSC < 0.4
  
- **Tumour segmentation of PET** scan.

  **results**: signs of slight overfitting, visually acceptable, median DSC > 0.65

- **Tumour segmentation of PET/CT** scan.

  **results**: no signs of overfitting, visually correct, median DSC > 0.74
  

### Segmentation of tumour on PET/CT scan

Trained model available: ```/master/deeplearning_models/trained_model_09241142.h5```

<p align="center">
<img style="display: block; margin: auto;" alt="photo" src="./GIF_example_segmentation.gif">
</p>

### Development progress

#### DONE
- modality PET-CT available ```/master/class_modalities/modality_PETCT.py```

#### ONGOING
- fix image shifting between PET and CT

#### TODO
- add  modality CT
- add modality IRM
- add modality PET
