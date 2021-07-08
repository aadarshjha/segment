
# Instance Segmentation for Whole Slide Imaging: End-to-End or Detect-Then-Segment

This repository contains both code and data relating to the experimentation performed to better understand effective segmentation methodologies utilizing deep learning techniques that improve glomeruli characterization on high-resolution Whole Slide Imaging.

### To Apply Our Model

1. Download the model and a test image from: [Google Drive](https://drive.google.com/drive/folders/123qruJKe8pXGy48M91lqPZW0_xhl1w3U?usp=sharing)
2. Download the test image, provided in Google Drive. 
3. Navigate to `segment/automatic-detection/deep-prediction`
4. Set up the enviornment as described below
5. `!python train.py`
6. Evaluate results in `../dsc-evaluation` via `!python train.py`

Please let me know if you have any questions @ aadarsh.jha@vanderbilt.edu 

### Summary:

This research project comprehensively analyzes several factors relating to semantic segmentation (image resolution, color space, and segmentation backbones), as well as proposes and compares a "detect-then-segment" framework against current conventional end-to-end segmentation ("Mask-RCNN") methods utilized in high-resolution WSI.


### General Methodology

Our project may be considered in two different phases:

1.  *Segmentation on Manual Detection Results*

* In this phase, we comprehensively analyze conventional semantic segmentation upon manually detected glomeruli images. Namely, we discover the effect of six distinct image resolutions, two segmentation backbones, and 2 color spaces.

2.  *Segmentation on Automatic Detection Results*

* In this phase, we directly evaluate the performance of our detect-then-segment approach relative to a standard Mask-RCNN implementation.

  

### Usage and Data:



#### Project Structure:

```
.
├── README.md
├── automatic-detection
│   ├── data
│   ├── deep-prediction
│   └── dsc-evaluation
│       ├── deeplab-evaluation
│       └── mask-rcnn-evaluation
└── manual-detection
    ├── data
    ├── dsc-evaluation
    ├── prediction-pipeline
    │   ├── deep-prediction
    │   └── u-net-prediction
    └── segmentation-pipeline
        ├── deeplab-segmentation
        └── u-net-segmentation
```

#### Data
1.  *Manual Detection Phase*
	* For this phase, manually detected glomeruli in 512x512 is provided, as well as a U-Net model (.pth) file for the 128x128 resolution. Further, ground truth masks are provided for DSC evaluation.
	
2.  *Automatic Detection Phase*
	* The data provided includes the original resolution glomeruli data (> 1000x1000), manually traced corresponding masks, and a DeepLab_v3 model file.

#### Usage
> Detailed Instructors are also located in each subfolder of this repository. 
1.  *Manual Detection Phase*
	* We begin by: 
		* `cd` into `./manual-detection`
		* The `segmentation-pipeline/` folder contains both U-Net and DeepLab_v3 code to produce respective model (.pth) files to perform and remember segmentation results upon a given input set. You may use this to perform your own segmentation. 
		* The `prediction-pipeline` folder takes a set of input images and a model file to produce predicted images. 
			* With the images provided in the `data/` folder, as well as the model file, you may use this prediction pipeline (namely, the `u-net-prediction/` folder) to reproduce the images used in the experiment. 
		* Finally, to evaluate the performance of any segmentation model, simply use the `dsc-evaluation/general-evaluation/` folder and input both ground truth and predicted masks to obtain a CSV file of data. Average the DSC values for each photo to obtain the results presented in this experiment (a value of `0.940` should be obtained). 
	
2.  *Automatic Detection Phase*
	*	We begin by: 
		*	`cd` into `./automatic-detection`
		*	Then, the `deep-prediction` folder contains code that will allow for the production of predicted images based on input images and a model file You may use the input images and the model file provided. 
		*	With the model file, `cd` to `dsc-evaluation/`. Two subfolders are present here: 
			*	`deeplab-evaluation/` will be used to evaluate DeepLab_v3 predicted images against the ground truth masks.
			*	`mask-rcnn-evaluation/` will be used to evaluate MaskRCNN predicted images against the ground truth masks.
			*	We draw a distinction between these two types of evaluation due to differing image representation (e.g., .png vs .jpg). 
        *   You should be able to obtain a DSC of `0.953` from DeepLab_v3, and `0.902` from Mask-RCNN. 


### General Notes
* If using Colab, run the following set-up script: `!pip uninstall imgaug && pip install git+https://github.com/aleju/imgaug.git && pip install -U PyYAML && pip install tensorboardx`
* Use `yaml` config files to properly alter the variables in the experiment: color space and resolution. 
    
### Results:
Our findings show that our detect-then-segment pipeline, with the DeepLab_v3 segmentation framework operating on a previously detected glomeruli of 512x512 resolution, achieved a 0.953 dice similarity coefficient (DSC), compared with a 0.902 DSC from the end-to-end Mask-RCNN pipeline.


### See Also:

[Mask-RCNN Implementation](https://github.com/facebookresearch/maskrcnn-benchmark) <br>
[U-Net Implementation](https://github.com/milesial/Pytorch-UNet) <br>
[DeepLab_v3 Implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py) <br>
