# Usage

2.  *Automatic Detection Phase*
	*	We begin by: 
		*	`cd` into `./automatic-detection`
		*	Then, the `deep-prediction` folder contains code that will allow for the production of predicted images based on input images and a model file You may use the input images and the model file provided. 
		*	With the model file, `cd` to `dsc-evaluation/`. Two subfolders are present here: 
			*	`deeplab-evaluation/` will be used to evaluate DeepLab_v3 predicted images against the ground truth masks.
			*	`mask-rcnn-evaluation/` will be used to evaluate MaskRCNN predicted images against the ground truth masks.
			*	We draw a distinction between these two types of evaluation due to differing image representation (e.g., .png vs .jpg). 
        *   You should be able to obtain a DSC of `0.953` from DeepLab_v3, and `0.902` from Mask-RCNN. 
