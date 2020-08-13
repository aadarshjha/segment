# Usage

1.  *Manual Detection Phase*
	* We begin by: 
		* `cd` into `./manual-detection`
		* The `segmentation-pipeline/` folder contains both U-Net and DeepLab_v3 code to produce respective model (.pth) files to perform and remember segmentation results upon a given input set. You may use this to perform your own segmentation. 
		* The `prediction-pipeline` folder takes a set of input images and a model file to produce predicted images. 
			* With the images provided in the `data/` folder, as well as the model file, you may use this prediction pipeline (namely, the `u-net-prediction/` folder) to reproduce the images used in the experiment. 
		* Finally, to evaluate the performance of any segmentation model, simply use the `dsc-evaluation/general-evaluation/` folder and input both ground truth and predicted masks to obtain a CSV file of data. Average the DSC values for each photo to obtain the results presented in this experiment (a value of `0.940` should be obtained). 
