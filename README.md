# Automatically Detecting Bregma and Lambda Points in Rodent Skull Anatomy Images
by Peng Zhou; Zheng Liu; Hemmings Wu; Yuli Wang; Yong Lei; Shiva Abbaszadeh

Currently, injection sites of probes, cannula, and optic fibers in stereotactic neurosurgery are typically located manually. This step involves location estimations based on human experiences and thus introduces errors. In order to reduce localization error and improve repeatability of experiments and treatments, we investigate an automated method to locate injection sites. This paper proposes a localization framework, which integrates a regional convolutional network and a fully convolutional network, to locate specific anatomical points on skulls of rodents. Experiment results show that the proposed localization framework is capable of identifying and locating bregma and lambda in rodent skull anatomy images with mean errors less than 300 μm. This method is robust to different lighting conditions and mouse orientations, and has the potential to simplify the procedure of locating injection sites.

### Files/folders:

* `./stage1/`: folder for the training and result in stage one.
`train_box_coordinates.txt` contains the training box coordinates.
Result images in `./stage1/result_images`.
Result coordinates in `result_box_coordinates.txt`.
Modified coordinates as a square box in `.stage2/box_coordinates_s.txt`.

* `./stage2/`: folder for the training and result in stage two.
`box_coordinates_s.txt` obtained from stage 1.
`bregma_lambda_coodinates.txt` contains the ground truth coordinates.
`data_label_aug.ipynb` is for data augmentation and gaussian label generating.
`resize.ipynb` is for image resizing.
`./stage2/result_images` folder contains result images.
`resultAnalysis.ipynb` is for finding the bregma and lambda, calculating the error, and analyze the error.

Dependences: Python, h5py, Keras==2.0.3, numpy, opencv-python, sklearn, tensorflow==1.14.0, skimage

### How to use:
1. stage 1
- For training: run: `$ python train_frcnn.py -o simple -p input.txt` 
- For testing: run: `$ python test_frcnn.py -p /path/to/test_images/` 

2. stage 2
- For training: run: `$ python train_fcn.py` 
- For testing: run: `$ python python test_fcn.py`
