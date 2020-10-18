# SMALL_ANIMAL
Automated Detection of Bregma and Lambda Locations in Rodent Skull Anatomy Images

Stage 1:

train_box_coordinates.txt contains the training box coordinates.

For training: python train_frcnn.py -o simple -p input.txt

For testing: python test_frcnn.py -p /path/to/test_images/

Result images in result_images folder

Result coordinates in result_box_coordinates.txt

Modified coordinates as a square box in stage2/box_coordinates_s.txt

Stage 2:

box_coordinates_s.txt obtained from stage 1.

bregma_lambda_coodinates.txt contains the ground truth coordinates.

data_label_aug.ipynb is for data augmentation and gaussian label generating.

resize.ipynb is for image resizing.

For training: python train_fcn.py

For testing: python test_fcn.py

result_images folder contains result images.

resultAnalysis.ipynb is for finding the bregma and lambda, calculating the error, and analyze the error.
