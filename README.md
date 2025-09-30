# Advanced Convolutions and Data Augmentation

This folder contains an implementation of a Convolutional Neural Network (CNN) using some of the advanced convolutions and data augmentation techniques on CIFAR10 dataset. 

## Folder Structure
```
└── README.md
└── src/
    └── data_setup.py
    └── utils.py
    └── engine.py
    └── model_builder.py
└── models/
    └── S9Model1.pth
    └── incorrect_images.png
    └── loss_accuracy_plot.png
└── train.py
└── S9.ipynb
```

## How to Run the code
Clone the repo and run
Change your current directory to S9
```
python train.py
```

## Receptive Field Calculations
| |r_in|n_in|j_in|s|r_out|n_out|j_out| |kernal_size|padding|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|Conv |1|32|1|1|3|32|1| |3|1|
|Conv|3|32|1|1|5|32|1| |3|1|
|Conv (Dilated)|5|32|1|1|9|30|1| |5|1|
|Conv|9|30|1|1|11|30|1| |3|1|
|Conv|11|30|1|1|13|30|1| |3|1|
|Conv (Stride2)|13|30|1|2|15|15|2| |3|1|
|Conv|15|15|2|1|19|15|2| |3|1|
|Conv|19|15|2|1|23|15|2| |3|1|
|Conv (Stride2)|23|15|2|2|27|8|4| |3|1|
|Conv (DWS)|27|8|4|1|35|8|4| |3|1|
|Conv|35|8|4|1|43|8|4| |3|1|
|Conv |43|8|4|1|51|8|4| |3|1|
|GAP|51|8|4|1|79|1|4| |8|0|

## Convolutions

### Normal Convolution
![1_d03OGSWsBqAKBTP2QSvi3g](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/9ff2d277-e8bc-46a9-8df1-1e2479038d7f)


### Strided Convolution (stride = 2)
![1_NrsBkY8ujrGlq83f8FR2wQ](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/c5e1502f-1f8a-4c9e-8f7a-1f924dd690ad)


### Dilated Convolution (dilation=2)
![1_niGh2BkLuAUS2lkctkd3sA](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/a55d83cb-482f-4995-aab6-036f6be55066)


### Depthwise Separable Convolution
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/b5a6fc49-b574-4fac-a513-600313a212b9)


### Data Augmentations Used
1. Horizontal Flip
2. ShiftScaleRotate
3. CoarseDropout

## Training and Testing Results
* Total Parameters: 150,866
* Best Train Accuracy: 82.18
* Best Test Accuracy: 86.78

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/2bba9013-cc06-44d1-9546-c66b6875cb93)

## Key Results
1. Has the architecture to C1C2C3C40
2. Used Dilated kernels here instead of MP or strided convolution in first block
3. Total RF is more than 44 (79)
4. One of the layers must use Depthwise Separable Convolution
5. One of the layers must use Dilated Convolution
6. Used GAP and added FC after GAP to target #of classes 
7. Used albumentation library and applied
       - horizontal flip
       - shiftScaleRotate
       - coarseDropout 
8. Achieved more than 85% accuracy (86.78), in 50 epochs.
9. Total Params to be less than 200k. (150,866)
10. Created a modular code

## Incorrect Classified Images
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/53c474ec-95a0-468a-9200-3a6a6aa76324)


## References
* https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
* https://www.learnpytorch.io/05_pytorch_going_modular/
