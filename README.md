# Unet architecture using tensorflow 2.2


[![Logo](https://github.com/smalldan1022/Corneal-ulcer/blob/master/pictures/CAIM.jpg)](https://www1.cgmh.org.tw/intr/intr2/c3sf00/caim/home/index)

[![Website online](https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online)](https://www1.cgmh.org.tw/intr/intr2/c3sf00/caim/home/news)


## Introduction

Unet is an architecture used widely in medical field to solve the high resolution problem on most of the medical images. There are lots of version of the Unet architecture. However, it is the pytorch version or the old keras version, and thus can't get the advantage of the new vesrion tensorflow.Therefore, I provide a tensorflow 2.2.0 version and the simple architecture code for people to use and modify.


## Explanation


    1.MakeDataset.py
        For the purpose of making a dataset by the tensorflow iterator, tf.data.Dataset, which is faster than 
        using python list to read images.

    2.Unet_model.py
        To make the Unet model by using tensorflow inheritance, tf.keras.Model. You can use the Model func for sure, it's up to you.

    3.utils.py
        Provide the utilities for you to visualize the results of your AI model. There are also some image processing methods for you to enhance the segmentation results using the opencv.
    4.Unet_main.py
        The main function to control the whole process of the Unet model training. There are lots of thing you can tune yourself in it, like the hyperparameters and the callbacks. Feel free to modify it yourself.



## Check list

Be sure to check all the steps below to make sure nothing goes wrong in your training process.

- [x] Data leakage -> divide the dataset by patients
- [x] Different dataset distribution -> use five fold method
- [x] Save all the train values -> don't miss any train/valid values or the initial training hyperparameters
- [x] Plot the result for you to check -> check whether the model got the right features or not  



## Pre-requisites


### For the data structure

#### Prepare the dataset for Unet model

``` bash
Unet
├── Train
│   ├── images
│   └── masks 
└── Validation
    ├── images
    └── masks
```

### Make a csv file, you need to manually use different parameters


#### For the train image csvfile

```bash
find $PWD/Train/images/* -iname "*.jpg" | awk -F'/' 'BEGIN{OFS=",";print"Path,Y,ID"}{print $0,$6,$8}' > Train_image.csv
```
*Hint : Be sure to change the "$0,$6,$8" term. Modify them dependng on the path of your images. You can see the command awk for more details.*

#### For the train mask csvfile 

```bash
find $PWD/Train/masks/ -iname "*.jpg" | awk -F'/' 'BEGIN{OFS=",";print"Path,Y,ID"}{print $0,$6,$8}' > Train_mask.csv
```

*Hint:You can make yor csv file in any form, just make sure you have the three columns:["Path","Y","ID"]*



## How to use the model

Assume you have all the needed dataset. If not, go through the [Pre-requisites](#Pre-requisites) and get all the needed datasets.

### Use the Unet model


``` bash
python Unet.main.py

# You can modify the hyperparameters in the u2net_train.py, like:
# epoch = 1000
# learning rate = 0.0001
# callbacks = [x, y, z]
# metrics = IOU
# optimizer = adam
```
*Hint : Make sure to change all the file paths of yours, and also be aware of the different naming method between mine and yours.*

