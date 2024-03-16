# **Nordic Vehicle Dataset**

This repository contains code for fine-tuning YOLOv5, YOLOv8, and Faster R-CNN on the NVD dataset.

You can find the NVD dataset and the models' weights at the following link:

https://nvd.ltu-ai.dev/

Please refer to the setup instructions provided below.

## **Sample Frames**

<center>
<img src='./sample-images/2022-12-02 Asjo 01_stabilized_1.png' alt="Sample 1" width="640" height="360">
<img src='./sample-images/2022-12-02 Asjo 01_stabilized_2.png' alt="Sample 2" width="640" height="360">
<img src='./sample-images/2022-12-02 Asjo 01_stabilized_4.png' alt="Sample 3" width="640" height="360">
<img src='./sample-images/2022-12-03 Nyland 01_stabilized_1.png' alt="Sample 4" width="640" height="360">
<img src='./sample-images/2022-12-03 Nyland 01_stabilized_2.png' alt="Sample 5" width="640" height="360">
<img src='./sample-images/2022-12-03 Nyland 01_stabilized_3.png' alt="Sample 6" width="640" height="360">
<img src='./sample-images/2022-12-04 Bjenberg 02_1.png' alt="Sample 7" width="640" height="360">
<img src='./sample-images/2022-12-04 Bjenberg 02_2.png' alt="Sample 8" width="640" height="360">
<img src='./sample-images/2022-12-04 Bjenberg 02_3.png' alt="Sample 9" width="640" height="360">
<img src='./sample-images/2022-12-23 Asjo 01_HD 5x stab_1.png' alt="Sample 10" width="640" height="360">
<img src='./sample-images/2022-12-23 Asjo 01_HD 5x stab_5.png' alt="Sample 11" width="640" height="360">
<img src='./sample-images/2022-12-23 Asjo 01_HD 5x stab_6.png' alt="Sample 12" width="640" height="360">
<img src='./sample-images/2022-12-23 Bjenberg 02_stabilized_2.png' alt="Sample 13" width="640" height="360">
<img src='./sample-images/2022-12-23 Bjenberg 02_stabilized_6.png' alt="Sample 14" width="640" height="360">
</center>

## **Setup**
Clone the repository from GitHub:

    git clone 'https://github.com/Amirhossein-Nayebi/Nordic-Vehicle-Dataset'

Verify that Python is installed and the version is between 3.8 (inclusive) and 3.11 (exclusive).

On Windows:

    python --version

On Ubuntu:

    python3 --version

This will display the version of Python installed on your system. If the version is less than 3.8 or equal to or greater than 3.11, you will need to install a compatible version.

Create a directory (```videos directory```) and copy the video files and the accompanying annotation ```.xml``` files into it. Ensure that each annotation file has the same name as its corresponding video file. You can download NVD video and annotation files [here](https://nvd.ltu-ai.dev/).

Navigate to the ```source directory``` by changing the current directory:

    cd Nordic-Vehicle-Dataset

Create and activate a virtual environment:

On Windows:

    python -m venv .venv
    .venv\Scripts\Activate.ps1

On Ubuntu:

    python3 -m venv .venv
    source .venv/bin/activate


Note: ```.venv``` is the name of the virtual environment, you can change it to any other name of your choice.
Now upgrade ```pip``` and install the required packages:
    
    python -m pip install --upgrade pip
    pip install -r requirements.txt

After you have set up the environment, before executing any of the following commands, make sure to activate the environment if it has not been activated already.

Windows:

    .venv\Scripts\Activate.ps1

Ubuntu:
    
    source .venv/bin/activate

## **Data Visualization and Preparation**

### **Visualize Annotations**
Before training/validation/test data is prepared, it's crucial to ensure that the annotated data is correct and aligned with the video frames properly. There is a ```view_annotations.py``` script that can be utilized to visualize an annotation file:

    python view_annotations.py [path/to/an/annotation/file]

It's worth mentioning that there should be a video file with the same name as the annotation file located in the same directory or a directory containing extracted frames with the same name as the annotation file, located alongside the annotation file.

After running the script, the frames with augmented annotated data will be displayed one after another by pressing a key. You can halt the display process by pressing the ```'q'``` key.

### **Prepare data**
To prepare the data, we need to extract frames from the videos and convert the annotations for each frame into the YOLO format. Also, we need to split the data into training, validation and test sets. This can be accomplished using the ```prepare_data.py``` script:

    python prepare_data.py [--source ANNOTATIONS_DIR or ANNOTATION_FILE] [--data_dir data]

This will create a ```data``` directory and store all of the extracted frames and label data inside it. 

As the ```source```, you can provide either of the following:

* A directory containing annotation files and their corresponding videos/frames with the same name alongside themselves.
* A single annotation file that has a video file or frames directory with the same name alongside itself.

### **Visualize Prepared Data**
You can visualize all of the prepared data or only the data created for a specific video using the ```view_data.py``` script as follows:

    python view_data.py [--type TYPE] [--video_file VIDEO_FILE_NAME]

The ```TYPE``` can be either ```train```, ```val```, or ```test```, indicating which split of the data you want to view. If the ```--video_file``` argument is provided, only the data related to that specific video file will be displayed. Due to the possibility of a large amount of data, it is recommended to check the data related to each video file individually by using the ```--video_file``` argument to ensure that the data has been correctly extracted from each video file. Similarly to the ```view_annotations.py``` script, this script also opens a window and displays the frames with augmented annotated data by pressing a key. To end the display process, press the ```'q'``` key.

## **Set up Logging**
Before starting the training process for the first time, you need to set up logging. To do this, you will need a ```ClearML``` account. To create an account or login to an existing one, navigate to [https://app.clear.ml/settings/workspace-configuration](https://app.clear.ml/settings/workspace-configuration), click on your profile picture, go to the ```Settings``` page, then click on ```Workspace```. Next, press ```Create new credentials``` and then press ```Copy to clipboard```. Finally, run the following command in the terminal with the activated environment:

    clearml-init
  
Then paste copied configuration when prompted and press ```Enter```. If everything is okay, you should see the following message:
    
    ClearML Hosts configuration:
    Web App: https://app.clear.ml
    API: https://api.clear.ml
    File Store: https://files.clear.ml

    Verifying credentials ...
    Credentials verified!

    New configuration stored in <path to the created clearml.conf>
    ClearML setup completed successfully.

## **YOLO**
### **Train**
To train the networks using the prepared data, run:

    python train.py [--epochs EPOCHS] [--yolo_model YOLO_MODEL] [--batch BATCH-SIZE]

Replace ```EPOCHS``` with the number of training epochs and ```YOLO_MODEL``` with one of the following YOLO models:
    
    yolov5n, yolov5s, yolov5m, yolov5l, yolov5x, yolov8n, yolov8s, yolov8m, yolov8l, yolov8x  

```yolov*n``` has the lowest number of parameters and the fastest speed. 
```yolov*x``` has the maximum number of parameters and the lowest speed.

If you get out of memory error try to reduce ```BATCH-SIZE```.

Training is a long process and requires a huge amount of system resources. You can log in to you ClearML account to view the progress and results.

### **Test**
After the training is finished, you can validate the trained model with the test set (the test set is created automatically in the data preparation phase and is not used for training.):

    python val.py [--yolo_model YOLO_MODEL]

Here for ```YOLO_MODEL``` provide the path to a trained network's weight file (```.pt```) located in the ```runs/train``` directory. The validation results are saved in ```runs/val``` directory.

### **Detect Cars**
To detect cars in videos and images you can simply pass a file (video/image) or a directory path containing videos/images to ```detect.py``` script:

    python detect.py [--yolo_model YOLO_MODEL] [--source FILE/DIR] [--conf_thres CONF_THRES]

As before ```YOLO_MODEL``` is the path to a trained network's weights file, ```FILE/DIR``` is a file path to an image or video or a directory path containing images and videos. ```CONF_THRES``` is the confidence threshold which is set to 0.25 by default. The detection results are saved in ```runs/detect``` directory.

## **Faster R-CNN**
Faster R-CNN only works on Linux systems and has been tested on Ubuntu 18.04. Before fine-tuning Faster R-CNN, you need to uncomment the last lines in the ```requirements.txt``` file. To fine-tune Faster R-CNN you can run ```finetune_faster_rcnn.py```:

    python finetune_faster_rcnn.py

The results will be saved in the ```output``` directory.

Please note that you should prepare the data as explained previously before running this script. 

To detect cars in images you can simply pass an image file or a directory path containing images to ```detect_frcnn.py``` script:

    python detect_frcnn.py [--model FRCNN_MODEL_WEIGHTS] [--source FILE/DIR] [--conf_thres CONF_THRES]

```FRCNN_MODEL_WEIGHTS``` is the path to a trained network's weights file, ```FILE/DIR``` is a file path to an image or a directory path containing images. ```CONF_THRES``` is the confidence threshold which is set to 0.25 by default. The detection results are saved in ```runs/detect``` directory.

## **Citation**
H. Mokayed, A. Nayebiastaneh, K. De, S. Sozos, O. Hagner, and B. Backe, “Nordic Vehicle Dataset (NVD): Performance of vehicle detectors using newly captured NVD from UAV in different snowy weather conditions.” arXiv, Apr. 27, 2023. doi: 10.48550/arXiv.2304.14466.

## **License**
The dataset provided in this project is made available under the Creative Commons Attribution-NonCommercial (CC-BY-NC) license. This means that you are free to use the dataset for scientific research purposes, subject to the following conditions:

* You must provide appropriate attribution to the original creator(s) of the dataset.
* You may not use the dataset for commercial purposes without obtaining explicit permission or a separate license from the dataset's owner.

By accessing and using the dataset, you agree to comply with the terms and conditions of the CC-BY-NC license.
