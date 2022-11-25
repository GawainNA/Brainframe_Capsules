# About the Capsule
## Usage
This capsule is for detecting objects in image. 

## How to use
1. Place the folder `detector_object_tf` under the server's data directory (`/var/local/brainframe` by default), there should be a directory called `capsules/`. If the `capsules/` directory does not exist, create it. Place `detector_object_tf` within this directory.

2. Start brainframe server by runing command
```bash
$ brainframe compose up -d
```

3. Then you should see `detector_object.cap` under the same directory, now open `capture_object.py` to see how to get output. The expamle output images are under `output/object/`

## Model
### Architecture & Performance
This model is a Mask R-CNN Inception ResNet V2 1024x1024 in Tensorflow.

### Sources
Trained using the [Tensorflow Object Detection API](
https://github.com/tensorflow/models/blob/master/research/object_detection)

###  Model File Origin
The pretrained model was downloaded from 
[vighneshbirodkar's github repo](
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

### Dataset
This model was trained on the [COCO 2017 dataset](http://cocodataset.org).
