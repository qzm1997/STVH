# Deep Spatiotemporal Interleaved Video Activity Hashing(STVH)

### The official implementation of Deep Spatiotemporal Interleaved Video Activity Hashing

## Requirements
Linux with Python >= 3.7\
PyTorch = 1.10.0\
torchvision that matches the PyTorch installation\
CUDA 11.3

## Dataset
The datasets files could be obtained from [VD](https://github.com/mostafa-saad/deep-activity-rec) and [CAD](https://github.com/mostafa-saad/deep-activity-rec). \

```
cd STVH
mkdirs ./data/VolleyBall
mv VD ./data/VolleyBall
mkdirs ./data/ActivityDataset
mv CAD ./data/ActivityDataset
```
## Training Weight
CAD weight is in https://pan.baidu.com/s/19AdVbnqWa0zM_cEIb7wn2w 
提取码：f1b2 and VD weight is in https://pan.baidu.com/s/16SVyV3UIoASibAG6JqG7kw 
提取码：suvt

## Run
CAD
```
python ./scripts/CAD_train.py
```
VD
```
python ./scripts/VD_train.py
```


