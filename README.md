# MuLA_GAN
Multi-Level Attention based GAN for Underwater Image Enhancement [paper link will be added soon]

### Before you start.

Clone Respository
```
git clone https://github.com/AhsanBaidar/MuLA_GAN
```
Navigate to folder MuLA_GAN
```
cd MuLA_GAN
```


### Training a Model
The training dataset split used in the paper is provided in the folder named "Dataset/train".
Start training by using the following command:

```  
python train_MuLA_GAN.py
```
Or by running this script using pthon IDE.

### Testing a Model
The testing dataset split used in the paper is provided in the folder named "Dataset/test".
Testing can be done using the following command by usign your trained model or pre-trained model weights using this link [weights](https://drive.google.com/file/d/17Z-VgIKjDuzoBnq9HU3y5lzvUHUHBmQy/view?usp=sharing:) Svae these weights in checkpoints/UIEB/ directory and run:
``` 
python test.py --weights_path checkpoints/UIEB/generator_299.pth
```
Or by running this script in python IDE
  
