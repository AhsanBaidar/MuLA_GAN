# MuLA_GAN
Multi-Level Attention based GAN for Underwater Image Enhancement [paper](https://www.sciencedirect.com/science/article/pii/S1574954124001730)

## Before you start.

Clone Respository
```
git clone https://github.com/AhsanBaidar/MuLA_GAN
```
Navigate to folder MuLA_GAN
```
cd MuLA_GAN
```


## Training a Model
The training dataset split used in the paper is provided in the folder named "Dataset/train". [Download from here](https://drive.google.com/file/d/1moiW0Ptf5blF-hncV38mNtqXrg3vhSwr/view?usp=sharing).
Start training by using the following command:

```  
python train_MuLA_GAN.py
```
Or by running this script using pthon IDE.

## Testing a Model
The testing dataset split used in the paper is provided in the folder named "Dataset/test".
Testing can be done using the following command by usign your trained model or pre-trained model weights using this link [weights](https://drive.google.com/file/d/17Z-VgIKjDuzoBnq9HU3y5lzvUHUHBmQy/view?usp=sharing:) Svae these weights in checkpoints/UIEB/ directory and run:
``` 
python test.py --weights_path checkpoints/UIEB/generator_299.pth
```
Or by running this script in python IDE.
  
## Visual Comparisons
The UIEB (Underwater Image Enhancement Benchmark) dataset is a diverse collection of degraded underwater images, widely used for evaluating underwater image enhancement techniques. The dataset was categorized into eight appearance-based groups, each capturing distinct underwater scenes. Here's a visual comparison for each scene:

#### Greenish Color Scene
![Greenish Color Scene](Visual%20Comparison/Greenish.jpg)

#### Bluish Color Scene
![Bluish Color Scene](Visual%20Comparison/Blueish.jpg)

#### Yellowish Color Scene
![Yellowish Color Scene](Visual%20Comparison/Yellowish.jpg)

#### Downward-looking Scene
![Downward-looking Scene](Visual%20Comparison/Downward_Looking.jpg)

#### Upward-looking Scene
![Upward-looking Scene](Visual%20Comparison/Upward_Looking.jpg)

#### Forward-looking Scene
![Forward-looking Scene](Visual%20Comparison/Forward_Looking.jpg)

#### Low Backscattered Scene
![Low Backscattered Scene](Visual%20Comparison/Low%20Back_Scattered.jpg)

#### High Backscattered Scene
![High Backscattered Scene](Visual%20Comparison/High%20Back_Scattered.jpg)

---


## 📚 Citation

If you use **MULA-GAN** in your work, please cite:

    @article{BAKHT2024102631,
      author = {Ahsan B. Bakht and Zikai Jia and Muhayy Ud Din and Waseem Akram and Lyes Saad Saoud and Lakmal Seneviratne and Defu Lin and Shaoming He and Irfan Hussain},
      title = {MuLA-GAN: Multi-Level Attention GAN for Enhanced Underwater Visibility},
      year = {2024},
      journal = {Ecological Informatics}
    }

---

