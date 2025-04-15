The following Readme is an adapted version of the DDPM & DDIM PyTorch re-implementation. It is mainly based on [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
This project also uses an external, custom dataset that found in ([Drunkard's Odometry](https://drive.google.com/drive/folders/1AZHUKMbe7bR1xwRmAAZ0AHgcEqRnNjms)

Reimplementation is done using the below project to work with a small portion of the Drunkards Odometry; this is using google colab restrictions. 
The requirements.txt is up to date

Specifically for this project I used the 320 dataset, scene 0 and level 0 "Color" dataset.

To reimplement exactly as seen in the results for the experimentation do the following in a Google Colab notebook :


Copy over any of the color.zip files into any directory in google drive
Tested for python 3.8.17 with torch 1.12.1+cu113 and torchvision 0.13.1+cu113.
- from google.colab import drive
- drive.mount('/content/drive')
- !git clone https://github.com/sanketnadg/DDPM.git
- !unzip '<location_of_color_compressed_folder>/color'.zip -d '/content/'
- !pip install -r requirements.txt
- python '/content/DDPM/train.py' -c '/content/DDPM/config/drunkards.yaml' (this will begin the training process)


Results of this codebase on preexisting datasets

|  Dataset  | Model checkpoint name | FID (↓) | 
|:---------:|:---------------------:|:-------:|
|  Cifar10  |   [cifar10_64dim.pt](https://drive.google.com/file/d/1vHfP8f_viyadhuXMaLfAQ1Iu5UE0WiiJ/view?usp=drive_link)    |  11.81  |
|  Cifar10  |   [cifar10_128dim.pt](https://drive.google.com/file/d/1NtysETxHPinns6JabjawyWTnkjJKT34M/view?usp=drive_link)   |  8.31   |
| CelebA-HQ |   [celeba_hq_256.pt](https://drive.google.com/file/d/1zzZbkNkMYCFKmWKW5Sh2JsrUsNrWyDCs/view?usp=drive_link)    |  11.97  |

DDPM objective: minimize E[ || ε - ε_θ(x_t, t) ||^2 ]

What the diffusion process actually looks like :

<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/cifar10_128_ex1.png" height="350" width="350">
  <img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/cifar10_128_ex2.gif" height="350" width="350">
</div>

<br><br>

<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/celeba_hq_ex1.png" height="350" width="350">
  <img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/celeba_hq_ex1.gif" height="350" width="350">
</div>

- cifar10_64dim

<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/cifar10_64_ex1.png" height="350" width="350"> &nbsp; &nbsp; &nbsp;
<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/cifar10_64_ex2.gif" height="350" width="350">

- cifar10_128dim

<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/cifar10_128_ex3.png" height="350" width="350"> &nbsp; &nbsp; &nbsp;
<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/cifar10_128_ex4.gif" height="350" width="350">

- celeba_hq_256

<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/celeba_hq_ex2.png" height="350" width="350"> &nbsp; &nbsp; &nbsp;
<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/celeba_hq_ex2.gif" height="350" width="350">
<br><br>
<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/celeba_hq_ex3.png" height="350" width="350"> &nbsp; &nbsp; &nbsp;
<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/celeba_hq_ex3.gif" height="350" width="350">
<br><br>
<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/celeba_hq_ex4.png" height="350" width="350"> &nbsp; &nbsp; &nbsp;
<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/celeba_hq_ex4.gif" height="350" width="350">


```
- DDPM
    - /config
    - /src
    - /images_README
    - inference.py
    - train.py
    ...
    - /data (make the directory if you don't have it)
        - /celeba_hq_256
            - 00000.jpg
            - 00001.jpg
            ...
            - 29999.jpg
    
```
Expected Drunkards Odometry Dataset
<br><br>
<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/samples_datasets.jpg" height="600" width="800">

The drunkards odometry flow is very complex, but the main idea behind it is to analyze constantly deforming scenes in all direction to create plausible navigation
<br><br>
<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/Overview_drunk2.jpg" height="600" width="800">


Results from training on a small color dataset using DDPM with many computational limitations
<br><br>
<img src="https://raw.githubusercontent.com/sanketnadg/DDPM/master/images_README/drunkardresult.png" height="600" width="800">


Here is the link to my presentation for this : [https://drive.google.com/file/d/1XuWaglvz6RuRboCfKwrPK70ITXy1nLmx/view?usp=drive_link](https://drive.google.com/file/d/1pGT5mvfvbcbD6sca6Z8OvUwfBIIMGlzz/view?usp=drive_link)
