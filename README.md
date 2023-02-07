# Image-Anomaly
Repo for detecting Anomalies in Images.

Anamoly here is a very losse open ended term which can means for example if we see some dataset which is "out of ordinary" it can be out of domain data, camera failure, gaussian noise, foreign objects e.t.c


We mainly Implement two Methods for Image Anamoly Detection for both of the methods we first train an autoencoder to map the image to higher level representation and

##Reconstruction based
Detect Anamoly based on Reconstruction loss of Autoencoder

##Kernel Density based
LAKE Model
Fit multinomial gaussian/ KDE on latent variable and find the anamoly based on PDF with some modifications

For both we acheived 76.9% F1 score for bottle anamoly

# Using Aomalib Library

DfDKM acheived  F1 score for bottle dataset of __92.7%__


<p align="center">
  <img alt="Light" src="vis_imgs/bottles/anamoly_1.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="vis_imgs/bottles/anamoly_2.png" width="45%">
</p>
<p align="center">
  <img alt="Light" src="vis_imgs/bottles/normal_1.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="vis_imgs/bottles/normal_2.png" width="45%">
</p>

DfDKM acheived  F1 score for Driving dataset of __94%__

<p align="center">
  <img alt="Light" src="vis_imgs/driving/anamoly_1.jpg" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="vis_imgs/driving/anamoly_2.jpg" width="45%">
</p>
<p align="center">
  <img alt="Light" src="vis_imgs/driving/normal_1.jpg" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="vis_imgs/driving/normal_2.jpg" width="45%">
</p>



## General Tips

1. Use Sigmoid as last layer to restrict output between 0 and 1
2. Use Nearest interpolation for resizing can effect max and min values a lot if using bilibear or cubic interpolation.
4. Donot use cosine similarity metric as proposed in LAKE on images as images which can vary immensely can have very high cosine similarity metric