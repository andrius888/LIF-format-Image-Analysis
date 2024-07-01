LIF file format microscopy Image Analysis Toolkit and example analysis

Description:
This repository contains a collection of Python tools made for the analysis of microscopy images that are saved in the LIF file format, both in 2D and 3D formats. It includes functions for importing LIF files, viewing images, 3D cell counting, DAPI dyed nuclei counting and pixel intensity extraction. It also contains two example analyses, one for a 2D dataset of images where  a differentiated cell group and an undifferentiated cell group is compared and another for a 3D dataset where the ratio of live and dead cells was counted after they were dyed with LIVE/DEAD™ Viability/Cytotoxicity Kit.

Data for example analyses can be found here:
(2D)
https://www.dropbox.com/scl/fi/c9gkcch2uoxbhh4hog101/RBMC029-epi-dif-2D.lif?rlkey=v55godl6p9hd970stz0vznbpo&st=5t353nss&dl=0

(3D)
https://www.dropbox.com/scl/fi/kstoxz1y4d2c295emeuku/RASC027-LiveDead-3D.lif?rlkey=s6ehozgi6hcyr9dl4o4w16dts&st=9pyop7d0&dl=0

Requirements:
python=3.11.5
readlif==0.6.5
numpy==1.26.0
matplotlib==3.8.0
scikit-image==0.20.0
scipy==1.11.3
pandas==2.1.4

