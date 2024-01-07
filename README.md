LIF file format microscopy Image Analysis Toolkit and example analysis

Description:
This repository contains a collection of Python tools made for the analysis of microscopy images that are saved in the LIF file format, both in 2D and 3D formats. It includes functions for importing LIF files, viewing images, 3D cell counting, DAPI dyed nuclei counting and pixel intensity extraction. It also contains two example analyses, one for a 2D dataset of images where  a differentiated cell group and an undifferentiated cell group is compared and another for a 3D dataset where the ratio of live and dead cells was counted after they were dyed with LIVE/DEAD™ Viability/Cytotoxicity Kit.

Data for example analyses can be found here:
https://www.dropbox.com/scl/fo/7wwwq3phqfjf9zlag0vaw/h?rlkey=vumirgc95v39383tf6pjn5pjr&dl=0

Requirements:
python=3.11.5
readlif==0.6.5
numpy==1.26.0
matplotlib==3.8.0
scikit-image==0.20.0
scipy==1.11.3
pandas==2.1.4
