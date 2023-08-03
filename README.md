# Refined lithology identification

Refined lithology identification is the reliable basis for determining the spatial distribution of lithology, reserve estimation and 3D modeling. With the development of artificial intelligence, there have been some intelligent identification methods that can identify coarse-grained classes of rock. However, the research progress of fine-grained lithology identification is slow due to the difficulty of fine-grained identification, the high cost of data collection and complexity of engineering site conditions. Against this background, we conducted research on fine-grained identification based on macroscopic images, analyzed the differences between fine-grained identification and coarse-grained identification, explored the differences between image identification under laboratory conditions and on-site conditions, and discussed the development direction of lithology identification. Specifically, we constructed a laboratory dataset (160 lithologies) and an on-site dataset (13 lithologies); then, we developed a classification model specifically for extracting lithological features; next, we conducted identification experiments using the self-built model and detection model on these two datasets and comparative analysis was carried out. The results show that the self-developed model has an evaluation metric (F1-score) of 0.9764 on the images collected in the laboratory, but an F1-score of only 0.6143 on the images collected in the engineering site. Unlike coarse-grained identification, image augmentation that changes local features degraded the fine-grained lithology identification performance. Factors such as surface contamination caused by geological process, weathering and similarity in appearance due to metamorphism in the engineering site can affect the accuracy of image-based methods. This study explores in detail the potential and existing problems of fine-grained lithology identification based on macroscopic images, which can provide reference for future reservoir-related issues.

## Dependencies

This code requires the following:

Python >= 3.7
numpy==1.22.2
Pillow==5.4.1
scikit-learn==0.21.1
scipy==1.2.1
torch==1.9.1
torchvision==0.4.2
tqdm==4.36.1
tensorboardx==1.7
tensorboard==1.13.1

## Training


