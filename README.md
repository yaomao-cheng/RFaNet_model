# RFaNet_model
=================================================================
# Code for the paper

**Please cite this paper if you use the code**

=================================================================

## Statement
  * you can find the model in **utils/models/**
  * The code is trained and tested on **ASL Dataset, NTU Dataset, and OUHANDS Dataset** in numpy file
  * you can train your datasets by following the **Training Steps** shown below
  * If the released files exist any error, please contact me. Thanks !
  
## Requirements
  * **Python 3.6**
  * **pytorch 1.1.0**
  * tensorflow-gpu 1.15.0
  * matplotlib 3.0.3
  * cupy 5.4.0
  * pillow==6.2.1
  * scipy==1.1.0
  * scikit-learn == 0.23.2
  * opencv-python == 4.4.0.46
  * scikit-image==0.17.2 (python3.6+ is needed)
  * mmcv == 1.2.1
  * tensorboardX == 2.1
  
  * **GPU environment**
    * CUDA Version 10.0.130-410.48
    * CUDNN Version 7.3.0.29
      * #define CUDNN_MAJOR      7
      * #define CUDNN_MINOR      3
      * #define CUDNN_PATCHLEVEL 0
      * #define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
    
## Dataset
  * The paper which proposed the **ASL finger-spelling dataset** can be found in  [**Spelling It Out: Real–Time ASL Fingerspelling Recognition**](https://empslocal.ex.ac.uk/people/staff/np331/publications/PugeaultBowden2011b.pdf) 
    * [1] Pugeault, N., and Bowden, R. (2011). Spelling It Out: Real-Time ASL Fingerspelling Recognition In Proceedings of the 1st IEEE Workshop on Consumer Depth Cameras for Computer Vision, jointly with ICCV'2011.
    
  * The paper which proposed the **NTU hand difits dataset** can be found in  [**Robust Hand Gesture Recognition Based on FingerEarth Mover’s Distance with a Commodity Depth Camera**](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/Ren_Yuan_Zhang_MM11short-1.pdf)  
    * [2] Ren, Zhou, Junsong Yuan, and Zhengyou Zhang. "Robust hand gesture recognition based on finger-earth mover's distance with a commodity depth camera." Proceedings of the 19th ACM international conference on Multimedia. 2011.
    
  * The paper which proposed the **OUHANDS dataset** can be found in  [**OUHANDS database for hand detection and pose recognition**](https://ieeexplore.ieee.org/abstract/document/7821025)
    * [3] Matilainen, Matti, et al. "OUHANDS database for hand detection and pose recognition." 2016 Sixth International Conference on Image Processing Theory, Tools and Applications (IPTA). IEEE, 2016. 

## pre-processing steps
    ASL dataset
        Depth : (1) segment the hand region from RAW 16bit depth image and save as 8 bit image
                (2) resize to 64x64 and save as .npy
                
    NTU, OUHANDS dataset
        Depth : (1) segment the hand region from RAW 16bit depth image and save as 8 bit image
                (2) perform CCL to remove the remainder of background
                (3) crop out the hand region by detecting the bounding box of hand region within the entire image
                (4) resize to 64x64 and save as .npy
 
## Training Steps
  1. Run **main_D_Smart_Val.py** with one GPU or multiple GPU to train the Network. Furthermore, set your certain ArgumentParser or default one.
  
      **ArgumentParser elements**
      ```python
      -sr --s <weights decay parameter> --arch <model architecture> --depth <model depth> --batch-size <batch number of training stage> --test-batch-size<atch number of testing stage> --subject <training/testing subject in training/testing stage>
      ```

      **Command example**
      ```python
      python main_D_Smart_Val.py -sr --s 0.0001 --arch vgg_D_ASPPSA_v4_DSB --depth 10 --batch-size 64 --test-batch-size 32 --subject SubjectE/
