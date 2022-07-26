# Image Emotion Detection

## Emotion Detection via a Deep Learning Framework

## [Abstract]
![image](https://user-images.githubusercontent.com/729885/181081760-0c90f758-f3cb-49ba-8d45-611f565581c6.png)


## [Requirement]
- Python >= 3.x
- scikit-learn https://scikit-learn.org/stable/
- Numpy https://numpy.org/
- Keras https://keras.io/
- CV2 https://pypi.org/project/opencv-python/

## [Data]
- `data/pretrained_data`: only a subset of sampled images. For the full set, please email: kpzhang@umd.edu or refer to: https://dl.acm.org/doi/10.5555/3015812.3015857
- `data/kickstarter_data`: images of all crowdfunding projects for this study

## [Trained Model]
The trained model can be downloaded <a href='https://github.com/kpzhang/kpzhang.github.io/tree/main/models/model_weights.h5'>here</src>

## [Run] 
- `FPN_train.py`: Feature Pyramid Network
- `train.py`: training procedure of our model
- `predict.py`: predicting image emotions
- `xgboost.py`: one of baselines using XGBoost
- `get_objects_google.py`: using google vision API to obtain objects embedded in an image

- To obtain the ANPs, please refer to the details here: https://www.ee.columbia.edu/ln/dvmm/vso/download/sentibank.html


## [Disclaimer]

Please refer to our forthcoming paper at MIS Quarterly for details: 
"Pictures that are Worth a Thousand Donations: How Emotions in Project Images Drive the Success of Online Charity Fundraising Campaigns? An Image Design Perspective," Jian-Ren Hou, Jennifer Zhang, and Kunpeng Zhang.

