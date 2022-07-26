# Image Emotion Detection
Emotion Detection via a Deep Learning Framework

![image](https://user-images.githubusercontent.com/729885/181081760-0c90f758-f3cb-49ba-8d45-611f565581c6.png)


## [Requirement]
- Python >= 3.x
- scikit-learn https://scikit-learn.org/stable/
- Numpy https://numpy.org/
- Keras https://keras.io/
- CV2 https://pypi.org/project/opencv-python/

## [Data]
- data/pretrained_data: only a subset of sampled images. For the full set, please email: `kpzhang@umd.edu`
- data/kickstarter_data: images of all crowdfunding project for this study

## [Trained Model]
Given the model size, please contact the author (kpzhang@umd.edu) for trained model.

## [Run] 
- `FPN_train.py`: Feature Pyramid Network
- `train.py`: training procedure of our model
- `predict.py`: predicting image emotions
- `xgboost.py`: one of baselines using XGBoost


## [Disclaimer]
Please refer to our forthcoming paper at MIS Quarterly for details: "Pictures that are Worth a Thousand Donations: How Emotions in Project Images Drive the Success of Online Charity Fundraising Campaigns? An Image Design Perspective", Jian-Ren Hou, Jennifer Zhang, and Kunpeng Zhang.
