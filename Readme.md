### DM data science assessment task

Dataset contains list of series of measurements of air pressure provided by Air Pump, where in some cases Pump failed and in some cases not. Goal is to build a predictive model that will predict whether Pump worked fine or not. 

There are presented 2 approaches for this task:
1. Deriving of standard statistical features from individual series of measurements and then development of Gradient boosting model based on these features.
2. Converting Individual series to the image files and approach this task as the image classification, where convolutional neural network is developed from those images.

For the comparison purposes, test set is the same for both approaches.

Please check first tabular data approach, because some general thoughts are more described there.