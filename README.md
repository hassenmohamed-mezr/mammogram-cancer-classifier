# Mammogram Cancer Classifier

An experimental binary classifier that maps a mammogram image to a 0/1 prediction. Its output is a mathematical prediction, not a medical assessment.

## Planned pipeline

Breast ROI crop -> ConvNeXt classifier -> probability -> threshold -> 0/1 label.

## Credits

The ROI (YOLOX) model is from the RSNA Breast Cancer Detection winning solution by dangnh0611: https://github.com/dangnh0611/kaggle_rsna_breast_cancer

## Disclaimer

Research/demo only. Not for clinical use.
