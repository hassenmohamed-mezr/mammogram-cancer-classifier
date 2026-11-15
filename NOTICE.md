# Third-party notice

## RSNA Screening Mammography Breast Cancer Detection - 1st place solution
- Source: https://github.com/dangnh0611/kaggle_rsna_breast_cancer (MIT licence), by dangnh0611
- Used here: the trained YOLOX-nano breast ROI detector (`yolox_nano_416_roi_torch.pth`, from the author's
  Kaggle dataset `dangnh0611/rsna-breast-cancer-detection-best-ckpts`) and the ROI-extraction logic.
- `src/dicom_roi.py` is a port (pydicom + PyTorch checkpoint instead of dicomsdl/DALI/TensorRT) of
  `src/tools/_prepair_classification_dataset_stage2.py`, `src/roi_det/roi_extract.py` and `src/utils/windowing.py`.
- The author's weights and code are NOT redistributed in this repository; they are downloaded at run time.
- The classifier (ConvNeXt) in this repository is trained separately by this project.
