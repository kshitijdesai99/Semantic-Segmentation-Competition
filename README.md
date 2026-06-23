# Efficient Road-Scene Semantic Segmentation

This project explores semantic segmentation for autonomous-driving scenes under a combined accuracy and computational-cost objective. The work begins with a large fully convolutional baseline, evaluates more modern alternatives, and finishes with a lightweight FastSCNN-style model designed to preserve useful spatial detail at a small fraction of the baseline's compute.

The task assigns one of 19 semantic classes to every image pixel. Performance is measured with mean Intersection over Union (mIoU), while computational cost is measured in GFLOPs.

## Final result

The selected FastSCNN model achieved:

- **0.4458 mIoU** (44.58%);
- **2.17 GFLOPs**;
- **1.58 million parameters**;
- an accuracy-efficiency score of approximately **20.54 mIoU-per-GFLOP units**.

The main improvement was efficiency: the original FCN baseline required 66.97 GFLOPs, while the final model used 2.17 GFLOPs.

## Experiment summary

| Model | mIoU | GFLOPs | Efficiency |
| --- | ---: | ---: | ---: |
| Baseline FCN | 0.284 | 66.97 | 0.42 |
| Baseline SegFormer | 0.689 | 15.52 | 4.44 |
| SegFormer pruned with knowledge distillation | 0.552 | 7.35 | 7.52 |
| **FastSCNN final model** | **0.445** | **2.17** | **20.54** |

SegFormer produced the highest raw mIoU, but FastSCNN provided the best reported accuracy-to-compute trade-off and was therefore selected as the final competition model.

## Dataset and task

The supplied road-scene dataset contains:

- 150 labeled training images;
- 50 testing images;
- 19 semantic classes, including road, sidewalk, building, vegetation, sky, person, rider, car, bus, train, motorcycle, and bicycle;
- images processed at approximately 375 × 1242 pixels.

The dataset is not included in this repository. The notebooks expect a structure similar to:

```text
seg_data/
├── colors.txt
├── training/
│   ├── image/
│   └── label/
└── testing/
    ├── image/
    └── label/
```

Update the notebook's `data_dir` or `dataFolder` variable to match the local or Kaggle dataset path.

## Final architecture

### Learning to Downsample

The first branch progressively reduces resolution from full size to 1/8:

```text
3 channels → 32 → 48 → 64
```

One standard convolution is followed by two depthwise-separable convolutions. This preserves more boundary information than repeatedly applying max pooling while reducing the cost of high-resolution processing.

### Global Feature Extractor

The semantic branch operates from 1/8 to 1/32 resolution. It uses MobileNetV2-style inverted residual blocks with channel growth from 64 to 96 to 128, followed by adaptive pooling for scene-level context.

### Feature Fusion

Low-resolution semantic features are bilinearly upsampled to the spatial branch's resolution. `1×1` projections align the channels, and element-wise addition combines fine boundaries with global meaning.

### Classifier

Two depthwise-separable convolutions refine the fused features. Dropout provides regularization, and a final `1×1` convolution produces logits for the 19 classes before full-resolution interpolation.

## Training approach

The final pipeline includes:

- random multi-scale training at 0.75×, 1.0×, and 1.25×;
- horizontal flips;
- brightness, contrast, and saturation jitter;
- Gaussian blur and gamma correction;
- mixed-precision training;
- weighted cross-entropy plus Dice loss in a **0.8:0.2** ratio;
- AdamW with weight decay;
- cosine-annealing learning-rate scheduling;
- gradient clipping;
- checkpointing and early stopping;
- evaluation every five epochs.

Relative to the baseline, batch size increased from 4 to 16 and the maximum channel width fell from 2,048 to 128.

## Repository contents

| File | Purpose |
| --- | --- |
| `A4_final_report.pdf` | Concise final report and ablation table |
| `A4_Road Segmentation_report.pdf` | Exported implementation/report notebook |
| `A4_Road Segmentation_report.ipynb` | Final FastSCNN architecture, training, inference, mIoU, and GFLOP calculations |
| `segmentation-experiments-1.ipynb` | SegFormer experiment and training outputs |
| `segmentation-experiments-2.ipynb` | Additional segmentation architecture/optimization experiments |

The final notebook expects `best_fastscnn_model.pth` when running evaluation-only cells. That checkpoint is not committed, so either train the model first or provide a compatible state dictionary.

## Running the project

Use Python 3 with:

```text
torch
torchvision
numpy
pillow
opencv-python
matplotlib
pandas
fvcore
transformers
```

A CUDA-capable GPU is recommended.

Suggested workflow:

1. Place the dataset in the expected folder structure.
2. Open `A4_Road Segmentation_report.ipynb`.
3. Set `data_dir` to the dataset location.
4. Run the architecture and training cells to create `best_fastscnn_model.pth`.
5. Run prediction generation and `cal_acc` to compute mIoU.
6. Run the FLOP calculation to reproduce the efficiency score.

The experiment notebooks document the path from the original FCN through SegFormer variants to the final lightweight architecture.

## Outputs

The implementation can produce:

- a best-model checkpoint;
- grayscale class-index prediction masks;
- colorized segmentation visualizations;
- training-loss and mIoU curves;
- per-class IoU values;
- GFLOP and efficiency calculations.

## Interpretation

The project demonstrates that the model with the best raw mIoU is not necessarily the best model under a deployment constraint. FastSCNN sacrifices some segmentation accuracy relative to SegFormer but greatly reduces computation through low-resolution semantic processing, narrow channel widths, depthwise convolutions, and targeted feature fusion.

## Limitations

- The dataset is very small, with only 150 training images.
- The reported test result is based on one competition dataset and does not establish broad driving-scene generalization.
- Element-wise feature fusion is simpler than learned attention-based fusion.
- mIoU does not separately measure boundary quality, rare-class performance, or temporal consistency.
- Depthwise convolutions are efficient in theory but may not be equally optimized on every device.
- The dataset and trained checkpoint are absent, so the repository is not immediately executable without external files.

## Authors

Kshitij Desai and Sakshi Vishwakarma, Computer Vision, University of Adelaide, 2025.
