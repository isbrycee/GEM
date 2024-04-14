# Prepare Datasets for GEM

The dataset is assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  GSD-S/
  TED3/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

[comment]: <> (The [model zoo]&#40;https://github.com/facebookresearch/MaskFormer/blob/master/MODEL_ZOO.md&#41;)

[comment]: <> (contains configs and models that use these builtin datasets.)

## Expected dataset structure for GSD-S:
```
└── datasets
    └── GSD-S
        ├── images
        │   ├── training
        │   └── validation
        ├── annotations
        │   ├── training
        │   └── validation
```

## Expected dataset structure for TED3:

```
datasets/
│
├── TED3-X-Ray-6k/
│   ├── images/
│   │   ├── training/
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   │   ├── validation/
│   │   │   ├── image3.jpg
│   │   │   └── ...
│   ├── annotations_zero_one/
│   │   ├── training/
│   │   │   ├── image1.png
│   │   │   └── image2.png
│   │   ├── validation/
│   │   │   ├── image3.png
│   │   │   └── ...
│   │   └── 
```

- If you want to change the image size in evaluzation, fixing here:

https://github.com/isbrycee/GEM-Glass-Segmentor/blob/main/detectron2-main/detectron2/data/detection_utils.py#L637

- If you want to change the structure of folers, pls fixing here:

https://github.com/isbrycee/GEM-Glass-Segmentor/blob/main/detectron2-main/detectron2/data/datasets/builtin.py#L237

- If you want to change your custom class name, pls fixing here:

https://github.com/isbrycee/GEM-Glass-Segmentor/blob/main/detectron2-main/detectron2/data/datasets/builtin_meta.py#L231

