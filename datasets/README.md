# Prepare Datasets for GEM

The dataset is assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  GSD-S/
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