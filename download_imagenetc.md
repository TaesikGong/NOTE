1. Download original ImageNet: https://www.kaggle.com/c/imagenet-object-localization-challenge/data

2. Put the content of `ILSVRC` folder under `PROJECT_ROOT/dataset/ImageNet-C/origin/`:
- `PROJECT_ROOT/dataset/ImageNet-C/origin/Annotations`
- `PROJECT_ROOT/dataset/ImageNet-C/origin/Data`
- `PROJECT_ROOT/dataset/ImageNet-C/origin/ImageSets`

3. Download ImageNet-C: https://zenodo.org/records/2235448

4. Unzip each file and put each domain under `PROJECT_ROOT/dataset/ImageNet-C/corrupted`:
- `PROJECT_ROOT/dataset/ImageNet-C/corrupted/gaussian_noise`
- `PROJECT_ROOT/dataset/ImageNet-C/corrupted/gaussian_blur`
- ...
- `PROJECT_ROOT/dataset/ImageNet-C/corrupted/jpeg_compression/`


5. Process validation data

    python process_imagenet.py
