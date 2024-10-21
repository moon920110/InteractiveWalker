# Guidelines
```angular2html
conda activate p36
```

## Calibration
Save calibration video at  `../calibration_data` with `.mkv` format.
```angular2html
python preprocessing.py --calibrate True
```
Run `calibrate.m`

![calibration](https://user-images.githubusercontent.com/46220978/209917779-4ec20054-b644-457a-940c-b23b251add52.png)
- Add images from `../calibration_data/YYYY-MM-DD/calibration/cam1/image`, `../calibration_data/YYYY-MM-DD/calibration/cam2/image`
- Size of checkerboard square : `80`
- Save from Workspace (`stereoParams_MMDD.mat`)

## Generate GT keypoints
```angular2html
python run_preprocessing.py --trim_video True
```
Pull [docker image](https://hub.docker.com/r/cwaffles/openpose)

```angular2html
sudo docker run --runtime=nvidia -it --rm cwaffles/openpose /bin/bash
```

```angular2html
docker cp ../data/ CONTAINER_ID:/openpose/
```

```angular2html
docker cp ./run_openpose.py CONTAINER_ID:/openpose/
```

```angular2html
# Docker
python run_openpose.py
```

```angular2html
python run_preprocessing.py --read_keypoints True
```
Run `run_StereoTriangulate.m` using camera parameters.

```angular2html
python run_preprocessing.py --trainable True
```
```angular2html
python make_dir --lunge True
python make_dir --all True
```