# wave_simulation
wave simulation and visualization with python

video: https://www.youtube.com/watch?v=Yg7p70wOER4

## usage
* run
```code
> ./run.py
```
* record movie
```code
> ./run.py --record
```
* record full speed
```code
> ./run.py -f 0
```

## env
* python libraries -> requirements.txt
  * glumpy
  * pycuda
  * torch
  * etc...
* packages
  * nvidia-driver-450
  * cuda-cuda-toolkit-10-2
  * ffmpeg (for recording)

## reference
* [pytorch-glumpy.py](https://gist.github.com/victor-shepardson/5b3d3087dc2b4817b9bffdb8e87a57c4)
