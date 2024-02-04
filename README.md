# wave_simulation
wave simulation and visualization with python

video:
 * sim1: https://www.youtube.com/watch?v=Yg7p70wOER4
 * sim2: https://www.youtube.com/watch?v=MWfgh1KlJ7k
 * sim3: https://youtu.be/MGHZIDVWqqk
 * sim4: https://youtu.be/p7oMrNavxsY

# virtualenv
```
pyenv install 3.9.18
pyenv virtualenv 3.9.18 sim
pyenv local sim
pip install -r requirements.txt
```

# install pycuda
```
mkdir repos
cd repos
git clone https://github.com/inducer/pycuda.git
cd pycuda
python configure.py --cuda-enable-gl
sudo make install
```

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

## reference
* [pytorch-glumpy.py](https://gist.github.com/victor-shepardson/5b3d3087dc2b4817b9bffdb8e87a57c4)
