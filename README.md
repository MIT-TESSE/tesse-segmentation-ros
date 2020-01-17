# semantic-segmentation-ros

A package for running semantic segmentation networks in ROS using TensorRT.

<div align="center">
  <img src="docs/tesse-semantic-segmentation.gif">
</div>

## Installation

### TensorRT
This requires TensorRT 6.0.0 or greater. To install via a tar file 

* Download a TensorRT tar file compatible with your machine through the [NVIDIA TensorRT page](https://developer.nvidia.com/tensorrt)
* Unpack the tar file into a directory, `<TRT_DIR>`
* Add the absolute path to the TensorRT `lib` directory to your `LD_LIBRARY_PATH`
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TRT_DIR>/lib
```
* Install the Python wheel file matching your Python version.
```sh
cd <TRT_DIR>/python
sudo pip install tensorrt-<TRT_VERSION>-cp<PYTHON_VERSION>-none-linux_x86_64.whl
```

### Install PyCUDA
```sh
pip install pycuda
```

### Clone and build this repo
```sh
# setup catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init

cd src 
git clone git@github.mit.edu:TESS/semantic-segmentation-ros.git
cd ..

# install dependencies
wstool init
wstool merge gym-ros-interface/install/gym_ros_interface.rosinstall 
cd ..

# compile
catkin build

# source workspace
source ~/catkin_ws/devel/setup.bash
```

## Usage
To run in the lastest TESSE build (v5.3), use the provided launch file
```
roslaunch semantic_segmentation_ros semantic_segmentation_tesse.launch 
```


## Disclaimer

Distribution authorized to U.S. Government agencies and their contractors. Other requests for this document shall be referred to the MIT Lincoln Laboratory Technology Office.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2019 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
