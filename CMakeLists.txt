cmake_minimum_required(VERSION 2.8.3)
project(semantic_segmentation_ros)

find_package(catkin_simple REQUIRED)

catkin_python_setup()
catkin_simple()

cs_install()
cs_export()
