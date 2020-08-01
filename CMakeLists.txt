cmake_minimum_required(VERSION 3.16)
project(ComputerVision)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(OpenCV_DIR /Users/hwax0r/installation/OpenCV-master/lib/cmake/opencv4)
find_package( OpenCV REQUIRED )

# tell the build to include the headers from OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${OpenCV_LIBS} )

# specify the executable target to be built
add_executable(ComputerVision main.cpp moduleVideo.h)

# tell it to link the executable target against OpenCV
target_link_libraries(ComputerVision ${OpenCV_LIBS} )

