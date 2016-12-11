cmake_minimum_required(VERSION 2.8.3)
project(control)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
)
 find_package(OpenCV REQUIRED)

catkin_package()


include_directories(
  ${catkin_INCLUDE_DIRS}
)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(control src/main.cpp)
target_link_libraries(control ${catkin_LIBRARIES} )
 target_link_libraries(control ${OpenCV_LIBRARIES})
