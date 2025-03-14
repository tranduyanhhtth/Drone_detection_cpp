cmake_minimum_required(VERSION 3.10)
project(yolov11_cpp_project)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR "/usr/local/onnxruntime-linux-x64-1.19.2")

include_directories(${ONNXRUNTIME_DIR}/include)

link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(yolov11_cpp_project 
    src/main.cpp 
    src/yolov11_onnx.cpp 
    src/bbox.cpp
    )

target_link_libraries(yolov11_cpp_project ${OpenCV_LIBS} onnxruntime)