cmake_minimum_required(VERSION 3.12)
project(YOLOv7_CPP)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# Set ONNX Runtime paths (GPU version 1.15.1)
set(ONNXRUNTIME_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.15.1)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)

# Check if ONNX Runtime exists
if(NOT EXISTS ${ONNXRUNTIME_ROOT_PATH})
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT_PATH}")
endif()

# Set library path based on your system
if(EXISTS ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)
elseif(EXISTS ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so.1.15.1)
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so.1.16.3)
else()
    message(FATAL_ERROR "ONNX Runtime library not found")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

# Create executable
add_executable(yolo_inference main.cpp)

# Link libraries
target_link_libraries(yolo_inference 
    ${OpenCV_LIBS} 
    ${ONNXRUNTIME_LIB}
)

# Set runtime path so the executable can find the shared library
set_target_properties(yolo_inference PROPERTIES
    INSTALL_RPATH ${ONNXRUNTIME_ROOT_PATH}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
)
