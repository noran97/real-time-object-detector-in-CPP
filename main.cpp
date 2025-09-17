#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <iostream>
#include <tuple>
#include <cassert>
#include <chrono>
#include "/home/nor/projects/yolov7/yolo-cxx/Dectector.h"

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        ProcessingMode mode = IMAGE_MODE;
        if (argc > 1) {
            std::string arg = argv[1];
            if (arg == "-v") {
                mode = VIDEO_MODE;
            } else if ( arg == "-i") {
                mode = IMAGE_MODE;
            } else {
                std::cout << "Usage: " << argv[0] << " [--image|-i] [--video|-v] " << std::endl;
                std::cout << "  --image, -i:  Process single image (default)" << std::endl;
                std::cout << "  --video, -v:  Process video file" << std::endl;
                
                return 0;
            }
        }
        
        std::cout << "Initializing YOLOv11 inference..." << std::endl;
        
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv11");
        Ort::SessionOptions options;
        
       
            std::cout << "Using CPU execution provider..." << std::endl;
        
        
        // Load the ONNX model
        std::cout << "Loading model from: " << model_path << std::endl;
        Ort::Session session(env, model_path.c_str(), options);
        
        // Process based on mode
        switch (mode) {
            case IMAGE_MODE:
                process_image(session);
                break;
            case VIDEO_MODE:
                process_video(session);
                break;
        }
        
        std::cout << "Detection completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}