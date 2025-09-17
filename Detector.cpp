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
bool use_cuda = false;  // Use CPU for now
int image_size = 640;
std::string model_path = "/home/nor/projects/yolov7/yolo-cxx/yolo11n.onnx";  
std::string image_path = "/home/nor/projects/yolov7/images (2).jpeg";
std::string video_path = "/home/nor/projects/yolov7/2053100-uhd_3840_2160_30fps.mp4";  
std::string output_video_path = "/home/nor/projects/yolov7/output.mp4";  
// Detection parameters
float confidence_threshold = 0.5;
float nms_threshold = 0.4;
const char *class_names[] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
};

std::tuple<Array, Shape> read_image_for_inference(const cv::Mat &image, int size) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, {size, size});
    Shape shape = {1, resized_image.channels(), resized_image.rows, resized_image.cols};
    cv::Mat im = cv::dnn::blobFromImage(resized_image, 1.0 / 255.0, {}, {}, true);
    Array array(im.ptr<float>(), im.ptr<float>() + im.total());
    return {array, shape};
}
std::pair<Array, Shape> process_frame(Ort::Session &session, Array &array, Shape shape) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto input = Ort::Value::CreateTensor<float>(
        memory_info, (float *)array.data(), array.size(), shape.data(), shape.size());

    const char *input_names[] = {"images"};
    const char *output_names[] = {"output0"};
    auto output = session.Run({}, input_names, &input, 1, output_names, 1);

    // Get the full shape from the output tensor
    Shape output_shape = output[0].GetTensorTypeAndShapeInfo().GetShape();

    // The total number of elements in the output tensor
    size_t total_elements = 1;
    for(long dim : output_shape) {
        total_elements *= dim;
    }

    // Get the data pointer and copy the full tensor to your Array
    auto ptr = output[0].GetTensorData<float>();

    // Return the full array and the correct shape
    return {Array(ptr, ptr + total_elements), output_shape};
}
std::vector<Detection> parse_detections_yolov11(const Array &output, const Shape &shape, 
                                               int orig_width, int orig_height) {
    std::vector<Detection> detections;
    
    std::cout << "=== YOLOV11 DEBUG INFO ===" << std::endl;
    std::cout << "Output shape: [";
    for (long dim : shape) {
        std::cout << dim << ", ";
    }
    std::cout << "]" << std::endl;

    // Expected shape: [1, 84, 8400]
    if (shape.size() != 3) {
        std::cout << "Unexpected output shape dimensions" << std::endl;
        return detections;
    }

    int num_values = shape[1];
    int num_detections = shape[2]; 
    
    if (num_values < 84) {
        std::cout << "Unexpected number of values per detection: " << num_values << std::endl;
        return detections;
    }
    
    std::cout << "Number of detections: " << num_detections << std::endl;
    std::cout << "Values per detection: " << num_values << std::endl;
    
    // Collect all detections before NMS
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    
    for (int i = 0; i < num_detections; i++) {
        // In YOLOv11, the output is transposed: [84, 8400]
        // So for detection i, the data starts at output[0*8400 + i], output[1*8400 + i], etc.
        
        // Extract bbox coordinates (center_x, center_y, width, height)
        float center_x = output[0 * num_detections + i];
        float center_y = output[1 * num_detections + i];
        float width = output[2 * num_detections + i];
        float height = output[3 * num_detections + i];
        
        // Find the class with maximum score
        float max_score = 0.0f;
        int best_class_id = -1;
        
        for (int c = 0; c < 80; c++) {  // 80 COCO classes
            float class_score = output[(4 + c) * num_detections + i];
            if (class_score > max_score) {
                max_score = class_score;
                best_class_id = c;
            }
        }
        
        // Debug: Print first few detections
        if (i < 3) {
            std::cout << "Detection " << i << ": center_x=" << center_x << ", center_y=" << center_y 
                      << ", w=" << width << ", h=" << height << ", max_score=" << max_score 
                      << ", class=" << best_class_id << std::endl;
        }
        
        // Apply confidence threshold
        if (max_score < confidence_threshold) continue;
        
        // Convert from center format to corner format
        float x1 = center_x - width / 2.0f;
        float y1 = center_y - height / 2.0f;
        float x2 = center_x + width / 2.0f;
        float y2 = center_y + height / 2.0f;
        
        // Scale coordinates to original image size
        // YOLOv11 outputs are typically in the range [0, input_size]
        float scale_x = static_cast<float>(orig_width) / image_size;
        float scale_y = static_cast<float>(orig_height) / image_size;
        
        x1 *= scale_x;
        y1 *= scale_y;
        x2 *= scale_x;
        y2 *= scale_y;
        
        // Convert to integers and clamp to image boundaries
        int ix1 = std::max(0, std::min(static_cast<int>(x1), orig_width - 1));
        int iy1 = std::max(0, std::min(static_cast<int>(y1), orig_height - 1));
        int ix2 = std::max(0, std::min(static_cast<int>(x2), orig_width - 1));
        int iy2 = std::max(0, std::min(static_cast<int>(y2), orig_height - 1));
        
        // Validate bounding box
        if (ix2 > ix1 && iy2 > iy1 && best_class_id >= 0) {
            cv::Rect bbox(ix1, iy1, ix2 - ix1, iy2 - iy1);
            boxes.push_back(bbox);
            scores.push_back(max_score);
            class_ids.push_back(best_class_id);
        }
    }
    
    std::cout << "Valid detections before NMS: " << boxes.size() << std::endl;
    
    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, nms_threshold, indices);
    
    std::cout << "Valid detections after NMS: " << indices.size() << std::endl;
    
    // Create final detections
    for (int idx : indices) {
        Detection detection;
        detection.bbox = boxes[idx];
        detection.confidence = scores[idx];
        detection.class_id = class_ids[idx];
        detections.push_back(detection);
        
        std::cout << "Final detection: class=" << class_ids[idx] 
                  << " (" << class_names[class_ids[idx]] << "), conf=" << scores[idx]
                  << ", bbox=[" << boxes[idx].x << "," << boxes[idx].y << ","
                  << boxes[idx].width << "," << boxes[idx].height << "]" << std::endl;
    }
    
    return detections;
}

std::vector<Detection> apply_nms(const std::vector<Detection> &detections) {
    if (detections.empty()) return {};
    
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    
    for (const auto &det : detections) {
        boxes.push_back(det.bbox);
        scores.push_back(det.confidence);
    }
    
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, nms_threshold, indices);
    
    std::vector<Detection> nms_detections;
    for (int idx : indices) {
        nms_detections.push_back(detections[idx]);
    }
    
    return nms_detections;
}

void draw_detections(cv::Mat &frame, const std::vector<Detection> &detections) {
    for (const auto &detection : detections) {
        // Generate color based on class_id
        cv::Scalar color(
            (detection.class_id * 50) % 255,
            (detection.class_id * 80) % 255,
            (detection.class_id * 120) % 255
        );
        
        // Draw bounding box
        cv::rectangle(frame, detection.bbox, color, 2);
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << detection.confidence;
        
        // Get the formatted confidence string
        std::string confidence_str = ss.str();
        // Prepare label
        std::string label = class_names[detection.class_id]+confidence_str;
        
        // Get text size for background
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        // Draw background rectangle for text
        cv::Point text_origin(detection.bbox.x, detection.bbox.y - 5);
        cv::rectangle(frame, 
                     cv::Point(text_origin.x, text_origin.y - text_size.height - baseline),
                     cv::Point(text_origin.x + text_size.width, text_origin.y + baseline),
                     color, -1);
        
        // Draw text
        cv::putText(frame, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                   cv::Scalar(255, 255, 255), 1);
    }
}

void process_image(Ort::Session &session) {
    std::cout << "Processing image: " << image_path << std::endl;
    
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not load image " << image_path << std::endl;
        return;
    }
    
    // Prepare input for inference
    auto [input_array, input_shape] = read_image_for_inference(image, image_size);
    
    // Run inference
    auto [output_array, output_shape] = process_frame(session, input_array, input_shape);
    
    // Parse detections
    auto detections = parse_detections_yolov11(output_array, output_shape, image.cols, image.rows);
    
    // Apply NMS
    detections = apply_nms(detections);
    
    // Draw detections
    draw_detections(image, detections);
    
    std::cout << "Found " << detections.size() << " objects" << std::endl;
    
    // Display result
    cv::imshow("YOLOv11 Image Detection", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void process_video(Ort::Session &session) {
    std::cout << "Processing video: " << video_path << std::endl;
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video " << video_path << std::endl;
        return;
    }
    
    // Get video properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "Video properties: " << frame_width << "x" << frame_height 
              << " @ " << fps << " FPS, " << total_frames << " frames" << std::endl;
    
    // Setup video writer
    cv::VideoWriter writer(output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                          fps, cv::Size(frame_width, frame_height));
    
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video writer" << std::endl;
        return;
    }
    
    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (cap.read(frame)) {
        frame_count++;
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Prepare input for inference
        auto [input_array, input_shape] = read_image_for_inference(frame, image_size);
        
        // Run inference
        auto [output_array, output_shape] = process_frame(session, input_array, input_shape);
        
        // Parse detections
        auto detections = parse_detections_yolov11(output_array, output_shape, frame.cols, frame.rows);
        
        // Apply NMS
        detections = apply_nms(detections);
        
        // Draw detections
        draw_detections(frame, detections);
        
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
        
        // Add FPS and detection count to frame
        std::string info = "Frame: " + std::to_string(frame_count) + "/" + std::to_string(total_frames) +
                          " | Objects: " + std::to_string(detections.size()) +
                          " | Time: " + std::to_string(frame_duration.count()) + "ms";
        
        cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                   cv::Scalar(0, 255, 0), 2);
        
        // Write frame to output video
        writer.write(frame);
        
        // Display frame (optional - comment out for faster processing)
        cv::imshow("YOLOv11 Video Detection", frame);
        if (cv::waitKey(1) == 27) break; // ESC key to exit
        
        // Print progress every 30 frames
        if (frame_count % 30 == 0) {
            double progress = (double)frame_count / total_frames * 100;
            std::cout << "Progress: " << std::fixed << std::setprecision(1) 
                      << progress << "% (" << frame_count << "/" << total_frames << ")" << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "Video processing completed!" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl;
    std::cout << "Average FPS: " << frame_count / total_duration.count() << std::endl;
    std::cout << "Output saved to: " << output_video_path << std::endl;
    
    cap.release();
    writer.release();
    cv::destroyAllWindows();
}

