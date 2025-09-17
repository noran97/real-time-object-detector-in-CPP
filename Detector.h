#ifndef DETECTOR_H
#define DETECTOR_H
using Array = std::vector<float>;
using Shape = std::vector<long>;

// Configuration

extern bool use_cuda;
extern int image_size;
extern std::string model_path;
extern std::string image_path;
extern std::string video_path;
extern std::string output_video_path;
extern float confidence_threshold;
extern float nms_threshold;
extern const char * class_names[];


// Processing mode
enum ProcessingMode {
    IMAGE_MODE,
    VIDEO_MODE,
};


struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
};
std::tuple<Array, Shape> read_image_for_inference(const cv::Mat &image, int size);
std::pair<Array, Shape> process_frame(Ort::Session &session, Array &array, Shape shape);
std::vector<Detection> parse_detections_yolov11(const Array &output, const Shape &shape, 
                                               int orig_width, int orig_height);
std::vector<Detection> apply_nms(const std::vector<Detection> &detections) ;
void draw_detections(cv::Mat &frame, const std::vector<Detection> &detections);
void process_image(Ort::Session &session);
void process_video(Ort::Session &session) ;
#endif