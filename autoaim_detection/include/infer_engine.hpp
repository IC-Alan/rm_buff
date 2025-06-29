#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include <autoaim_interfaces/msg/detection_array.hpp>

class InferEngine {
public:
    virtual ~InferEngine() = default;

    virtual void preprocess() = 0;
    virtual void infer() = 0;
    virtual void postprocess() = 0;

    virtual void set_input_image(const cv::Mat img) = 0;
    virtual std::vector<autoaim_interfaces::msg::Detection> get_detection_arr() const = 0;

    virtual cv::Mat debug_draw_armors() = 0;
};

#ifdef INFER_BACKEND_CUDA
#include <infer_cuda.hpp>
std::unique_ptr<InferEngine> create_infer_engine(Config config) {
    return std::unique_ptr<InferEngine>(new CUDAInferEngine(config));
}
#endif

#ifdef INFER_BACKEND_OPENVINO
#include <infer_openvino.hpp>
std::unique_ptr<InferEngine> create_infer_engine(Config config) {
    return std::unique_ptr<InferEngine>(new OpenVINOInferEngine(config));
}
#endif
