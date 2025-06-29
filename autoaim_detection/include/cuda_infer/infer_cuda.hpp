#pragma once

#include <omp.h>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include "autoaim_interfaces/msg/detection_array.hpp"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "cuda_fp16.h"
#include "logging.h"
#include "common.hpp"
#include "preprocess.h"
#include "cuda_utils.h"

#define DEVICE 0   // GPU id
#define MAX_IMAGE_INPUT_SIZE_THRESH 800 * 800 // maximum input size

enum COLOR
{
    GRAY = 0,
    BLUE = 1,
    RED = 2,
    PURPLE = 3
};

enum TAG
{
    SENTRY = 0,
    HERO = 1,
    ENGINEER = 2,
    INFANTRY_3 = 3,
    INFANTRY_4 = 4,
    OUTPOST = 5,
    BASE_SMALL = 6,
    BASE_BIG = 7
};

struct Config {
    std::string model_path;
    int dim_bbox;
    int num_color;
    int num_tag;
    int num_keypoints;
    int dim_keypoints;
    float conf_threshold;
    float nms_threshold;
    std::vector<std::string> tag_name;
};

class CUDAInferEngine: public InferEngine {
public:
    CUDAInferEngine(Config config);
    ~CUDAInferEngine();

    void set_input_image(const cv::Mat image) override;
    void blob_from_image(const cv::Mat &img);
    void preprocess() override;
    void infer() override;
    void generate_proposals(const float *infer_res_);
    void postprocess() override;
    
    std::vector<autoaim_interfaces::msg::Detection> get_detection_arr() const override;
    cv::Mat debug_draw_armors() override;

private:
    // model config
    const int DIM_BBOX;
    const int NUM_COLOR;
    const int NUM_TAG;
    const int NUM_KEYPOINTS;
    const int DIM_KEYPOINTS;
    const float conf_threshold;
    const float nms_threshold;
    const std::vector<std::string> name;

    // tensorRT related
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_;
    Logger logger;

    int input_size_;
    int output_size_;

    // CUDA memory iter
    void* device_buffers_[2];
    float* host_input_ = nullptr;
    float* host_output_ = nullptr;

    cv::Mat original_img;
    cv::Mat infer_img;

    std::vector<Object> proposals; 
    std::vector<autoaim_interfaces::msg::Detection> detection_arr;
};

CUDAInferEngine::CUDAInferEngine(Config config)
      : DIM_BBOX(config.dim_bbox), NUM_COLOR(config.num_color), NUM_TAG(config.num_tag), NUM_KEYPOINTS(config.num_keypoints), DIM_KEYPOINTS(config.dim_keypoints),
        conf_threshold(config.conf_threshold), nms_threshold(config.nms_threshold), name(config.tag_name) {
    
    CUDA_CHECK(cudaSetDevice(DEVICE));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    
    // load tensorRT model
    std::ifstream file(config.model_path, std::ios::binary);
    if (!file) throw std::runtime_error("Engine file not found");

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    // create tensorRT pointers
    runtime_ = nvinfer1::createInferRuntime(logger);
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    context_ = engine_->createExecutionContext();

    const auto input_dims = engine_->getTensorShape(engine_->getIOTensorName(0));
    input_size_ = 1;
    for (int j = 0; j < input_dims.nbDims; ++j) {
        input_size_ *= input_dims.d[j];
    }

    const auto output_dims = engine_->getTensorShape(engine_->getIOTensorName(1));
    output_size_ = 1;
    for (int j = 0; j < output_dims.nbDims; ++j) {
        output_size_ *= output_dims.d[j];
    }

    // malloc device memory
    CUDA_CHECK(cudaMalloc(&device_buffers_[0], input_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_buffers_[1], output_size_ * sizeof(float)));

    // malloc host memory
    CUDA_CHECK(cudaMallocHost((void**)&host_input_, input_size_ * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&host_output_, output_size_ * sizeof(float)));
}

CUDAInferEngine::~CUDAInferEngine() {
    // CUDA free
    CUDA_CHECK(cudaFree(device_buffers_[0]));
    CUDA_CHECK(cudaFree(device_buffers_[1]));
    CUDA_CHECK(cudaStreamDestroy(stream_));
  
    // delete tensorRT pointers
    delete context_;
    delete engine_;
    delete runtime_;
}

void CUDAInferEngine::set_input_image(const cv::Mat image){
    this->original_img = image;
}

void CUDAInferEngine::blob_from_image(const cv::Mat &img) {
    if (img.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    const auto input_dims = engine_->getTensorShape(engine_->getIOTensorName(0)); //[B, C, H, W]
    const int channels = input_dims.d[1];
    const int input_h = input_dims.d[2]; 
    const int input_w = input_dims.d[3];

    size_t img_size = img.rows * img.cols * channels;
    // size_t input_size = input_h * input_w * channels;

    // CUDA stream memcpy async
    void* img_device = nullptr;
    CUDA_CHECK(cudaMalloc(&img_device, img_size));
    CUDA_CHECK(cudaMemcpyAsync(host_input_, img.data, img_size, cudaMemcpyHostToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(img_device, host_input_, img_size, cudaMemcpyHostToDevice, stream_));

    if(img.cols != input_w || img.rows != input_h)
        std::cerr<<"通道不匹配！"<<std::endl;
    // image preprocess on CUDA
    preprocess_kernel_img(
        static_cast<u_int8_t*>(img_device),   
        img.cols,                           
        img.rows,                           
        static_cast<float*>(device_buffers_[0]), 
        input_w,                            
        input_h,                            
        stream_                             
    );

    CUDA_CHECK(cudaFree(img_device));
}

void CUDAInferEngine::preprocess() {
    infer_img = original_img;
    blob_from_image(infer_img);
}

void CUDAInferEngine::infer() {
    context_->setTensorAddress(engine_->getIOTensorName(0), device_buffers_[0]);
    context_->setTensorAddress(engine_->getIOTensorName(1), device_buffers_[1]);

    // inference
    context_->enqueueV3(stream_);

    // copy to host
    CUDA_CHECK(cudaMemcpyAsync(
        host_output_,               
        device_buffers_[1],         
        output_size_ * sizeof(float), 
        cudaMemcpyDeviceToHost,     
        stream_                     
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDAInferEngine::generate_proposals(const float* infer_res) {
    proposals.clear();
  
    int DETECTION_SIZE = DIM_BBOX + NUM_COLOR * NUM_TAG + NUM_KEYPOINTS * DIM_KEYPOINTS;  
    const int dets = output_size_ / DETECTION_SIZE;

    // process result paralleled (4 kernel for default)
    #pragma omp parallel for num_threads(4) schedule(guided)
    for (int i = 0; i < dets; ++i) {
        const int base_idx = i * DETECTION_SIZE;
        Object obj;

        // generate bbox (x, y, w, h)
        const float* bbox_ptr = &infer_res[base_idx];
        std::copy(bbox_ptr, bbox_ptr + 4, obj.bbox);

        // generate tag and color
        const float* cls_ptr = &bbox_ptr[4];
        const auto max_iter = std::max_element(cls_ptr, cls_ptr + NUM_COLOR * NUM_TAG);
        obj.conf = *max_iter;
        const int max_index = std::distance(cls_ptr, max_iter);

        if (obj.conf < conf_threshold) continue;

        obj.color = max_index / NUM_TAG;
        obj.class_id = max_index % NUM_TAG;

        // generate keypoints
        const float* kpts_ptr = &cls_ptr[NUM_COLOR * NUM_TAG];
        for (int k = 0; k < NUM_KEYPOINTS; ++k) {
            obj.kpts.emplace_back(
                keypoint{
                    cv::Point2f(
                        kpts_ptr[DIM_KEYPOINTS * k],
                        kpts_ptr[DIM_KEYPOINTS * k + 1]
                    )
                }
            );
        }

        #pragma omp critical
        proposals.emplace_back(std::move(obj));
    }
}

void CUDAInferEngine::postprocess(){
    // generate proposals
    generate_proposals(host_output_);

    // apply nms
    std::vector<int> picked;
    qsort_descent_inplace(proposals);
    nms(picked, proposals, nms_threshold);

    // generate detection
    int color_map[] = {COLOR::BLUE, COLOR::RED, COLOR::GRAY};
    detection_arr.clear();    
    #pragma omp parallel for num_threads(4) schedule(guided)
    for (const auto& i : picked) {
        autoaim_interfaces::msg::Detection detection;
        detection.color = color_map[proposals[i].color];
        detection.label = proposals[i].class_id;
        detection.confidence = proposals[i].conf;
        detection.tl.x = proposals[i].kpts[0].pt.x; detection.tl.y = proposals[i].kpts[0].pt.y;
        detection.bl.x = proposals[i].kpts[1].pt.x; detection.bl.y = proposals[i].kpts[1].pt.y;
        detection.br.x = proposals[i].kpts[2].pt.x; detection.br.y = proposals[i].kpts[2].pt.y;
        detection.tr.x = proposals[i].kpts[3].pt.x; detection.tr.y = proposals[i].kpts[3].pt.y;
        #pragma omp critical
        detection_arr.push_back(detection);
    }
}

cv::Mat CUDAInferEngine::debug_draw_armors() {
    
    cv::Mat debug_img = infer_img.clone();

    const std::vector<cv::Scalar> kpt_colors = {
        cv::Scalar(255, 0, 0),    
        cv::Scalar(0, 255, 0),    
        cv::Scalar(0, 0, 255),    
        cv::Scalar(255, 255, 0)   
    };

    for (const auto& detection : detection_arr) {
        std::array<cv::Point2f, 4> kpts = {
            cv::Point2f(detection.tl.x, detection.tl.y),
            cv::Point2f(detection.bl.x, detection.bl.y),
            cv::Point2f(detection.br.x, detection.br.y),
            cv::Point2f(detection.tr.x, detection.tr.y)
        };

        for (size_t i = 0; i < kpts.size(); ++i) {
            cv::circle(debug_img, kpts[i], 5, kpt_colors[i], -1);
        }

        const cv::Scalar box_color = detection.color == COLOR::BLUE ? cv::Scalar(255, 0, 0) :
                                    detection.color == COLOR::RED ? cv::Scalar(0, 0, 255) :  
                                    cv::Scalar(114, 114, 114);
                                  
        for (int j = 0; j < 4; ++j) {
            cv::line(debug_img, kpts[j], kpts[(j + 1) % 4], box_color, 2);
        }

        cv::line(debug_img, kpts[0], kpts[2], box_color, 1);
        cv::line(debug_img, kpts[1], kpts[3], box_color, 1);

        const std::string label = name[detection.label] + " " + 
                                std::to_string(detection.confidence).substr(0, 4);
        cv::putText(debug_img, label, kpts[0] - cv::Point2f(0, 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }

    return debug_img;
}

std::vector<autoaim_interfaces::msg::Detection> CUDAInferEngine::get_detection_arr() const {
    return detection_arr;
}