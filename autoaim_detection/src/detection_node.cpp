#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoaim_interfaces/msg/detection_array.hpp>
#include <autoaim_interfaces/msg/comm_recv.hpp>

#include "infer_engine.hpp"
#include "antidart_detection.hpp"

namespace autoaim_detection {
float get_fps() {
    static auto prev = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float elapsed_sec = (now - prev).count() / 1e9;
    prev = now;
    return elapsed_sec == 0 ? 0 : 1 / elapsed_sec;
}

class YoloDetectNode: public rclcpp::Node {
public:
    YoloDetectNode(const rclcpp::NodeOptions& options): Node("autoaim_detection", options) {
        get_parameters();

        RCLCPP_INFO(this->get_logger(), "初始化YOLO armor...");
        const std::vector<std::string> armor_name = {
            "Sentry",
            "1",
            "2",
            "3",
            "4",
            "Outpost",
            "Base small",
            "Base big",
        };
        Config armor_config = {
            armor_model_path_, 
            4,
            3,
            8,
            4,
            2,
            armor_confidence_threshold_, 
            armor_nms_threshold_,
            armor_name
        };
        armor_infer_engine_ = create_infer_engine(armor_config);
        RCLCPP_INFO(this->get_logger(), "初始化YOLO armor完成");

        RCLCPP_INFO(this->get_logger(), "初始化YOLO buff...");
        const std::vector<std::string> buff_name = {
            "inactive",
            "active"
        };
        Config buff_config = {
            buff_model_path_, 
            4,
            2,
            2,
            4,
            2,
            buff_confidence_threshold_, 
            buff_nms_threshold_,
            buff_name
        };
        buff_infer_engine_ = create_infer_engine(buff_config);
        RCLCPP_INFO(this->get_logger(), "初始化YOLO buff完成");

        infer_engine_ = armor_infer_engine_;

        green_light_detector_ = std::make_shared<GreenLightDetector>(
            200.0f, 0.7f, 240 * 3 // minArea, minCircularity, minBrightness
        );

        // subscribe
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            img_topic_,
            rclcpp::SensorDataQoS().keep_last(1),
            std::bind(&YoloDetectNode::img_callback, this, std::placeholders::_1)
        );
        serial_sub_ = this->create_subscription<autoaim_interfaces::msg::CommRecv>(
            serial_topic_,
            rclcpp::SensorDataQoS().keep_last(1),
            std::bind(&YoloDetectNode::serial_callback, this, std::placeholders::_1)
        );
        // pub
        armor_send_pub_ = this->create_publisher<autoaim_interfaces::msg::DetectionArray>(
            armor_send_topic_,
            rclcpp::SensorDataQoS().keep_last(1)
        );
        buff_send_pub_ = this->create_publisher<autoaim_interfaces::msg::DetectionArray>(
            buff_send_topic_,
            rclcpp::SensorDataQoS().keep_last(1)
        );
        highshoot_send_pub_ = this->create_publisher<autoaim_interfaces::msg::DetectionArray>(
            highshoot_send_topic_,
            rclcpp::SensorDataQoS().keep_last(1)
        );
        antidart_send_pub_ = this->create_publisher<autoaim_interfaces::msg::DetectionArray>(
            antidart_send_topic_,
            rclcpp::SensorDataQoS().keep_last(1)
        );
        img_detected_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            img_detected_topic_,
            10
        );
    }

private:
    std::string img_topic_;
    std::string img_detected_topic_;
    std::string serial_topic_;

    std::string armor_send_topic_;
    std::string buff_send_topic_;
    std::string highshoot_send_topic_;
    std::string antidart_send_topic_;
    
    std::string armor_model_path_;
    float armor_confidence_threshold_;
    float armor_nms_threshold_;

    std::string buff_model_path_;
    float buff_confidence_threshold_;
    float buff_nms_threshold_;

    bool enable_detected_image_;
    bool enable_fps_;

    int mode_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Subscription<autoaim_interfaces::msg::CommRecv>::SharedPtr serial_sub_;
    rclcpp::Publisher<autoaim_interfaces::msg::DetectionArray>::SharedPtr armor_send_pub_;
    rclcpp::Publisher<autoaim_interfaces::msg::DetectionArray>::SharedPtr buff_send_pub_;
    rclcpp::Publisher<autoaim_interfaces::msg::DetectionArray>::SharedPtr highshoot_send_pub_;
    rclcpp::Publisher<autoaim_interfaces::msg::DetectionArray>::SharedPtr antidart_send_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_detected_pub_;

    std::shared_ptr<InferEngine> armor_infer_engine_, buff_infer_engine_;
    std::shared_ptr<InferEngine> infer_engine_;
    std::shared_ptr<GreenLightDetector> green_light_detector_;

    void get_parameters() {
        img_topic_ = declare_parameter<std::string>("image_topic", "/camera/image_raw");
        img_detected_topic_ =
            declare_parameter<std::string>("image_detected_topic", "/camera/color/image_detection");
        serial_topic_ = declare_parameter<std::string>("serial_topic", "/serial/comm_recv");
        
        armor_send_topic_ = declare_parameter<std::string>("armor_send_topic", "armor/detection");
        buff_send_topic_ = declare_parameter<std::string>("buff_send_topic", "buff/detection");
        highshoot_send_topic_ = declare_parameter<std::string>("highshoot_send_topic", "highshoot/detection");
        antidart_send_topic_ = declare_parameter<std::string>("antidart_send_topic", "antidart/detection");

        armor_model_path_ = ament_index_cpp::get_package_share_directory("autoaim_detection") + "/model/"
            + declare_parameter<std::string>("armor_model_path", "NULL");
        armor_confidence_threshold_ = declare_parameter<float>("armor_confidence_threshold", 0.5);
        armor_nms_threshold_ = declare_parameter<float>("armor_nms_threshold", 0.4);

        buff_model_path_ = ament_index_cpp::get_package_share_directory("autoaim_detection") + "/model/"
            + declare_parameter<std::string>("buff_model_path", "NULL");
        buff_confidence_threshold_ = declare_parameter<float>("buff_confidence_threshold", 0.5);
        buff_nms_threshold_ = declare_parameter<float>("buff_nms_threshold", 0.4);

        enable_detected_image_ = declare_parameter<bool>("enable_detected_image", false);
        enable_fps_ = declare_parameter<bool>("enable_fps", false);
        
    }

    void serial_callback(const autoaim_interfaces::msg::CommRecv::SharedPtr msg) {
        mode_ = msg->mode;
        //mode_ = 0x03;
        if(mode_ >= 0x03 && mode_ <= 0x06)
            infer_engine_ = buff_infer_engine_;
        else 
            infer_engine_ = armor_infer_engine_;
    }  

    void img_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (enable_fps_) {
            RCLCPP_INFO(get_logger(), "Detection FPS: %.0f", get_fps());
        }

        if(mode_ == 0x07){
            autoaim_interfaces::msg::DetectionArray detection_array;
            detection_array.header = msg->header;
            highshoot_send_pub_->publish(detection_array);
            return;
        }

        else if (mode_ == 0x08) {
            const auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            const cv::Mat img = cv_ptr->image;

            green_light_detector_->set_input_image(img);
            green_light_detector_->detect();
            auto detection_vec = green_light_detector_->get_detection_arr();

            autoaim_interfaces::msg::DetectionArray detection_array;
            detection_array.header = msg->header;
            detection_array.detections = detection_vec;

            antidart_send_pub_->publish(detection_array);

            if(enable_detected_image_) {
                sensor_msgs::msg::Image::SharedPtr img_detected =
                    cv_bridge::CvImage(msg->header, "bgr8", green_light_detector_->debug_draw_armors())
                        .toImageMsg();
                img_detected_pub_->publish(*img_detected);
            }

            return;
        }

        // RCLCPP_INFO(get_logger(), "detection mode: %d", mode_);
        const auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        const cv::Mat img = cv_ptr->image;

        // RCLCPP_INFO(get_logger(), "image size: %d x %d", img.cols, img.rows);
        infer_engine_->set_input_image(img);
        // RCLCPP_INFO(get_logger(), "image set");
        infer_engine_->preprocess();
        // RCLCPP_INFO(get_logger(), "image preprocess");
        infer_engine_->infer();
        // RCLCPP_INFO(get_logger(), "image infer");
        infer_engine_->postprocess();
        // RCLCPP_INFO(get_logger(), "image postprocess");
        auto detection_vec = infer_engine_->get_detection_arr();

        autoaim_interfaces::msg::DetectionArray detection_array;
        detection_array.header = msg->header;
        detection_array.detections = detection_vec;

        // RCLCPP_INFO(get_logger(), "detection size: %d", detection_vec.size());
        if(mode_ >= 0x03 && mode_ <= 0x06);
            // buff_send_pub_->publish(detection_array);
        else   
            armor_send_pub_->publish(detection_array);

        if (enable_detected_image_) {
            sensor_msgs::msg::Image::SharedPtr img_detected =
                cv_bridge::CvImage(msg->header, "bgr8", infer_engine_->debug_draw_armors())
                    .toImageMsg();
            img_detected_pub_->publish(*img_detected);
        }
    }
};
} // namespace autoaim_detection

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_detection::YoloDetectNode)