#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <autoaim_interfaces/msg/detection_array.hpp>
#include <autoaim_interfaces/msg/detection.hpp>
#include <autoaim_interfaces/msg/comm_send.hpp>
#include <autoaim_interfaces/msg/comm_recv.hpp>

#include <pnp_solver.hpp>
#include <trajectory.hpp>

namespace autoaim_antidart {

using autoaim_interfaces::msg::Detection;
using autoaim_interfaces::msg::DetectionArray;
using autoaim_interfaces::msg::CommRecv;
using autoaim_interfaces::msg::CommSend;
using sensor_msgs::msg::CameraInfo;

    namespace math {
        constexpr float rad_period_correction(const float rad) {
            return rad + round((-rad) / (2 * M_PI)) * (2 * M_PI);
        }
        constexpr float r2d(const float rad) {
            return rad * 180.0 / M_PI;
        }
        constexpr float d2r(const float deg) {
            return deg * M_PI / 180.0;
        }
    } 

class AntiDartNode: public rclcpp::Node {
public:
    explicit AntiDartNode(const rclcpp::NodeOptions& options);
    ~AntiDartNode() = default;

private:
    void get_parameters();
    void send_dynamic_tf_transforms(rclcpp::Time time_stamp);
    void init_publishers();
    void init_subscriptions();
    void detection_callback(const DetectionArray::SharedPtr msg);
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    geometry_msgs::msg::Transform try_get_transform(
        const std::string& target,
        const std::string& source,
        const rclcpp::Time& time_point
    ) const;

    //获取串口节点信息，全是弧度制
    float roll_;                   
    float pitch_ ;                  
    float yaw_;
    float bullet_speed_;
    bool can_shoot_ = false;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    rclcpp::Subscription<DetectionArray>::SharedPtr detection_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<CommRecv>::SharedPtr comm_recv_sub_;

    rclcpp::Publisher<CommSend>::SharedPtr comm_send_pub_;

    std::unique_ptr<PnPSolver> pnp_solver_;

    float target_to_light_z_;
    float shoot_compensate_pitch_, shoot_compensate_yaw_;
};

AntiDartNode::AntiDartNode(const rclcpp::NodeOptions& options): Node("autoaim_antidart", options) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    pnp_solver_ = std::make_unique<PnPSolver>();

    get_parameters();
    init_publishers();
    init_subscriptions();
}

void AntiDartNode::get_parameters() {
    target_to_light_z_ = declare_parameter("target_to_light_z", 0.08);
    bullet_speed_ = declare_parameter("bullet_speed", 24.0);
    shoot_compensate_pitch_ = math::d2r(declare_parameter("shoot_compensate_pitch", 0.0));
    shoot_compensate_yaw_ = math::d2r(declare_parameter("shoot_compensate_yaw", 0.0));
}

void AntiDartNode::init_publishers() {
    std::string comm_send_pub_topic = declare_parameter("comm_send_pub_topic","/serial/comm_send");
    comm_send_pub_ = create_publisher<CommSend>(
        comm_send_pub_topic,
        rclcpp::SensorDataQoS().keep_last(1)
    );
}

void AntiDartNode::init_subscriptions() {
    std::string camera_info_topic =
        declare_parameter("camera_info_topic", "camera/color/camera_info");
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const sensor_msgs::msg::CameraInfo::SharedPtr msg) { camera_info_callback(msg); }
    );

    std::string detection_sub_topic = declare_parameter("detection_sub_topic", "detection");
    detection_sub_ = create_subscription<DetectionArray>(
        detection_sub_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const DetectionArray::SharedPtr msg) { detection_callback(msg); }
    );

    std::string comm_recv_sub_topic = declare_parameter("comm_recv_sub_topic","/serial/comm_recv");
    comm_recv_sub_ = create_subscription<CommRecv>(
        comm_recv_sub_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const CommRecv::SharedPtr msg) {
            roll_ = math::d2r(msg->roll);
            pitch_ = math::d2r(msg->pitch);
            yaw_ = math::d2r(msg->yaw);
            bullet_speed_ = msg->shoot_speed;
            send_dynamic_tf_transforms(msg->header.stamp);
        }
    );
}

void AntiDartNode::send_dynamic_tf_transforms(rclcpp::Time time_stamp){
    const auto time_compensate = rclcpp::Duration::from_seconds(0.02);

    geometry_msgs::msg::TransformStamped gimbal_to_chassis;
    gimbal_to_chassis.header.stamp = time_stamp + time_compensate;
    gimbal_to_chassis.header.frame_id = "chassis";
    gimbal_to_chassis.child_frame_id = "gimbal";
    gimbal_to_chassis.transform.translation.x = 0;
    gimbal_to_chassis.transform.translation.y = 0;
    gimbal_to_chassis.transform.translation.z = 0;
    tf2::Quaternion q_gimbal_chassis;
    q_gimbal_chassis.setRPY(roll_, pitch_, yaw_); 
    gimbal_to_chassis.transform.rotation.x = q_gimbal_chassis.getX();
    gimbal_to_chassis.transform.rotation.y = q_gimbal_chassis.getY();
    gimbal_to_chassis.transform.rotation.z = q_gimbal_chassis.getZ();
    gimbal_to_chassis.transform.rotation.w = q_gimbal_chassis.getW();
    tf_broadcaster_->sendTransform(gimbal_to_chassis);
}

void AntiDartNode::detection_callback(const DetectionArray::SharedPtr msg) {

    if (msg->detections.empty()) {
        autoaim_interfaces::msg::CommSend comm_send;
        //comm_send.header.stamp = now();
        comm_send.target_find = false;
        comm_send.shoot_flag = 0;
        comm_send.pitch = math::r2d(pitch_);
        comm_send.yaw = math::r2d(yaw_);
        comm_send_pub_->publish(comm_send); 
        return;
    }

    const std::vector<cv::Point2f> detected_pts = {
        cv::Point2f(msg->detections[0].tl.x, msg->detections[0].tl.y),
        cv::Point2f(msg->detections[0].bl.x, msg->detections[0].bl.y),
        cv::Point2f(msg->detections[0].br.x, msg->detections[0].br.y),
        cv::Point2f(msg->detections[0].tr.x, msg->detections[0].tr.y)
    };

    const cv::Point3f translation = pnp_solver_->get_translation(detected_pts);

    geometry_msgs::msg::TransformStamped dart_target_to_cam;
    dart_target_to_cam.header.stamp = msg->header.stamp;
    dart_target_to_cam.header.frame_id = "autoaim_camera";
    dart_target_to_cam.child_frame_id = "dart_target";
    dart_target_to_cam.transform.translation.x = translation.x;
    dart_target_to_cam.transform.translation.y = translation.y;
    dart_target_to_cam.transform.translation.z = translation.z + target_to_light_z_;
    tf_broadcaster_->sendTransform(dart_target_to_cam);

    geometry_msgs::msg::Transform dart_target_to_fric;
    try {
        dart_target_to_fric = try_get_transform("fric", "dart_target", msg->header.stamp);
    } catch (const std::exception& ex) {
        RCLCPP_WARN(
            get_logger(),
            "Failed to get transform from target to fric: %s",
            ex.what()
        );
        return;
    }

    const float target_pitch = - trajectory::calc_pitch(
        dart_target_to_fric.translation.x,
        dart_target_to_fric.translation.y,
        dart_target_to_fric.translation.z,
        bullet_speed_
    ) + shoot_compensate_pitch_;

    const float target_yaw = math::rad_period_correction(
        atan2(
            dart_target_to_fric.translation.y,
            dart_target_to_fric.translation.x
        ) + yaw_ + shoot_compensate_yaw_ );

    autoaim_interfaces::msg::CommSend comm_send;
    //comm_send.header.stamp = now();
    comm_send.target_find = true;
    comm_send.shoot_flag = can_shoot_? 2 : 0;
    comm_send.pitch = math::r2d(target_pitch);
    comm_send.yaw = math::r2d(target_yaw);
    comm_send_pub_->publish(comm_send);
    return ;
}

geometry_msgs::msg::Transform AntiDartNode::try_get_transform(
    const std::string& target,
    const std::string& source,
    const rclcpp::Time& time_point
) const {
    constexpr int MAX_ATTEMPTS = 100;
    geometry_msgs::msg::Transform transform;
    for (int i = 0; i < MAX_ATTEMPTS; i++) {
        try {
            transform = tf_buffer_->lookupTransform(target, source, time_point).transform;
            return transform;
        } catch (const std::exception& ex) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    throw std::runtime_error("try_get_transform failed after 100 attempts");
}

void AntiDartNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    pnp_solver_->set_cam_matrix(
        cv::Mat(3, 3, CV_64F, msg->k.data()),
        cv::Mat(1, 5, CV_64F, msg->d.data())
    );
    // 相机内参和畸变在运行中不会改变，所以设置后即可取消camera_info订阅
    camera_info_sub_.reset();
    camera_info_sub_ = nullptr;
}
} // namespace autoaim_antidart

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_antidart::AntiDartNode)