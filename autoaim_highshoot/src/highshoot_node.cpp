#include <string>
#include <vector>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <autoaim_interfaces/msg/detection_array.hpp>
#include <autoaim_interfaces/msg/comm_send.hpp>
#include <autoaim_interfaces/msg/comm_recv.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <trajectory.hpp>

namespace autoaim_highshoot {

enum COLOR
{
    GRAY = 0,
    BLUE = 1,
    RED = 2,
    PURPLE = 3,
};

using namespace std;
using autoaim_interfaces::msg::Detection;
using autoaim_interfaces::msg::DetectionArray;
using autoaim_interfaces::msg::CommRecv;
using autoaim_interfaces::msg::CommSend;
using nav_msgs::msg::Odometry;

const geometry_msgs::msg::Transform EMPTY_TRANSFORM;

double to_sec(builtin_interfaces::msg::Time t) {
    return t.sec + t.nanosec * 1e-9;
}

class HighshootNode: public rclcpp::Node {
public:
    explicit HighshootNode(const rclcpp::NodeOptions& options);
    ~HighshootNode() = default;

private:
    void get_parameters();
    void highshoot_callback(const DetectionArray::SharedPtr msg);
    void get_VTM_camera_param();
    cv::Point2f get_pretiction_VTM(const builtin_interfaces::msg::Time& header_stamp);

    cv::Mat VTM_intrinsic_, VTM_distortion_;

    // 尝试获取指定时间点对应的变换。若尝试MAX_ATTEMPTS后仍没有找到，throw一个std::runtime_error
    geometry_msgs::msg::Transform try_get_transform(
        const std::string& target,
        const std::string& source,
        const rclcpp::Time& time_point
    ) const;

    // 获取指定时间的自己云台的yaw, pitch, roll（相对于chassis）。
    // 之所以是ypr不是rpy，是因为我们采用的旋转顺序是yaw, pitch, roll。
    std::tuple<float, float, float> get_gimbal_ypr(const rclcpp::Time& time_point) const;

    //获取串口节点信息
    float roll_;                   
    float pitch_;                  
    float yaw_;                               
    int mode_;                                
    int target_color_;
    float bullet_speed_;

    float control_to_fire_time_;
    float shoot_compensate_pitch_;
    float shoot_compensate_yaw_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    std::shared_ptr<rclcpp::Subscription<DetectionArray>> detection_sub_;
    std::shared_ptr<rclcpp::Subscription<CommRecv>> comm_recv_sub_;
    std::shared_ptr<rclcpp::Publisher<CommSend>> comm_send_pub_;
    std::shared_ptr<rclcpp::Publisher<Odometry>> localization_sub_;
};

HighshootNode::HighshootNode(const rclcpp::NodeOptions& options):
    Node("autoaim_highshoot", options) {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    get_parameters();
}

void HighshootNode::get_parameters() {
    this->get_VTM_camera_param();

    target_color_ = declare_parameter("target_color", 0);
    bullet_speed_ = declare_parameter("bullet_speed", 30.0);
    shoot_compensate_pitch_ = math::d2r(declare_parameter("shoot_compensate_pitch", 0.0));
    shoot_compensate_yaw_ = math::d2r(declare_parameter("shoot_compensate_yaw", 0.0));

    const double lidar_to_gimbal_x = declare_parameter("lidar_to_gimbal_x", 0.0);
    const double lidar_to_gimbal_y = declare_parameter("lidar_to_gimbal_y", 0.0);
    const double lidar_to_gimbal_z = declare_parameter("lidar_to_gimbal_z", 0.0);
    const double lidar_to_gimbal_qx = declare_parameter("lidar_to_gimbal_qx", 0.0);
    const double lidar_to_gimbal_qy = declare_parameter("lidar_to_gimbal_qy", 0.0);
    const double lidar_to_gimbal_qz = declare_parameter("lidar_to_gimbal_qz", 0.0);
    const double lidar_to_gimbal_qw = declare_parameter("lidar_to_gimbal_qw", 1.0);
    geometry_msgs::msg::TransformStamped lidar_to_gimbal;
    lidar_to_gimbal.header.stamp = this->now();
    lidar_to_gimbal.header.frame_id = "gimbal";
    lidar_to_gimbal.child_frame_id = "lidar_link";
    lidar_to_gimbal.transform.translation.x = lidar_to_gimbal_x;
    lidar_to_gimbal.transform.translation.y = lidar_to_gimbal_y;
    lidar_to_gimbal.transform.translation.z = lidar_to_gimbal_z;
    lidar_to_gimbal.transform.rotation.x = lidar_to_gimbal_qx;
    lidar_to_gimbal.transform.rotation.y = lidar_to_gimbal_qy;
    lidar_to_gimbal.transform.rotation.z = lidar_to_gimbal_qz;
    lidar_to_gimbal.transform.rotation.w = lidar_to_gimbal_qw;
    static_tf_broadcaster_->sendTransform(lidar_to_gimbal);

    const double blue_base_to_odom_x = declare_parameter("blue_base_to_odom_x", 0.0);
    const double blue_base_to_odom_y = declare_parameter("blue_base_to_odom_y", 0.0);
    const double blue_base_to_odom_z = declare_parameter("blue_base_to_odom_z", 0.0);
    geometry_msgs::msg::TransformStamped blue_base_to_odom;
    blue_base_to_odom.header.stamp = this->now();
    blue_base_to_odom.header.frame_id = "lidar_init";
    blue_base_to_odom.child_frame_id = "blue_base";
    blue_base_to_odom.transform.translation.x = blue_base_to_odom_x;
    blue_base_to_odom.transform.translation.y = blue_base_to_odom_y;
    blue_base_to_odom.transform.translation.z = blue_base_to_odom_z;
    blue_base_to_odom.transform.rotation.x = 0;
    blue_base_to_odom.transform.rotation.y = 0;
    blue_base_to_odom.transform.rotation.z = 0;
    blue_base_to_odom.transform.rotation.w = 1;
    static_tf_broadcaster_->sendTransform(blue_base_to_odom);

    const double red_base_to_odom_x = declare_parameter("red_base_to_odom_x", 0.0);
    const double red_base_to_odom_y = declare_parameter("red_base_to_odom_y", 0.0);
    const double red_base_to_odom_z = declare_parameter("red_base_to_odom_z", 0.0);
    geometry_msgs::msg::TransformStamped red_base_to_odom;
    red_base_to_odom.header.stamp = this->now();
    red_base_to_odom.header.frame_id = "lidar_init";
    red_base_to_odom.child_frame_id = "red_base";
    red_base_to_odom.transform.translation.x = red_base_to_odom_x;
    red_base_to_odom.transform.translation.y = red_base_to_odom_y;
    red_base_to_odom.transform.translation.z = red_base_to_odom_z;
    red_base_to_odom.transform.rotation.x = 0;
    red_base_to_odom.transform.rotation.y = 0;
    red_base_to_odom.transform.rotation.z = 0;
    red_base_to_odom.transform.rotation.w = 1;
    static_tf_broadcaster_->sendTransform(red_base_to_odom);

    const double blue_outpost_to_odom_x = declare_parameter("blue_outpost_to_odom_x", 0.0);
    const double blue_outpost_to_odom_y = declare_parameter("blue_outpost_to_odom_y", 0.0);
    const double blue_outpost_to_odom_z = declare_parameter("blue_outpost_to_odom_z", 0.0);
    geometry_msgs::msg::TransformStamped blue_outpost_to_odom;
    blue_outpost_to_odom.header.stamp = this->now();
    blue_outpost_to_odom.header.frame_id = "lidar_init";
    blue_outpost_to_odom.child_frame_id = "blue_outpost";
    blue_outpost_to_odom.transform.translation.x = blue_outpost_to_odom_x;
    blue_outpost_to_odom.transform.translation.y = blue_outpost_to_odom_y;
    blue_outpost_to_odom.transform.translation.z = blue_outpost_to_odom_z;
    blue_outpost_to_odom.transform.rotation.x = 0;
    blue_outpost_to_odom.transform.rotation.y = 0;
    blue_outpost_to_odom.transform.rotation.z = 0;
    blue_outpost_to_odom.transform.rotation.w = 1;
    static_tf_broadcaster_->sendTransform(blue_outpost_to_odom);

    const double red_outpost_to_odom_x = declare_parameter("red_outpost_to_odom_x", 0.0);
    const double red_outpost_to_odom_y = declare_parameter("red_outpost_to_odom_y", 0.0);
    const double red_outpost_to_odom_z = declare_parameter("red_outpost_to_odom_z", 0.0);
    geometry_msgs::msg::TransformStamped red_outpost_to_odom;
    red_outpost_to_odom.header.stamp = this->now();
    red_outpost_to_odom.header.frame_id = "lidar_init";
    red_outpost_to_odom.child_frame_id = "red_outpost";
    red_outpost_to_odom.transform.translation.x = red_outpost_to_odom_x;
    red_outpost_to_odom.transform.translation.y = red_outpost_to_odom_y;
    red_outpost_to_odom.transform.translation.z = red_outpost_to_odom_z;
    red_outpost_to_odom.transform.rotation.x = 0;
    red_outpost_to_odom.transform.rotation.y = 0;
    red_outpost_to_odom.transform.rotation.z = 0;
    red_outpost_to_odom.transform.rotation.w = 1;
    static_tf_broadcaster_->sendTransform(red_outpost_to_odom);

    std::string detection_sub_topic = declare_parameter("detection_sub_topic", "detection");
    std::string comm_recv_sub_topic = declare_parameter("comm_recv_sub_topic","/serial/comm_recv");
    std::string comm_send_pub_topic = declare_parameter("comm_send_pub_topic","/serial/comm_send");

    detection_sub_ = create_subscription<DetectionArray>(
        detection_sub_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const DetectionArray::SharedPtr msg) { detection_callback(msg); }
    );
    
    comm_recv_sub_ = create_subscription<CommRecv>(
        comm_recv_sub_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const CommRecv::SharedPtr msg) {
            roll_ = msg->roll;
            pitch_ =  msg->pitch;
            yaw_ = msg->yaw;
            bullet_speed_ = msg->shoot_speed;
            mode_ =  msg->mode;
            target_color_  = msg->target_color;
            is_aiming_ =  mode_ + 1;
        }
    );
    
    comm_send_pub_ = create_publisher<CommSend>(
        comm_send_pub_topic,
        rclcpp::SensorDataQoS().keep_last(1)
    );
}

void HighshootNode::highshoot_callback(const DetectionArray::SharedPtr msg) {
    if(target_color_ == COLOR::BLUE){
        try {
            auto armor_to_chassis = try_get_transform(
                "fric",
                "fric",
                msg->header.stamp
            );
            tracker_->push(armor_to_chassis);
        } catch (const std::exception& ex) {
            RCLCPP_WARN(
                get_logger(),
                "Failed to get transform from %s to chassis: %s",
                armor_name.c_str(),
                ex.what()
            );
        }
    }
    else if(target_color_ == COLOR::RED){

    }
}

std::tuple<float, float, float> HighshootNode::get_gimbal_ypr(const rclcpp::Time& time_point) const {
    // 保存之前找过的ypr，在lookupTransform出现异常时返回
    static std::tuple<float, float, float> prev_ypr = std::make_tuple(0, 0, 0);
    geometry_msgs::msg::Transform transform;
    try {
        transform = tf_buffer_->lookupTransform("chassis", "gimbal_pitch", time_point).transform;
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(),
            "Failed to get transform from gimbal_pitch to chassis: %s",
            ex.what()
        );
        return prev_ypr;
    }
    double yaw, pitch, roll;
    tf2::Quaternion quat(
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w
    );
    tf2::Matrix3x3 rot_mat(quat);
    rot_mat.getEulerYPR(yaw, pitch, roll);
    prev_ypr = std::make_tuple(yaw, pitch, roll);
    return std::make_tuple(yaw, pitch, roll);
}

geometry_msgs::msg::Transform HighshootNode::try_get_transform(
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
    throw std::runtime_error("try_get_transform failed after 1000 attempts");
}

void HighshootNode::get_VTM_camera_param(){
    cv::FileStorage cameraConfig(ament_index_cpp::get_package_share_directory("autoaim_prediction") + 
    "/config/VTM_camera_params.yaml", cv::FileStorage::READ);
    cameraConfig["camera_CI_MAT"] >> this->VTM_intrinsic_;
    cameraConfig["camera_D_MAT"] >> this->VTM_distortion_ ;
}

cv::Point2f HighshootNode::get_pretiction_VTM(const builtin_interfaces::msg::Time& header_stamp){
    geometry_msgs::msg::Transform VTM_to_predicted_transform;
    constexpr int MAX_ATTEMPTS = 1000;
    try {
        VTM_to_predicted_transform = this->try_get_transform(
            "VTM",
            "target",
            header_stamp
        );
    } catch (const std::exception& ex) {
        RCLCPP_WARN(
            get_logger(),
            "Failed to get transform from target to VTM: %s",
            ex.what()
        );
        return cv::Point2f(0.0,0.0);
    }
    cv::Point3f predicted(0.0,0.0,0.0);
    std::vector<cv::Point3f> pw={predicted};
    std::vector<cv::Point2f> projectedPoint;
    cv::Mat tVec = (cv::Mat_<double>(3, 1) << VTM_to_predicted_transform.translation.x,
                    VTM_to_predicted_transform.translation.y,VTM_to_predicted_transform.translation.z);
    // 提取四元数并转换为旋转向量 rvec
    geometry_msgs::msg::Quaternion q = VTM_to_predicted_transform.rotation;
    Eigen::Quaterniond eigen_quat(q.w, q.x, q.y, q.z);
    eigen_quat.normalize();  // 确保四元数归一化

    // 将四元数转换为旋转矩阵
    Eigen::Matrix3d eigen_rot = eigen_quat.toRotationMatrix();
    // 将旋转矩阵转换为 OpenCV 格式
    // 将旋转矩阵转换为 OpenCV 格式
    cv::Mat rotMat;
    cv::eigen2cv(eigen_rot, rotMat);
    // 将旋转矩阵转换为旋转向量
    cv::Mat rotVec;
    cv::Rodrigues(rotMat, rotVec);

    std::cout << "predicted : " << predicted  << std::endl;

    cv::projectPoints(pw, rotVec, tVec, VTM_intrinsic_, VTM_distortion_, projectedPoint);
    // std::cout << "projectedPoint: " << projectedPoint[0] << std::endl;

    return projectedPoint[0];
}


} // namespace autoaim_prediction

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_highshoot::HighshootNode)