#include <string>
#include <vector>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <autoaim_interfaces/msg/detection_array.hpp>
#include <autoaim_interfaces/msg/comm_send.hpp>
#include <autoaim_interfaces/msg/comm_recv.hpp>
#include <autoaim_interfaces/msg/debug_info.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include <pnp_solver.hpp>
#include <tracker.hpp>
#include <trajectory.hpp>

namespace autoaim_prediction {

using namespace std;
using autoaim_interfaces::msg::Detection;
using autoaim_interfaces::msg::DetectionArray;
using autoaim_interfaces::msg::CommRecv;
using autoaim_interfaces::msg::CommSend;
using autoaim_interfaces::msg::DebugInfo;
using sensor_msgs::msg::CameraInfo;

enum COLOR { GRAY, BLUE, RED };

enum TAG { SENTRY, HERO, ENGINEER, INFANTRY_3, INFANTRY_4, OUTPOST, BASE_SMALL, BASE_BIG};

enum MODE { IDLE, AIMING };

const geometry_msgs::msg::Transform EMPTY_TRANSFORM;

double to_sec(builtin_interfaces::msg::Time t) {
    if (t.sec < 0 || t.nanosec < 0) return 0.0;
    return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
}

std::string get_tf_armor_name(int color, int label, int index) {
    std::string name;
    char color_map[3] = {'G', 'B', 'R'}; // gray, blue, red
    name += color_map[color];
    name += static_cast<char>(label + '0');
    name += '-';
    name += static_cast<char>(index + '0');
    return name;
}

class PredictionNode: public rclcpp::Node {
public:
    explicit PredictionNode(const rclcpp::NodeOptions& options);
    ~PredictionNode() = default;

private:
    void get_parameters();
    void init_publishers();
    void init_subscriptions();
    void detection_callback(const DetectionArray::SharedPtr msg);
    void camera_info_callback(const CameraInfo::SharedPtr msg);

    void get_debug_info(DebugInfo& msg);

    // 找出所有视野里的敌人，按照优先级和是否能打（不打死人和无敌），设置目标敌人编号
    void decide_target_label(const DetectionArray::SharedPtr msg);

    // 选择目标颜色和标签的装甲板，并按照一定规则进行排序
    void select_armors(const std::vector<Detection>& src, std::vector<Detection>& dst) const;

    // 尝试获取指定时间点对应的变换。若尝试MAX_ATTEMPTS后仍没有找到，throw一个std::runtime_error
    geometry_msgs::msg::Transform try_get_transform(
        const std::string& target,
        const std::string& source,
        const rclcpp::Time& time_point
    ) const;

    // 获取指定时间的自己云台的yaw, pitch, roll（相对于chassis）。
    // 之所以是ypr不是rpy，是因为我们采用的旋转顺序是yaw, pitch, roll。
    std::tuple<float, float, float> get_gimbal_ypr(const rclcpp::Time& time_point) const;

    cv::Point2f get_pretiction_VTM(const builtin_interfaces::msg::Time& header_stamp);

    bool is_big_armor(int label) const {
        return (label == 1 || label == 7);
    }

    void reload_params_callback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reload_params_service_;    

    cv::Mat VTM_intrinsic_, VTM_distortion_ ;

    bool enable_print_state_;

    //获取串口节点信息，全是弧度制
    float roll_;                   
    float pitch_ ;                  
    float yaw_   ;                               
    int mode_  ;                                
    int target_color_;
    float bullet_speed_;

    float control_to_aim_time_;
    float control_to_shoot_time_;
    float shoot_compensate_pitch_;
    float shoot_compensate_yaw_;

    int target_label_ = -1;
    bool check_target_label_ = true;

    std::vector<int> enemy_priority_;
    bool is_enemy_can_shoot_[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    rclcpp::Subscription<DetectionArray>::SharedPtr detection_sub_;
    rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<CommRecv>::SharedPtr comm_recv_sub_;

    rclcpp::Publisher<CommSend>::SharedPtr comm_send_pub_;
    rclcpp::Publisher<DebugInfo>::SharedPtr debug_info_pub_;

    std::unique_ptr<PnPSolver> pnp_solver_;
    
    std::unique_ptr<Tracker> tracker_;
};

PredictionNode::PredictionNode(const rclcpp::NodeOptions& options):
     Node("autoaim_prediction", options) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    pnp_solver_ = std::make_unique<PnPSolver>();
    
    std::string tracker_params_path =
        ament_index_cpp::get_package_share_directory("autoaim_prediction")
        + "/config/tracker_params.yaml";
    tracker_ = std::make_unique<Tracker>(tracker_params_path);

    get_parameters();
    init_publishers();
    init_subscriptions();

    reload_params_service_ = create_service<std_srvs::srv::Trigger>(
        "/autoaim_prediction/reload_params",
        std::bind(&PredictionNode::reload_params_callback, this,
            std::placeholders::_1, std::placeholders::_2));
}

void PredictionNode::get_parameters() {
    enable_print_state_ = declare_parameter("enable_print_state", false);

    target_color_ = declare_parameter("target_color", 0);
    bullet_speed_ = declare_parameter("bullet_speed", 30.0);
    control_to_aim_time_ = declare_parameter("control_to_aim_time", 0.098);
    control_to_shoot_time_ = declare_parameter("control_to_shoot_time", 0.098);
    shoot_compensate_pitch_ = math::d2r(declare_parameter("shoot_compensate_pitch", 0.0));
    shoot_compensate_yaw_ = math::d2r(declare_parameter("shoot_compensate_yaw", 0.0));

    cv::FileStorage cameraConfig(ament_index_cpp::get_package_share_directory("autoaim_prediction") + 
    "/config/VTM_camera_params.yaml", cv::FileStorage::READ);
    cameraConfig["camera_CI_MAT"] >> this->VTM_intrinsic_;
    cameraConfig["camera_D_MAT"] >> this->VTM_distortion_ ;
}

void PredictionNode::init_publishers(){
    std::string comm_send_pub_topic = declare_parameter("comm_send_pub_topic","/serial/comm_send");
    comm_send_pub_ = create_publisher<CommSend>(
        comm_send_pub_topic,
        rclcpp::SensorDataQoS().keep_last(1)
    );

    std::string debug_info_pub_topic = declare_parameter("debug_info_pub_topic","/debug_info");
    debug_info_pub_ = create_publisher<DebugInfo>(
        debug_info_pub_topic,
        rclcpp::QoS(1)
    );
}

void PredictionNode::init_subscriptions(){
    std::string camera_info_topic =
        declare_parameter("camera_info_topic", "camera/color/camera_info");
    camera_info_sub_ = create_subscription<CameraInfo>(
        camera_info_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const CameraInfo::SharedPtr msg) { camera_info_callback(msg); }
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
            target_color_ = msg->target_color;
            if(msg->mode == MODE::AIMING && mode_ == MODE::IDLE)
                check_target_label_ = true;
            mode_ =  msg->mode;
        }
    );
}

void PredictionNode::detection_callback(const DetectionArray::SharedPtr msg) {
    std::tuple<float, float, float> gimbal_ypr = get_gimbal_ypr(msg->header.stamp);
    float gimbal_yaw, gimbal_pitch, gimbal_roll;
    std::tie(gimbal_yaw, gimbal_pitch, gimbal_roll) = gimbal_ypr;
    std::vector<Detection> target_armors;
    decide_target_label(msg);
    select_armors(msg->detections, target_armors);
    const int len = target_armors.size();
    for (int i = 0; i < len; i++) {
        const Detection& armor = target_armors[i];
        const std::string armor_name = get_tf_armor_name(armor.color, armor.label, i);
        geometry_msgs::msg::TransformStamped armor_to_cam;
        armor_to_cam.header.stamp = msg->header.stamp;
        armor_to_cam.header.frame_id = "autoaim_camera";
        armor_to_cam.child_frame_id = armor_name;
        // 计算装甲板相对于相机坐标系的位姿
        if (!pnp_solver_->solve_pnp(armor, gimbal_ypr, armor_to_cam.transform)) {
            continue;
        }
        tf_broadcaster_->sendTransform(armor_to_cam);
        // 把装甲板的位姿转换到世界坐标系下进行滤波
        try {
            auto armor_to_chassis = try_get_transform("chassis", armor_name, msg->header.stamp);
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
    tracker_->update(to_sec(msg->header.stamp), target_label_);

    DebugInfo debug_info;
    debug_info.header.stamp = msg->header.stamp;
    this->get_debug_info(debug_info);
    tracker_->get_debug_info(debug_info);
    debug_info_pub_->publish(debug_info);

    if (tracker_->tracker_status != TRACKER_STATUS::LOST) {
        cv::Point3f target;
        bool can_shoot;
        std::tie(target, can_shoot) = tracker_->get_target_pos(
            gimbal_yaw,
            gimbal_pitch,
            bullet_speed_,
            to_sec(now()) - to_sec(msg->header.stamp) + control_to_aim_time_,
            to_sec(now()) - to_sec(msg->header.stamp) + control_to_shoot_time_
        );

        geometry_msgs::msg::TransformStamped target_to_chassis;
        target_to_chassis.header.stamp = msg->header.stamp;
        target_to_chassis.header.frame_id = "chassis";
        target_to_chassis.child_frame_id = "target";
        target_to_chassis.transform.translation.x = target.x;
        target_to_chassis.transform.translation.y = target.y;
        target_to_chassis.transform.translation.z = target.z;
        tf_broadcaster_->sendTransform(target_to_chassis);

        //VTM坐标系
        cv::Point2f point_in_VTM = this->get_pretiction_VTM(msg->header.stamp);

        geometry_msgs::msg::Transform target_to_shoot;
        try {
            // shoot是原点在摩擦轮系，但姿态没有pitch和roll的系。解出来的角度方便控车
            target_to_shoot = try_get_transform("shoot", "target", msg->header.stamp);
        } catch (const std::exception& ex) {
            RCLCPP_WARN(
                get_logger(),
                "Failed to get transform from target to shoot: %s",
                ex.what()
            );
            return;
        }

        // 注意：pitch向下为正
        const float target_pitch = - trajectory::calc_pitch(
            target_to_shoot.translation.x,
            target_to_shoot.translation.y,
            target_to_shoot.translation.z,
            bullet_speed_
        ) + shoot_compensate_pitch_;
        const float target_yaw = math::rad_period_correction(
            atan2(
                target_to_shoot.translation.y,
                target_to_shoot.translation.x
            ) + shoot_compensate_yaw_);

        if (enable_print_state_) {
            tracker_->debug_print_state();
            if(target_armors.size()>0){
                std::printf("Target color %d, label %d\n", target_color_, target_armors[0].label);
            }
            else{
                std::printf("Target color %d.", target_color_);
            }
            std::printf(
                "Pitch: %4.1f  Yaw: %4.1f (degree)  Shoot_flag: %s\n",
                math::r2d(target_pitch),
                math::r2d(target_yaw),
                (can_shoot ? "true" : "false")
            );
        }

        if(target_armors.size() > 0){
            autoaim_interfaces::msg::CommSend comm_send;
            //comm_send.header.stamp = now();
            comm_send.target_find = true;
            comm_send.shoot_flag = can_shoot == true? 2:0;
            comm_send.pitch = math::r2d(target_pitch);
            comm_send.yaw = math::r2d(target_yaw);
            comm_send.vtm_x = point_in_VTM.x;
            comm_send.vtm_y = point_in_VTM.y;
            comm_send_pub_->publish(comm_send);
        }
        else{ //这个不分可能需要商榷
            autoaim_interfaces::msg::CommSend comm_send;
            //comm_send.header.stamp = now();
            comm_send.target_find = false;
            comm_send.shoot_flag = 0;
            comm_send.pitch = math::r2d(gimbal_pitch);
            comm_send.yaw = math::r2d(gimbal_yaw);
            comm_send_pub_->publish(comm_send);    
        }
    }
    else{
        autoaim_interfaces::msg::CommSend comm_send;
        //comm_send.header.stamp = now();z
        comm_send.target_find = false;
        comm_send.shoot_flag = 0;
        comm_send.pitch = math::r2d(gimbal_pitch);
        comm_send.yaw = math::r2d(gimbal_yaw);
        comm_send_pub_->publish(comm_send);    
    }
}

void PredictionNode::decide_target_label(const DetectionArray::SharedPtr msg) {
    static int current_target_lost_frames = 0;
    static int armor_appear_frames[5][15] = {0};
    bool occurred_armors[5][15] = {0};
    if (!msg->detections.empty()) {
        for (const auto& armor: msg->detections) {
            occurred_armors[armor.color][armor.label] = 1;
        }
    }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 15; j++) {
            if (occurred_armors[i][j]) {
                armor_appear_frames[i][j]++;
            } else {
                armor_appear_frames[i][j] = 0;
            }
        }
    }
    if (target_label_ != -1) {
        if (!occurred_armors[target_color_][target_label_]) {
            current_target_lost_frames++;
        } else {
            current_target_lost_frames = 0;
        }
    }

    if(enemy_priority_.empty()) {
        // RCLCPP_INFO(get_logger(), "target_color, %d, current_target_lost_frames, %d, ", target_color_, current_target_lost_frames);
        if(target_label_ == -1 || check_target_label_){
            float min_distance = std::numeric_limits<float>::max();
            int closest_label = -1;
            
            for (const auto& armor : msg->detections) {
                if (!is_enemy_can_shoot_[armor.label] || armor.color != target_color_) {
                    continue;
                }

                Eigen::Vector2f vtm_pt = pnp_solver_->get_center_in_VTM(armor);
                
                float distance = sqrt(pow(vtm_pt.x() - VTM_intrinsic_.at<float>(0,2), 2) + 
                                    pow(vtm_pt.y() - VTM_intrinsic_.at<float>(1,2), 2));
                                    
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_label = armor.label;
                }
            }

            // RCLCPP_INFO(get_logger(), "closest_label, %d", closest_label);
            if(check_target_label_ && closest_label == target_label_){
                check_target_label_ = false;
                return;
            }

            if (closest_label != -1) {
                target_label_ = closest_label;
                current_target_lost_frames = 0;
                tracker_->tracker_status = TRACKER_STATUS::LOST;
                check_target_label_ = false;
                return;
            }
        }
        else {
            if (is_enemy_can_shoot_[target_label_] && current_target_lost_frames <= (target_label_ == TAG::OUTPOST ? 100 : 10)) {
                return;
            }
        }
    }
    else {
        for (const int label: enemy_priority_) {
            if (label == target_label_) {
                if (is_enemy_can_shoot_[label] && current_target_lost_frames <= (target_label_ == TAG::OUTPOST ? 100 : 10)) {
                    return;
                }
            } else {
                if (is_enemy_can_shoot_[label] && armor_appear_frames[target_color_][label] > 5) {
                    target_label_ = label;
                    current_target_lost_frames = 0;
                    // 把tracker_status设置为lost，下次进入的时候就会重置滤波器了
                    tracker_->tracker_status = TRACKER_STATUS::LOST;
                    return;
                }
            }
        }
    }
    target_label_ = -1;
    tracker_->tracker_status = TRACKER_STATUS::LOST;
}

void PredictionNode::select_armors(const std::vector<Detection>& src, std::vector<Detection>& dst) const {
    // RCLCPP_INFO(get_logger(), "target_label: %d", target_label_);
    constexpr auto get_center_x = [](const Detection& d) -> int {
        return (d.bl.x + d.br.x + d.tr.x + d.tl.x) / 4;
    };
    constexpr auto get_area = [](const Detection& d) -> int {
        return (d.br.x - d.tl.x) * (d.br.y - d.tl.y);
    };
    constexpr auto get_length_height_ratio = [](const Detection& d) -> float {
        return fabs((d.tl.x + d.bl.x - d.tr.x - d.br.x) / (d.tl.y + d.tr.y - d.bl.y - d.br.y));
    };
    static int center_x_prev = 0;
    std::vector<Detection> filtered;
    // 筛选出目标颜色和标签的装甲板
    for (const auto& armor: src) {
        // 用装甲板长宽比筛掉太斜的装甲板
        // std::cout <<" length_height_ratio: " << get_length_height_ratio(armor) << std::endl;
        const bool yaw_too_large = get_length_height_ratio(armor) < (is_big_armor(armor.label) ? 2.0 : 1.60);
        // const bool yaw_too_large = get_length_height_ratio(armor) < (is_big_armor(armor.label) ? 2.0 : 1.7);
        if (!yaw_too_large && armor.label == target_label_) {
            if (target_color_ == armor.color) {
                filtered.emplace_back(armor);
            } else if (armor.color == COLOR::GRAY) { // 特殊处理灰色装甲板
                if (abs(get_center_x(armor) - center_x_prev) <= 15) {
                    // 这里只根据灰色装甲板位置与上次瞄的位置差判断是否是被打成灰的
                    filtered.emplace_back(armor);
                }
            }
        }
    }
    if (filtered.empty()) {
        center_x_prev = 0;
        return;
    }

    // 对目标装甲板进行排序
    if (filtered.size() == 1) {
        dst.push_back(filtered[0]);
    } else {
        // 根据击打面积和装甲板位置与正在瞄准位置间的差异排序
        std::sort(filtered.begin(), filtered.end(), [&](const Detection& a, const Detection& b) {
            if (center_x_prev == 0) {
                return get_area(a) > get_area(b);
            } else {
                return get_area(a) - abs(get_center_x(a) - center_x_prev)
                    > get_area(b) - abs(get_center_x(b) - center_x_prev);
            }
        });
        dst.emplace_back(filtered[0]);
        // 接下来选择击打面积次之，且和原来那个位置有较大差异的装甲板。
        // 虽然理论上detection中的nms已经能去除同一个装甲板的多个识别结果，但有时候还是会出现。
        const int len = filtered.size();
        const int center_x_first = get_center_x(filtered[0]);
        for (int i = 1; i < len; i++) {
            const int center_x_i = get_center_x(filtered[i]);
            if (abs(center_x_first - center_x_i) >= 15) {
                dst.push_back(filtered[i]);
                break;
            }
        }
    }
    center_x_prev = get_center_x(dst[0]);
}

std::tuple<float, float, float> PredictionNode::get_gimbal_ypr(const rclcpp::Time& time_point) const {
    // 保存之前找过的ypr，在lookupTransform出现异常时返回
    static std::tuple<float, float, float> prev_ypr = std::make_tuple(0, 0, 0);
    geometry_msgs::msg::Transform transform;
    try {
        transform = try_get_transform("chassis", "gimbal", time_point);
    } catch (const std::exception& ex) {
        // Try getting latest transform instead
        RCLCPP_WARN(get_logger(), "Failed to get transform: %s", ex.what());
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

geometry_msgs::msg::Transform PredictionNode::try_get_transform(
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

void PredictionNode::get_debug_info(DebugInfo& msg) {
    msg.target_color = target_color_;
    msg.target_label = target_label_;
    msg.bullet_speed = bullet_speed_;
    for (int i = 0; i < 10; i++) {
        msg.is_enemy_can_shoot.push_back(is_enemy_can_shoot_[i]);
    }
}

void PredictionNode::camera_info_callback(const CameraInfo::SharedPtr msg) {
    auto VTM_to_camera_transform = this->try_get_transform(
        "autoaim_camera",
        "VTM",
        rclcpp::Time(0)
    );
    pnp_solver_->set_cam_matrix(
        cv::Mat(3, 3, CV_64F, msg->k.data()),
        cv::Mat(1, 5, CV_64F, msg->d.data()),
        this->VTM_intrinsic_,
        this->VTM_distortion_,
        VTM_to_camera_transform.translation.x,
        VTM_to_camera_transform.translation.y,
        VTM_to_camera_transform.translation.z
    );
    // 相机内参和畸变在运行中不会改变，所以设置后即可取消camera_info订阅
    camera_info_sub_.reset();
    camera_info_sub_ = nullptr;
}

cv::Point2f PredictionNode::get_pretiction_VTM(const builtin_interfaces::msg::Time& header_stamp){
    geometry_msgs::msg::Transform predicted_to_VTM_transform;
    try {
        predicted_to_VTM_transform = this->try_get_transform(
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
    cv::Mat tVec = (cv::Mat_<double>(3, 1) << predicted_to_VTM_transform.translation.x,
                    predicted_to_VTM_transform.translation.y,predicted_to_VTM_transform.translation.z);
    // 提取四元数并转换为旋转向量 rvec
    geometry_msgs::msg::Quaternion q = predicted_to_VTM_transform.rotation;
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

    cv::projectPoints(pw, rotVec, tVec, VTM_intrinsic_, VTM_distortion_, projectedPoint);

    return projectedPoint[0];
}

void PredictionNode::reload_params_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    try {
        // Reload prediction node params
        get_parameter("control_to_aim_time", control_to_aim_time_);
        get_parameter("control_to_shoot_time", control_to_shoot_time_);
        get_parameter("shoot_compensate_pitch", shoot_compensate_pitch_);
        get_parameter("shoot_compensate_yaw", shoot_compensate_yaw_);
        shoot_compensate_pitch_ = math::d2r(shoot_compensate_pitch_);
        shoot_compensate_yaw_ = math::d2r(shoot_compensate_yaw_);

        // Reload tracker and kalman filter params
        std::string tracker_params_path = 
            ament_index_cpp::get_package_share_directory("autoaim_prediction") + 
            "/config/tracker_params.yaml";
        tracker_->reload_params(tracker_params_path);

        response->success = true;
        response->message = "Successfully reloaded all parameters";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = "Failed to reload parameters: " + std::string(e.what());
    }
}

} // namespace autoaim_prediction

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_prediction::PredictionNode)