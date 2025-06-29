// 维护一个整车状态的跟踪器
// KFXYZ用于平动，KFYaw+UKFXY用于小陀螺，半径和高度采用惯性滤波
// 坐标系定义：除相机系（这里应该没涉及）外，其余都是向右x，向前y，向上z
// yaw角方向定义：逆时针（即从x到y）为正
// 距离和时间均使用国际单位制（m、s），角度使用弧度制

#pragma once

#include <kalman_filters.hpp>
#include <math_utils.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <autoaim_interfaces/msg/debug_info.hpp>
#include <trajectory.hpp>

enum class TRACKER_STATUS { CONVERGING, TRACKING, TEMP_LOST, LOST };

struct Armor {
    cv::Point3f center; // 装甲板中心坐标
    float angle; // 装甲板向心方向在xy平面的投影向量与正前方（y轴）的夹角，逆时针为正
};

class Tracker {
public:
    Tracker(const std::string& params_path);
    void push(const geometry_msgs::msg::Transform& transform);
    void update(const double time_stamp, const int label);
    void debug_print_state();
    void get_debug_info(autoaim_interfaces::msg::DebugInfo& debug_info);
    void reload_params(const std::string& params_path);

    /*!
        @brief 获取预测击打坐标（世界系下）
        @param gimbal_yaw 云台相对世界系的yaw（用于计算面向我们的装甲板是哪个）
        @param bullet_speed 子弹速度
        @param img_to_aim_time 图像时间到开火时间的估计值
        @param img_to_shoot_time 图像时间到开火时间的估计值
        @return 预测的击打坐标，和是否发弹（即shoot_flag）
        @attention 不应在tracker_status为lost时调用
    */
    std::tuple<cv::Point3f, bool> get_target_pos(
        const float gimbal_yaw,
        const float gimbal_pitch,
        const float bullet_speed, 
        const float img_to_aim_time,
        const float img_to_shoot_time
    );

    TRACKER_STATUS tracker_status = TRACKER_STATUS::LOST;

private:
    float INITIAL_RADIUS = 0.26;
    float MIN_RADIUS = 0.2, MAX_RADIUS = 0.35;
    float OUTPOST_RADIUS = 0.22;
    float SWITCH_ARMOR_ANGLE = math::d2r(50);
    float CLOSE_RADIUS_FILTER_RATIO = 0.7;
    float FAR_RADIUS_FILTER_RATIO = 0.8;
    float CLOSE_HEIGHT_FILTER_RATIO = 0.6;
    float FAR_HEIGHT_FILTER_RATIO = 0.7;
    float ANTITOP_PALSTANCE_THRESHOLD = math::d2r(50);
    float ANTITOP_CAN_SHOOT_ANGLE_THIN = math::d2r(25);
    float ANTITOP_CAN_SHOOT_ANGLE_WIDE = math::d2r(30);
    float ANTITOP_FOLLOW_ANGLE = math::d2r(30);
    float OUTPOST_CAN_SHOOT_ANGLE_THIN = math::d2r(55);
    float OUTPOST_CAN_SHOOT_ANGLE_WIDE = math::d2r(60);
    int MAX_LOST_FRAMES = 5;
    int CONVERGE_FRAMES = 5;
    int OUTPOST_MAX_LOST_FRAMES = 40;
    float SHOOT_FORBIDEN_D_PALSTANCE = math::d2r(23);

    std::shared_ptr<KFXYZ> kf_xyz_;
    std::shared_ptr<KFYaw> kf_yaw_;
    std::shared_ptr<UKFXY> ukf_;
    unsigned track_frames_ = 0; // 从不为LOST开始时一直跟踪的帧数
    unsigned appear_frames_ = 0; // 从不为TEMP_LOST开始时连续出现的帧数
    unsigned lost_frames_ = 0; // 从TEMP_LOST或LOST开始时连续消失的帧数
    unsigned observing_armor_id_ = 0; // 正在观测的装甲板编号。定义第一块看到的装甲板为0，车逆时针转时看到的依次编号1、2、3
    float radius_[2]; // radius_[0]对应0、2装甲板半径，radius_[1]对应1、3
    // float height_[4]; // height_[0]对应0、2装甲板中心z坐标，height_[1]对应1、3
    float height_[2]; // height_[0]对应0、2装甲板中心z坐标，height_[1]对应1、3
    float accumulated_yaw_ = 0; // 根据帧间差累计的yaw角，用于更新kf_yaw_

    double prev_update_time_ = 0;
    float prev_update_angle_ = 0;
   int target_label_; // 当前正在跟踪的目标编号，用于特判前哨站
    std::vector<Armor> armors_;

    void load_params(const std::string& params_path);
    void update_radius();
    void update_height();
    bool is_outpost() const { return (target_label_ == 5); }
};

Tracker::Tracker(const std::string& params_path) {
    load_params(params_path);
    kf_xyz_ = std::make_unique<KFXYZ>(params_path);
    kf_yaw_ = std::make_unique<KFYaw>(params_path);
    ukf_ = std::make_unique<UKFXY>(params_path);
}

void Tracker::push(const geometry_msgs::msg::Transform& transform) {
    Armor armor;
    armor.center =
        cv::Point3f(transform.translation.x, transform.translation.y, transform.translation.z);
    tf2::Quaternion quaternion(
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w
    );
    tf2::Matrix3x3 rotation_mat(quaternion);
    double yaw, pitch, roll;
    rotation_mat.getEulerYPR(yaw, pitch, roll);
    armor.angle = yaw;
    armors_.emplace_back(armor);
}

void Tracker::update(const double time_stamp, const int label) {
    target_label_ = label;
    using TS = TRACKER_STATUS;
    const float time_elapsed = static_cast<float>(time_stamp - prev_update_time_); // 和上一帧比经过的时间

    if (tracker_status != TS::LOST) {
        track_frames_++;
    } else {
        track_frames_ = 0;
    }

    if (armors_.empty() || armors_.size() > 2) {
        appear_frames_ = 0;
        lost_frames_++;
        if (tracker_status != TS::LOST) { // 短暂失踪，只预测不更新
            kf_xyz_->predict(time_elapsed);
            kf_yaw_->predict(time_elapsed);
            ukf_->predict(time_elapsed);
            if (lost_frames_ >= (is_outpost() ? OUTPOST_MAX_LOST_FRAMES : MAX_LOST_FRAMES)) {
                tracker_status = TS::LOST;
            } else {
                tracker_status = TS::TEMP_LOST;
            }
        }
    } else {
        const int armors_count = is_outpost() ? 3 : 4;
        lost_frames_ = 0;
        appear_frames_++;
        if (tracker_status == TS::LOST) { // 初始化
            kf_xyz_->initialize(armors_[0].center);
            kf_yaw_->initialize(armors_[0].angle);
            const float radius = is_outpost() ? OUTPOST_RADIUS : INITIAL_RADIUS;
            const cv::Point2f car_center(
                armors_[0].center.x + radius * cos(armors_[0].angle),
                armors_[0].center.y + radius * sin(armors_[0].angle)
            );
            ukf_->initialize(car_center);
            observing_armor_id_ = 0;
            radius_[0] = radius_[1] = radius;
            // height_[0] = height_[1] = height_[2] = height_[3] = armors_[0].center.z;
            height_[0] = height_[1] = armors_[0].center.z;
            accumulated_yaw_ = prev_update_angle_ = armors_[0].angle;
        } else { // 正常预测并更新
            kf_xyz_->predict(time_elapsed);
            kf_yaw_->predict(time_elapsed);
            ukf_->predict(time_elapsed);
            float delta_angle = math::rad_period_correction(armors_[0].angle - prev_update_angle_);
            accumulated_yaw_ += delta_angle;
            if (delta_angle < -SWITCH_ARMOR_ANGLE) {
                // 逆时针转（角速度大于0）时切换装甲板
                observing_armor_id_ += 1;
                observing_armor_id_ %= armors_count;
                accumulated_yaw_ += M_PI * 2 / armors_count;
                kf_xyz_->force_change_position(armors_[0].center);
            } else if (delta_angle > SWITCH_ARMOR_ANGLE) {
                // 顺时针转（角速度小于0）时切换装甲板
                observing_armor_id_ += armors_count - 1;
                observing_armor_id_ %= armors_count;
                accumulated_yaw_ -= M_PI * 2 / armors_count;
                kf_xyz_->force_change_position(armors_[0].center);
            }
            kf_xyz_->update(armors_[0].center);
            kf_yaw_->update(accumulated_yaw_);
            update_radius();
            update_height();
            const float radius = is_outpost() ? OUTPOST_RADIUS : radius_[observing_armor_id_ % 2];
            // 计算车的中心时应直接用观测量armors_[0].angle而非update_angle
            const cv::Point2f car_center(
                armors_[0].center.x + radius * cos(armors_[0].angle),
                armors_[0].center.y + radius * sin(armors_[0].angle)
            );
            ukf_->update(car_center);

            prev_update_angle_ = armors_[0].angle;
        }
        if (track_frames_ >= CONVERGE_FRAMES) {
            tracker_status = TS::TRACKING;
        } else {
            tracker_status = TS::CONVERGING;
        }
    }

    armors_.clear();
    prev_update_time_ = time_stamp;
}

std::tuple<cv::Point3f, bool> Tracker::get_target_pos(
    const float gimbal_yaw,
    const float gimbal_pitch,
    const float bullet_speed, 
    const float img_to_aim_time,
    const float img_to_shoot_time
) {
    static bool last_shoot_flag = false; // 上一帧是否发弹
    static float last_kf_palstance = 0; // 上一帧kf_yaw的palstance
    static float sum_d_palstance = 0; // 
    // if(false){
    if (abs(kf_yaw_->palstance) < ANTITOP_PALSTANCE_THRESHOLD && !is_outpost()) { // 平动，只用KFXYZ预测
        // 使用迭代法求解精确的击打时间和位置，最多10次迭代
        // 初始值使用二阶近似
        float flight_time = math::get_distance(kf_xyz_->position) / bullet_speed ;
        flight_time = math::get_distance(kf_xyz_->position + flight_time * kf_xyz_->velocity) / bullet_speed;
        float total_aim_time = img_to_aim_time + flight_time;
        cv::Point3f pred_pos = kf_xyz_->position + total_aim_time * kf_xyz_->velocity;
        for (int i = 0; i < 10; i++) {
            // 计算从枪口到预测位置的飞行时间
            float new_flight_time = trajectory::calc_fly_time(
                pred_pos.x, pred_pos.y, pred_pos.z, bullet_speed);
            float new_total_aim_time = img_to_aim_time + new_flight_time;
            cv::Point3f new_pred_pos = kf_xyz_->position + new_total_aim_time * kf_xyz_->velocity;
            
            // 检查是否收敛
            if (fabs(new_total_aim_time - total_aim_time) < 0.0001f) {
                break;
            }
            
            pred_pos = new_pred_pos;
            total_aim_time = new_total_aim_time;
        }
        
        return std::make_tuple(
            pred_pos, 
            tracker_status != TRACKER_STATUS::CONVERGING
        );
    } else { // 转动，用KFYaw和UKFXY预测
         // 使用迭代法求解精确的击打时间和位置，最多10次迭代
        // 初始值使用二阶近似
        
        float flight_time = math::get_distance(cv::Point3f(ukf_->position.x, ukf_->position.y, (height_[0] + height_[1])/2)) / bullet_speed;
        flight_time = math::get_distance(cv::Point3f((ukf_->position + flight_time * ukf_->velocity).x, (ukf_->position + flight_time * ukf_->velocity).y, (height_[0] + height_[1])/2)) / bullet_speed;
        
        float total_aim_time = img_to_aim_time + flight_time;
        cv::Point2f pred_center = ukf_->position + ukf_->velocity * total_aim_time;
        float pred_yaw_to_world = kf_yaw_->yaw + kf_yaw_->palstance * total_aim_time;
        
        // 车的装甲板数量，前哨站只有三个装甲板
        const int armors_count = is_outpost() ? 3 : 4;
        cv::Point3f pred_pos;
        float target_angle_to_gimbal = M_PI * 2 / armors_count; // 声明在循环外部
        int target_armor_index = 0;
        // 定义follow_angle在循环外部
        float follow_angle = is_outpost() ? OUTPOST_CAN_SHOOT_ANGLE_WIDE : ANTITOP_FOLLOW_ANGLE;

        for (int i = 0; i < 30; i++) {
            // 0号装甲板在gimbal系下的预测yaw角
            float pred_yaw_to_gimbal = math::rad_period_correction(pred_yaw_to_world - gimbal_yaw);
 
            // 最面向我们的装甲板在gimbal系下的预测角
            target_angle_to_gimbal = M_PI * 2 / armors_count; 
            target_armor_index = 0;
            
            // 选择在total_aim_time之后，角度最小（即最面向我们）的那个装甲板
            for (int j = 0; j < armors_count; j++) {
                const float pred_angle_to_gimbal =
                    math::rad_period_correction(pred_yaw_to_gimbal - M_PI * j * 2 / armors_count);
                if (abs(pred_angle_to_gimbal) < abs(target_angle_to_gimbal)) {
                    target_angle_to_gimbal = pred_angle_to_gimbal;
                    target_armor_index = j % 2;
                }
            }
            
            // 确定预测坐标
            if (abs(target_angle_to_gimbal) < follow_angle) { // 跟随射击
                const float target_angle_to_world = 
                    math::rad_period_correction(target_angle_to_gimbal + gimbal_yaw);
                const float radius = is_outpost() ? OUTPOST_RADIUS : radius_[target_armor_index];
                const float height = is_outpost() ? height_[0] : height_[target_armor_index];
                pred_pos = cv::Point3f(
                    pred_center.x - cos(target_angle_to_world) * radius,
                    pred_center.y - sin(target_angle_to_world) * radius,
                    height
                );
            } else { // 去下一块装甲板出现位置准备射击
                const float next_follow_angle_to_world = math::rad_period_correction(
                    (kf_yaw_->palstance > 0 ? -1 : 1) * follow_angle + gimbal_yaw
                );
                const float radius = is_outpost() ? OUTPOST_RADIUS : radius_[target_armor_index];
                const float height = is_outpost() ? height_[0] : height_[target_armor_index];
                pred_pos = cv::Point3f(
                    pred_center.x - cos(next_follow_angle_to_world) * radius,
                    pred_center.y - sin(next_follow_angle_to_world) * radius,
                    height
                );
            }
            
            // 重新计算子弹飞行时间
            float new_flight_time = trajectory::calc_fly_time(
                pred_pos.x, pred_pos.y, pred_pos.z, bullet_speed);
            std::cout << "flight_time: " << flight_time << std::endl;
            std::cout << "new_flight_time: " << new_flight_time << std::endl;
            float new_total_aim_time = img_to_aim_time + new_flight_time;
     
            cv::Point2f new_pred_center = ukf_->position + ukf_->velocity * new_total_aim_time;
            float new_pred_yaw_to_world = kf_yaw_->yaw + kf_yaw_->palstance * new_total_aim_time;

            // 检查是否收敛
            if (fabs(new_total_aim_time - total_aim_time) < 0.0001f) {
                break;
            }
            
            pred_center = new_pred_center;
            pred_yaw_to_world = new_pred_yaw_to_world;
            total_aim_time = new_total_aim_time;
        }
         const float img_to_aim_time_2 = total_aim_time;
        const float img_to_shoot_time_2 = total_aim_time - img_to_aim_time + img_to_shoot_time;
        pred_center = ukf_->position + ukf_->velocity * img_to_aim_time_2;
        // 0号装甲板在世界系下的预测yaw角
        pred_yaw_to_world = kf_yaw_->yaw + kf_yaw_->palstance * img_to_aim_time_2;
        const float pred_yaw_to_world_shoot = kf_yaw_->yaw + kf_yaw_->palstance * img_to_shoot_time_2;
        // 0号装甲板在gimbal系下的预测yaw角
        const float pred_yaw_to_gimbal = math::rad_period_correction(pred_yaw_to_world - gimbal_yaw);
        const float pred_yaw_to_gimbal_shoot = math::rad_period_correction(pred_yaw_to_world_shoot - gimbal_yaw);
        // 最面向我们的装甲板在gimbal系下的预测角
        target_angle_to_gimbal = M_PI * 2 / armors_count; 
        float target_angle_to_gimbal_shoot = M_PI * 2 / armors_count; 
        target_armor_index = 0;
        // 选择在img_to_hit_time之后，角度最小（即最面向我们）的那个装甲板
        for (int i = 0; i < armors_count; i++) {
            const float pred_angle_to_gimbal =
                math::rad_period_correction(pred_yaw_to_gimbal - M_PI * i * 2 / armors_count);
            const float pred_angle_to_gimbal_shoot =
                math::rad_period_correction(pred_yaw_to_gimbal_shoot - M_PI * i * 2 / armors_count);
            if (abs(pred_angle_to_gimbal) < abs(target_angle_to_gimbal)) {
                target_angle_to_gimbal = pred_angle_to_gimbal;
                target_angle_to_gimbal_shoot = pred_angle_to_gimbal_shoot;
                target_armor_index = i % 2;
                //  target_armor_index = (i) ; //oyg_记得改
            }
        }
        follow_angle = is_outpost() ? OUTPOST_CAN_SHOOT_ANGLE_WIDE : ANTITOP_FOLLOW_ANGLE;
        if (abs(target_angle_to_gimbal) < follow_angle) { // 跟随射击
            const float target_angle_to_world = 
                math::rad_period_correction(target_angle_to_gimbal + gimbal_yaw);
            const float radius = is_outpost() ? OUTPOST_RADIUS : radius_[target_armor_index];
            const float height = is_outpost() ? height_[0] : height_[target_armor_index];
            const cv::Point3f target = cv::Point3f(
                pred_center.x - cos(target_angle_to_world) * radius,
                pred_center.y - sin(target_angle_to_world) * radius,
                height
            );
            const float can_shoot_angle = 
                last_shoot_flag ?
                (is_outpost() ? OUTPOST_CAN_SHOOT_ANGLE_WIDE : ANTITOP_CAN_SHOOT_ANGLE_WIDE) : 
                (is_outpost() ? OUTPOST_CAN_SHOOT_ANGLE_THIN : ANTITOP_CAN_SHOOT_ANGLE_THIN);
            const bool shoot_flag = abs(target_angle_to_gimbal_shoot) < can_shoot_angle;
            last_shoot_flag = shoot_flag;

            // 使用滑动窗口保存最近5帧palstance，计算滑动平均变化量
            static std::deque<float> palstance_window;
            static constexpr int PALSTANCE_OBSERVE_FRAMES = 4;

            palstance_window.push_back(kf_yaw_->palstance);
            if (palstance_window.size() > PALSTANCE_OBSERVE_FRAMES) {
                palstance_window.pop_front();
            }

            float palstance_avg_diff = 0.0f;
            if (palstance_window.size() == PALSTANCE_OBSERVE_FRAMES) {
                for (size_t i = 1; i < palstance_window.size(); ++i) {
                    palstance_avg_diff += std::abs(palstance_window[i] - palstance_window[i - 1]);
                    std::cout << "palstance_window[" << i << "]: " << palstance_window[i] - palstance_window[i - 1] << std::endl;
                }
                palstance_avg_diff /= (PALSTANCE_OBSERVE_FRAMES - 1);

                if (palstance_avg_diff > SHOOT_FORBIDEN_D_PALSTANCE) {
                    return std::make_tuple(target, false);
                }
            }

            return std::make_tuple(target, shoot_flag && tracker_status != TRACKER_STATUS::CONVERGING);
            // ...existing code...
            
        } else { // 去下一块装甲板出现位置准备射击
            const float next_follow_angle_to_world = math::rad_period_correction(
                (kf_yaw_->palstance > 0 ? -1 : 1) * follow_angle + gimbal_yaw
            );
            // const float radius = is_outpost() ? OUTPOST_RADIUS : radius_[1 - target_armor_index];
            // const float height = is_outpost() ? height_[0] : height_[1 - target_armor_index];

            const float radius = is_outpost() ? OUTPOST_RADIUS : radius_[target_armor_index];
            const float height = is_outpost() ? height_[0] : height_[target_armor_index];
          

            const cv::Point3f target = cv::Point3f(
                pred_center.x - cos(next_follow_angle_to_world) * radius,
                pred_center.y - sin(next_follow_angle_to_world) * radius,
                height
            );
            last_shoot_flag = false;
            return std::make_tuple(target, false);
        }
    }
}

void Tracker::update_radius() {
    if (!is_outpost() && armors_.size() == 2) {
        const int index = observing_armor_id_ % 2;
        const float delta_x = armors_[1].center.x - armors_[0].center.x;
        const float delta_y = armors_[1].center.y - armors_[0].center.y;
        const float theta = armors_[0].angle;
        const float r_first = abs(delta_x * cos(theta) + delta_y * sin(theta));
        const float r_next = abs(-delta_x * sin(theta) + delta_y * cos(theta));
        if (MIN_RADIUS <= r_first && r_first <= MAX_RADIUS) {
            radius_[index] = CLOSE_RADIUS_FILTER_RATIO * radius_[index]
                + (1 - CLOSE_RADIUS_FILTER_RATIO) * r_first;
        }
        if (MIN_RADIUS <= r_next && r_next <= MAX_RADIUS) {
            radius_[1 - index] = FAR_RADIUS_FILTER_RATIO * radius_[1 - index]
                + (1 - FAR_RADIUS_FILTER_RATIO) * r_next;
        }
    }
}

void Tracker::update_height() {
    if (is_outpost()) {
        height_[0] = CLOSE_HEIGHT_FILTER_RATIO * height_[0] 
            + (1 - CLOSE_HEIGHT_FILTER_RATIO) * armors_[0].center.z;
        return;
    }

    const int index = observing_armor_id_ % 2;
    // std::cout << "centrez0" << armors_[0].center.z << std::endl;
    // std::cout << "centrez1" << armors_[1].center.z << std::endl;
    height_[index] = CLOSE_HEIGHT_FILTER_RATIO * height_[index]
        + (1 - CLOSE_HEIGHT_FILTER_RATIO) * armors_[0].center.z;
    if (armors_.size() == 2) {
        height_[1 - index] = FAR_HEIGHT_FILTER_RATIO * height_[1 - index]
            + (1 - FAR_HEIGHT_FILTER_RATIO) * armors_[1].center.z;
    }
}

void Tracker::load_params(const std::string& params_path) {
    cv::FileStorage fs(params_path, cv::FileStorage::READ);
    
    fs["Tracker"]["switch_armor_angle"] >> SWITCH_ARMOR_ANGLE;
    SWITCH_ARMOR_ANGLE = math::d2r(SWITCH_ARMOR_ANGLE);
    fs["Tracker"]["close_height_filter_ratio"] >> CLOSE_HEIGHT_FILTER_RATIO;
    fs["Tracker"]["far_height_filter_ratio"] >> FAR_HEIGHT_FILTER_RATIO;
    fs["Tracker"]["converge_frames"] >> CONVERGE_FRAMES;

    fs["Tracker"]["initial_radius"] >> INITIAL_RADIUS;
    fs["Tracker"]["min_radius"] >> MIN_RADIUS;
    fs["Tracker"]["max_radius"] >> MAX_RADIUS;
    fs["Tracker"]["close_radius_filter_ratio"] >> CLOSE_RADIUS_FILTER_RATIO;
    fs["Tracker"]["far_radius_filter_ratio"] >> FAR_RADIUS_FILTER_RATIO;
    fs["Tracker"]["antitop_palstance_threshold"] >> ANTITOP_PALSTANCE_THRESHOLD;
    ANTITOP_PALSTANCE_THRESHOLD = math::d2r(ANTITOP_PALSTANCE_THRESHOLD);
    fs["Tracker"]["antitop_follow_angle"] >> ANTITOP_FOLLOW_ANGLE;
    ANTITOP_FOLLOW_ANGLE = math::d2r(ANTITOP_FOLLOW_ANGLE);
    fs["Tracker"]["antitop_can_shoot_angle_thin"] >> ANTITOP_CAN_SHOOT_ANGLE_THIN;
    ANTITOP_CAN_SHOOT_ANGLE_THIN = math::d2r(ANTITOP_CAN_SHOOT_ANGLE_THIN);
    fs["Tracker"]["antitop_can_shoot_angle_wide"] >> ANTITOP_CAN_SHOOT_ANGLE_WIDE;
    ANTITOP_CAN_SHOOT_ANGLE_WIDE = math::d2r(ANTITOP_CAN_SHOOT_ANGLE_WIDE);
    fs["Tracker"]["max_lost_frames"] >> MAX_LOST_FRAMES;

    fs["Tracker"]["outpost_radius"] >> OUTPOST_RADIUS;
    fs["Tracker"]["outpost_max_lost_frames"] >> OUTPOST_MAX_LOST_FRAMES;
    fs["Tracker"]["outpost_can_shoot_angle_thin"] >> OUTPOST_CAN_SHOOT_ANGLE_THIN;
    OUTPOST_CAN_SHOOT_ANGLE_THIN = math::d2r(OUTPOST_CAN_SHOOT_ANGLE_THIN);
    fs["Tracker"]["outpost_can_shoot_angle_wide"] >> OUTPOST_CAN_SHOOT_ANGLE_WIDE;
    OUTPOST_CAN_SHOOT_ANGLE_WIDE = math::d2r(OUTPOST_CAN_SHOOT_ANGLE_WIDE);
    fs["Tracker"]["shoot_forbiden_d_palstance"] >> SHOOT_FORBIDEN_D_PALSTANCE;
    SHOOT_FORBIDEN_D_PALSTANCE = math::d2r(SHOOT_FORBIDEN_D_PALSTANCE);
    
    fs.release();
}

void Tracker::reload_params(const std::string& params_path) {
    load_params(params_path);
    kf_xyz_->reload_params(params_path);
    kf_yaw_->reload_params(params_path); 
    ukf_->reload_params(params_path);
}

void Tracker::debug_print_state() {
    std::printf("----------\n");
    std::printf("current status: ");
    if (tracker_status == TRACKER_STATUS::CONVERGING) {
        printf("converging, appear_frames: %d\n", appear_frames_);
    } else if (tracker_status == TRACKER_STATUS::TRACKING) {
        printf("tracking, appear_frames: %d\n", appear_frames_);
    } else if (tracker_status == TRACKER_STATUS::LOST) {
        printf("lost, lost_frames: %d\n", lost_frames_);
    } else if (tracker_status == TRACKER_STATUS::TEMP_LOST) {
        printf("temp_lost, lost_frames: %d\n", lost_frames_);
    }
    std::printf(
        "kf xyz: [%3.0f, %3.0f, %3.0f] += [%3.0f, %3.0f, %3.0f] (cm)\n",
        kf_xyz_->position.x * 100,
        kf_xyz_->position.y * 100,
        kf_xyz_->position.z * 100,
        kf_xyz_->velocity.x * 100,
        kf_xyz_->velocity.y * 100,
        kf_xyz_->velocity.z * 100
    );
    std::printf(
        "kf yaw: %5.0f += %3.0f (degree)\n",
        math::r2d(kf_yaw_->yaw),
        math::r2d(kf_yaw_->palstance)
    );
    std::printf(
        "ukf center: [%3.0f, %3.0f] += [%3.0f, %3.0f] (cm)\n",
        ukf_->position.x * 100,
        ukf_->position.y * 100,
        ukf_->velocity.x * 100,
        ukf_->velocity.y * 100
    );
    std::printf("radius: %3.0f, %3.0f (cm)\n", radius_[0] * 100, radius_[1] * 100);
    // std::printf("height: %3.0f, %3.0f, %3.0f, %3.0f (cm)\n", height_[0] * 100, height_[1] * 100, height_[2] * 100, height_[3] * 100);
    std::printf("height: %3.0f, %3.0f (cm)\n", height_[0] * 100, height_[1] * 100);

}

void Tracker::get_debug_info(autoaim_interfaces::msg::DebugInfo& debug_info) {
    debug_info.tracker_status = static_cast<int>(tracker_status);
    debug_info.appear_frames = appear_frames_;
    debug_info.track_frames = track_frames_;
    debug_info.lost_frames = lost_frames_;
    debug_info.observing_armor_id = observing_armor_id_;
    constexpr auto cvpt3_to_tfpt = [](const cv::Point3f& p) {
        geometry_msgs::msg::Point32 ret;
        ret.x = p.x, ret.y = p.y, ret.z = p.z;
        return ret;
    };
    constexpr auto cvpt2_to_tfpt = [](const cv::Point2f& p) {
        geometry_msgs::msg::Point32 ret;
        ret.x = p.x, ret.y = p.y;
        return ret;
    };
    debug_info.kf_xyz_position = cvpt3_to_tfpt(kf_xyz_->position);
    debug_info.kf_xyz_velocity = cvpt3_to_tfpt(kf_xyz_->velocity);
    debug_info.ukf_xy_position = cvpt2_to_tfpt(ukf_->position);
    debug_info.ukf_xy_velocity = cvpt2_to_tfpt(ukf_->velocity);
    debug_info.kf_yaw = kf_yaw_->yaw;
    debug_info.kf_yaw_palstance = kf_yaw_->palstance;
    for (int i = 0; i < 2; i++) {
        debug_info.radius.emplace_back(radius_[i]);
        debug_info.height.emplace_back(height_[i]);
        // debug_info.height.emplace_back(height_[i+2]);
    }
}
