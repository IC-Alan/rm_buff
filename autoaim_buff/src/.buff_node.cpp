#include <string>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoaim_interfaces/msg/detection_array.hpp>
#include <autoaim_interfaces/msg/comm_send.hpp>
#include <autoaim_interfaces/msg/comm_recv.hpp>
#include <autoaim_interfaces/msg/detection.hpp>
#include <autoaim_interfaces/msg/debug_info.hpp>
#include <pnp_buff.hpp>
#include <ukf_buff.hpp>
#include <math_utils.hpp>
#include <fstream>
#include <select_utils.hpp>
#include <trajectory.hpp>
#include "filter.hpp"
#include <ceres/ceres.h>
#include <ceres/problem.h>
//解决滤波问题
//角度补偿的逻辑
//整套逻辑问题

   
namespace autoaim_buff
{
 
    using autoaim_interfaces::msg::CommRecv;
    using autoaim_interfaces::msg::CommSend;
    using autoaim_interfaces::msg::Detection;
    using autoaim_interfaces::msg::DetectionArray;
    using autoaim_interfaces::msg::DebugInfo;
    using namespace std::chrono;
    using namespace cv;
    using namespace std;
    using namespace Eigen;

    struct BuffData{
        float t;
        float angle;
        float speed;
            BuffData(float t_,float angle_,float speed_):t(t_), angle(angle_),speed(speed_){}
    };

    struct CURVE_FITTING_COST{
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}
    template <typename T>
    bool operator()(const T *const params, T *residual) const
    {
        
        T v = params[0]* ceres::sin(params[1] * T(_x) + params[2])+ params[3] ;
        //这里筛除异常值
        if (T(_y) < T(2.2) && T(_y) > T(0)) {
            residual[0] = T(_y) - v;
            return true;  // 成功计算残差
        } else {
            residual[0] = T(0);  // 设置残差为0
            return true;  // 跳过该残差项，Ceres将不会考虑此项
        }

    }
    const double _x, _y;
    };

    const geometry_msgs::msg::Transform EMPTY_TRANSFORM;
    // 将消息转换到秒
    double to_sec(builtin_interfaces::msg::Time t)
    {
        return t.sec + t.nanosec * 1e-9;
    }

    class BuffNode : public rclcpp::Node
    {
    public:
        // rclcpp::NodeOptions 为节点启动信息配置
        explicit BuffNode(const rclcpp::NodeOptions &options);
        ~BuffNode() = default;

    private:
        // 加载参数
        void get_parameters();
        void get_VTM_camera_param();
        //主回调函数
        void buff_callback(const DetectionArray::SharedPtr msg);
        // 获取相机参数信息
        void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
        // //动态tf树
        // void send_dynamic_tf_transforms(rclcpp::Time time_stamp);
        //计算buff角度
        float Angle(geometry_msgs::msg::Transform& buff_to_chassis_yaw);
        //float BuffNode::FinalAngle(float angle, float t, float dt)
        float cal_angle(Detection& buff_target,const rclcpp::Time& time_stamp);
        void Direction();
        //清理
        void clear();
        //拟合相关
        void FitCurveAsync();
        void FitCurve(const vector<BuffData> &historyBuffDataList_, const vector<float> &filteredAngleList_);
        float CalcRMSE(VectorXd params);
        float FinalAngle(float angle, float t_upper, float t_lower);
        float CalcRMSE();
        //从发来的消息中筛选是否有合适的未激活的buff，有且只有一个时返回真
        bool select_buff(std::vector<Detection> &src, Detection &dst);
        //照搬的
        cv::Point2f get_pretiction_VTM(const builtin_interfaces::msg::Time& header_stamp);
        cv::Point3f FinalPosition(float fianle_angle);
        cv::Point3f pred_target(
            const float real_angle,const float bullet_speed, 
            const float img_to_aim_time,const float t_now
        );
        // 尝试获取指定时间点对应的变换。若尝试MAX_ATTEMPTS后仍没有找到，返回EMPTY_TRANSFORM
        geometry_msgs::msg::Transform try_get_transform(
            const std::string &target,
            const std::string &source,
            const rclcpp::Time &time_point) const;
        //初始为init状态
        BUFF_STATUS buff_status = BUFF_STATUS::INITIATION;
        bool enable_predict_;
        bool can_shoot ;
        size_t lost_counts_ = 0; //找不到未激活符的帧数
        size_t predict_counts = -1;
        builtin_interfaces::msg::Time last_frameStamp, now_frameStamp, first_frameStamp;
        float frameDeltaTime;
        float time_till_now;
        float test_angle=0;
        float last_shoot_time=0;
        float real_angle,delta_angle=0,last_delta_angle,final_angle=0;
        float t_start,t_end,best_t_start,best_t_end;
        bool pre_flag=false;
        float v,f_v,v_t;
        Mat camera_CI_MAT;
        int followCounts=0;
        Mat R_flag = Mat::zeros(3, 1, CV_32F);  
        Mat buff_tvec = Mat::zeros(3, 1, CV_32F);  
        Mat buff_R = Mat::zeros(3, 3, CV_32F);  
        // mutex
        std::mutex mtx;
        std::future<void> futureResult;

        //最小二乘法拟合相关
        std::vector<BuffData> historyBuffDataList;
        std::vector<Mat> R_flag_list;
        std::vector<float> filteredAngleList;
        std::vector<double> den, num;
        double threshold;
        int direction = -1; 
        IIR filter;
        cv::Vec4d lower_bound, upper_bound;
        double iteration, rmse_threshold, min_v_threshold, max_rmse;
        int ignore_time, fit_time, follow_time, max_history_size;
        Eigen::Vector4d params, last_params;
        float now_angle,last_angle;
        float filter_now;
        float compensate_angle = 0;
        float buff_change_threshold,R_flag_num;
        float buff_noise_threshold;
        float a_weight,w_weight;
        cv::Mat VTM_intrinsic_,VTM_distortion_ ;
        float camera_cx, camera_fx, camera_cy, camera_fy;
        bool if_shoot;
        //待找出的目标buff
        Detection buff_target;
        Detection now_buff;
        Detection last_buff;
        // 获取串口节点信息
        float roll_;
        float pitch_;
        float yaw_;
        int mode_,last_mode_;
        float bullet_speed_;
        //debug
        bool if_predict;
        int target_color;
        int  count;
        int dir_time;
        int shoot_count=0,shoot_frames;
        float shoot_compensate_pitch_;
        float shoot_compensate_yaw_;
        float control_to_aim_time_;
        // 相机参数
        std::string camera_info_topic_;
        std::string buff_sub_topic_;
        std::string comm_pub_topic_;
        std::string comm_sub_topic_;
        std::string debug_send_topic_;
        std::string ukf_params_path;
        std::string ceres_params_path;
        std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
        std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

        std::shared_ptr<PnPBuff> pnp_solver_;
        std::shared_ptr<UkfBuff<1, 3, 3>> ukf_tracker;
        // sub
        std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>> camera_info_sub_;
        std::shared_ptr<rclcpp::Subscription<DetectionArray>> buff_sub_;
        std::shared_ptr<rclcpp::Subscription<CommRecv>> comm_recv_sub_;
        // pub
        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::CommSend>> comm_send_pub_;
        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::DebugInfo>> debug_send_pub_;
    };

    BuffNode::BuffNode(const rclcpp::NodeOptions &options) : Node("autoaim_buff", options)
    {
        tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        pnp_solver_ = std::make_shared<PnPBuff>();

        ukf_params_path = ament_index_cpp::get_package_share_directory("autoaim_buff") + "/config/ukf_params.yaml";
        ukf_tracker = std::make_shared<UkfBuff<1, 3, 3>>(ukf_params_path);

        ceres_params_path = ament_index_cpp::get_package_share_directory("autoaim_buff") + "/config/buff.yaml";
        get_parameters();
    }

    void BuffNode::get_parameters()
    {
       
        //加载buff参数
        FileStorage fs(ceres_params_path, FileStorage::READ);
        //分母
        cv::FileNode denNode = fs["den"];
        for (auto it = denNode.begin(); it != denNode.end(); ++it)
        {
            den.push_back(*it);
        }
        cv::FileNode numNode = fs["num"];
        for (auto it = numNode.begin(); it != numNode.end(); ++it)
        {
            num.push_back(*it);
        }
        fs["lower_bound"]       >> lower_bound;
        fs["upper_bound"]       >> upper_bound;
        fs["rmse_threshold"]    >> rmse_threshold;
        fs["iteration"]         >> iteration;
        fs["max_rmse"]          >> max_rmse;
        fs["min_v_threshold"]   >> min_v_threshold;
        fs["ignore_time"]       >> ignore_time;
        fs["fit_time"]          >> fit_time;
        fs["follow_time"]       >> follow_time;
        fs["max_history_size"]  >> max_history_size;
        fs.release(); 

        get_VTM_camera_param();
        a_weight = declare_parameter("a_weight", 0.0);
        w_weight = declare_parameter("w_weight", 0.0);
        filter.SetHypParam(den, num);
        params = {(0.784 + 1.045) / 2,(1.884 + 2) / 2,0,2.09 - (0.784 + 1.045) / 2};
        last_params = params;
        dir_time = declare_parameter("dir_time", 0);
        enable_predict_ = declare_parameter("enable_predict", true);
        bullet_speed_ = declare_parameter("bullet_speed", 30.0);
        shoot_compensate_pitch_ = math::d2r(declare_parameter("shoot_compensate_pitch", 0.0));
        shoot_compensate_yaw_ = math::d2r(declare_parameter("shoot_compensate_yaw", 0.0));
        control_to_aim_time_ = declare_parameter("control_to_aim_time_", 0.098);
        //buff
        buff_change_threshold = declare_parameter("buff_change_threshold", 50);
        buff_noise_threshold = declare_parameter("buff_noise_threshold", 10);
        shoot_frames = declare_parameter("shoot_frames", 10);
        // topic
        camera_info_topic_ = declare_parameter("camera_info_topic", "camera/color/camera_info");
        buff_sub_topic_ = declare_parameter<std::string>("buff_sub_topic_", "/buff/detection");
        comm_pub_topic_ = declare_parameter("comm_pub_topic", "/serial/comm_send");
        comm_sub_topic_ = declare_parameter("comm_recv_sub_topic", "/serial/comm_recv");
        debug_send_topic_ = declare_parameter("debug_send_topic_", "/buff/debug_msg");
        //debug
        int msg_num =  declare_parameter("msg_num", 1);
        if_shoot = declare_parameter("if_shoot", false);
        mode_ = declare_parameter("mode", 0);
        if_predict = declare_parameter("if_predict", false);


        camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic_,
            rclcpp::SensorDataQoS().keep_last(1),
            [&](const sensor_msgs::msg::CameraInfo::SharedPtr msg)
            { camera_info_callback(msg); });

        buff_sub_ = create_subscription<DetectionArray>(
            buff_sub_topic_,
            rclcpp::SensorDataQoS().keep_last(1),
            [&](const DetectionArray::SharedPtr msg)
            { buff_callback(msg); });

        comm_recv_sub_ = create_subscription<CommRecv>(
            comm_sub_topic_,
            rclcpp::SensorDataQoS().keep_last(1),
            [&](const CommRecv::SharedPtr msg)
            {
                roll_ = msg->roll;
                pitch_ = msg->pitch;
                yaw_ = msg->yaw;
                bullet_speed_ = msg->shoot_speed;
                //send_dynamic_tf_transforms(msg->header.stamp);
                //实际调试时要记得修改串口协议发送合适的mode
                //蓝1 红 2
                target_color = msg->target_color;
                last_mode_ = mode_;
                mode_ = msg->mode;
                 
                if(mode_!=last_mode_)
                {
                    cout<<"串口mode"<<mode_<<endl;
                    this->clear();
                }

            });
        comm_send_pub_ = create_publisher<CommSend>(
            comm_pub_topic_,
            rclcpp::SensorDataQoS().keep_last(1));

        debug_send_pub_ = create_publisher<DebugInfo>(
            debug_send_topic_,
            rclcpp::SensorDataQoS().keep_last(1));

    }

// 建议上面的shi都先别管了，先看函数
    void BuffNode::buff_callback(const DetectionArray::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "buff callback");
        //第一个是筛选函数，就是找到待激活对象，state为1就是找到了，放在buff_target中
        //找到了大小符
        bool state = select_buff(msg->detections, buff_target);
        if(state)
        {

            if (mode_ == BIG_MODE)
            {   
                //初始化
                if(buff_status == BUFF_STATUS::INITIATION)
                {
                    //初始化开始时间和当前时间
                    now_frameStamp = msg->header.stamp;
                    first_frameStamp = msg->header.stamp;
                    //初始化第一帧，这里now_angle是补偿到0-2pi的
                    //real_angle是真实角度
                    //test_angle是用差分法累积求出的角度（没有0-2pi的限制，每次加一个delta_angle)
                    //cal_angle求出当前真实角度
                    now_angle = cal_angle(buff_target,first_frameStamp);
                    real_angle = now_angle;
                    test_angle = real_angle;
                    //只有三个状态，初始化，跟踪，收敛，这里初始化完就进入跟踪
                    buff_status = BUFF_STATUS::TRACKING;
                    return;
                }
                //一定要执行的部分
                //更新上一次的记录量
                last_frameStamp = now_frameStamp;
                now_frameStamp = msg->header.stamp;
                //last_angle记录的是真实的角度的上一次值，now_angle后续会被做补偿处理成连续的
                last_angle = real_angle;
                now_angle = cal_angle(buff_target,now_frameStamp);
                real_angle = now_angle;
                //计算时间
                time_till_now = to_sec(now_frameStamp) - to_sec(first_frameStamp);
                frameDeltaTime = to_sec(now_frameStamp) - to_sec(last_frameStamp);
                //当前计算的速度对应的时间
                v_t = time_till_now + 0.5*frameDeltaTime;
                //检查超时 2s
                if(to_sec(now_frameStamp)-to_sec(last_frameStamp) > MAX_LOST_TIME)
                    clear();  
                //记录 上一次的变化角度，主要是防止噪声情况
                //只有当delta_angle在正常变化值内才会更新
                last_delta_angle =fabs(delta_angle)<buff_noise_threshold ? delta_angle :last_delta_angle ;
                //计算当前实际的变化角度
                delta_angle = real_angle - last_angle;
                //先判断是否有过0情况
                //实际上只是顺时针转了个小角度
                if (delta_angle > 330)
                    delta_angle -= 360;
                //实际上只是逆时针转了个小角度
                else if (delta_angle < -330)
                    delta_angle += 360;


               
                //切换进行角度补偿,这里要注意要减去一个近似的正常运动角
                //即补偿角度是当前变化的角度减去上一次变化角度，因为切换扇叶时扇叶实际上还是在运动的
                //可以用上一次的变化率近似代替这一次，这里是之前用位移法要做的
                if (fabs(delta_angle) > buff_change_threshold)
                {
                    compensate_angle += (delta_angle-last_delta_angle);
                }
                 //速度法涉及，如果delta_angle过大，认为忽略本次数据，用上次的变化量近似计算v和累积变化角度
                if(fabs(delta_angle)>buff_noise_threshold)
                {
                        test_angle += last_delta_angle;
                        v = D2R(last_delta_angle/frameDeltaTime);    
                }
                //如果是正常情况，则用当前的delta_angle更新计算
                else
                {
                    v = D2R(delta_angle/frameDeltaTime);
                    test_angle += delta_angle;
                }
                
                //这里补偿角不使用定值，而直接使用差值，这样出来的图像比较平滑
                now_angle -= compensate_angle;
                //限制幅度
                while(now_angle>360||now_angle<0)
                    now_angle = now_angle < 0 ? now_angle + 360 : ( now_angle > 360 ? now_angle - 360 :now_angle);
                //这里因为速度有正负，所以不好在一开始滤除绝对值较小的情况，现在放到拟合函数中滤除了
                f_v = filter.filter(v);
                if(abs(f_v)<2.2){
                    filteredAngleList.push_back(f_v);
                
                    BuffData history_(v_t, D2R(test_angle),v);
                    historyBuffDataList.push_back(history_);
                }
 
                cout
                << "  first_frameStamp  " << to_sec(first_frameStamp)
                << "  time_till_now  " << time_till_now
                << "  frameDeltaTime  " << frameDeltaTime
                << "  delta_angle  " << delta_angle
                << "    direction   "<< direction
                << "    count   "<< count
                << endl;
                //超过滑窗就动态更新
                if (historyBuffDataList.size() > max_history_size)
                {
                    historyBuffDataList.erase(historyBuffDataList.begin());
                    filteredAngleList.erase(filteredAngleList.begin());
                }
                //50帧时判断一次方向，5次统计正负方向角度多少
                if(historyBuffDataList.size() == dir_time)
                {
                    Direction();
                }
                //如果达到拟合条件（拟合思路基本全部和老代码一致）
                if (historyBuffDataList.size() > fit_time)
                {
                    //记录拟合帧数
                    followCounts++;
                    //拟合回调
                    FitCurveAsync();
                    //计算误差，优时才更新
                    double rmse = CalcRMSE();
                    cout<<"rmse"<<rmse<<endl;
                    float last_rmse = CalcRMSE(last_params);
                    if (last_rmse < rmse){
                        params = last_params;
                    }

                    if (followCounts > follow_time)
                            buff_status = BUFF_STATUS::CONVERGING;

                    last_params = params;
                }
            
                
                //调试部分
                DebugInfo debug_msg;
                debug_msg.a = params[0];
                debug_msg.w = params[1];
                debug_msg.w0 = params[2];
                debug_msg.b = params[3];

                cout
                    << "  a  " << params[0]
                    << "  w  " << params[1]
                    << "  w0  " << params[2]
                    << "  b  " << params[3]
                << endl;
                
                debug_msg.now_angle = now_angle;
                debug_msg.last_delta_angle = frameDeltaTime;
                debug_msg.delta_angle = v;
                debug_msg.time_till_now = time_till_now;
                debug_msg.theta0 = test_angle;
                debug_msg.real_angle = real_angle;
                debug_msg.com_angle = f_v;
                debug_msg.filtervalue =last_shoot_time;
                debug_send_pub_->publish(debug_msg);
     
            }

            //小符
            else if (mode_ == SMALL_MODE)
            {           

                if(buff_status == BUFF_STATUS::INITIATION)
                {
                    //初始化开始时间和当前时间
                    now_frameStamp = msg->header.stamp;
                    first_frameStamp = msg->header.stamp;
                    buff_status = BUFF_STATUS::TRACKING;
                    return;
                }
                real_angle = cal_angle(buff_target,msg->header.stamp);
                last_frameStamp = now_frameStamp;
                now_frameStamp = msg->header.stamp;
                frameDeltaTime = to_sec(now_frameStamp) - to_sec(last_frameStamp);
                time_till_now = to_sec(now_frameStamp) - to_sec(first_frameStamp);
                //检查超时 2s
                if( frameDeltaTime> MAX_LOST_TIME)
                    clear();  

                if(buff_status == BUFF_STATUS::TRACKING)
                {
                 if(historyBuffDataList.size()<dir_time)
                    {
                        //小符情况下速度无意义
                        BuffData history_(0, D2R(real_angle),0);
                        historyBuffDataList.push_back(history_);
                    }
                    if (historyBuffDataList.size() == dir_time)
                    {
                        Direction();
                        buff_status = BUFF_STATUS::CONVERGING;
                        historyBuffDataList.clear();
                        //这里必须return，否则这枪必歪
                        return;
                    }
                }
               
                DebugInfo debug_msg;
                debug_msg.last_delta_angle = frameDeltaTime;
                debug_msg.now_angle = real_angle;
                debug_msg.filtervalue =last_shoot_time;
                debug_send_pub_->publish(debug_msg);
        
            }     


            /*
            大符小符都共用的预测和击打代码
            */
            
            geometry_msgs::msg::TransformStamped target_to_chassis_yaw;
            target_to_chassis_yaw.header.stamp = msg->header.stamp;
            target_to_chassis_yaw.header.frame_id = "chassis_yaw";
            target_to_chassis_yaw.child_frame_id = "target_buff";
             //如果收敛了，就瞄准待击打位置，否则跟踪当前扇叶位置
            //if_predict为false的话，可以调试静止靶
            if(buff_status == BUFF_STATUS::CONVERGING&&(if_predict))
            {
                //实际上小符并未使用第四个参数，无需知道当前相对第一帧的时间
                //预测函数
                cv::Point3f target = pred_target(
                    real_angle,
                    bullet_speed_,
                    to_sec(now()) - to_sec(msg->header.stamp) + control_to_aim_time_,
                    to_sec(now()) - to_sec(first_frameStamp)
                    );
            
                target_to_chassis_yaw.transform.translation.x =  target.x ;
                target_to_chassis_yaw.transform.translation.y =  target.y ;
                target_to_chassis_yaw.transform.translation.z =  target.z ;
                cout
                    <<"frameDeltaTime"  <<frameDeltaTime
                    <<"   dir   "       <<direction
                    << "target color"   <<target_color
                    << "   mode   "     <<mode_
                    << "  real_angle  " << real_angle
                    << "  target.x  " << target.x
                    << "  target.y  " << target.y
                    << "  target.z  " << target.z
                    <<"R_flag"<<direction
                << endl;
            }
            //没有收敛或者调试静止靶就跟踪当前位置
            else
            {
                target_to_chassis_yaw.transform.translation.x =  buff_tvec.at<float>(0) ;
                target_to_chassis_yaw.transform.translation.y =  buff_tvec.at<float>(1) ;
                target_to_chassis_yaw.transform.translation.z =  buff_tvec.at<float>(2) ;
            }
            tf_broadcaster_->sendTransform(target_to_chassis_yaw);
            //图传
            cv::Point2f point_in_VTM = this->get_pretiction_VTM(msg->header.stamp);

            geometry_msgs::msg::Transform target_to_shoot;
                    try {
                        // shoot是原点在摩擦轮系，但姿态没有pitch和roll的系。解出来的角度方便控车
                        target_to_shoot = try_get_transform("shoot", "target_buff", msg->header.stamp);
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

            autoaim_interfaces::msg::CommSend comm_send;
            comm_send.target_find = true;
            //收敛80帧来一发
            //if_shoot没用了，常为ture就行
            if((buff_status == BUFF_STATUS::CONVERGING&&(if_shoot)))
            {
                if(shoot_count==0)
                {
                    can_shoot = true;
                    last_shoot_time = time_till_now;
                    shoot_count = shoot_frames;
                }else{
                    can_shoot = false;
                    shoot_count--;
                }
                
            }
            else
            {
                can_shoot = false;
            }

            //1 单发
            //cout<<"canshoot"<<can_shoot<<endl;
            cout<<"canshoot"<<can_shoot<<endl;
            comm_send.shoot_flag = can_shoot == true? 1:0;
            comm_send.pitch = math::r2d(target_pitch);
            comm_send.yaw = math::r2d(target_yaw);
            comm_send.vtm_x = point_in_VTM.x;
            comm_send.vtm_y = point_in_VTM.y;
            comm_send_pub_->publish(comm_send);
        }

        else
        {
            autoaim_interfaces::msg::CommSend comm_send;
            //comm_send.header.stamp = now();
            comm_send.target_find = false;
            comm_send.shoot_flag = 0;
            comm_send.pitch = math::r2d(pitch_);
            comm_send.yaw = math::r2d(yaw_);
            comm_send_pub_->publish(comm_send);  
            return;
        }
   

    }
    //判断方向，50个数据统计5组正负情况，比较鲁棒可靠
    void BuffNode::Direction()
    {
        count = 0;
        for(int i=1;i<=5;i++)
        {
            float delta = R2D(historyBuffDataList[i*10].angle-historyBuffDataList[(i-1)*10].angle);
            if (delta > 330)
                    delta -= 360;
                //实际上只是逆时针转了个小角度
                else if (delta < -330)
                    delta += 360;
            if(delta < 0)
                count--;
            else
                count++;
            cout<<"拟合count"<<count<<endl;
        }
        direction = count < 0 ? BUFF_DIRECTION::CW : BUFF_DIRECTION::CCW;

    }

    geometry_msgs::msg::Transform BuffNode::try_get_transform(
        const std::string &target,
        const std::string &source,
        const rclcpp::Time &time_point) const
    {
        constexpr int MAX_ATTEMPTS = 100;
        geometry_msgs::msg::Transform transform = geometry_msgs::msg::Transform(); // 初始化为空的Transform
        for (int i = 0; i < MAX_ATTEMPTS; i++)
        {
            try
            {
                // 获取变换
                transform = tf_buffer_->lookupTransform(target, source, time_point).transform;
                break; // 成功获取变换后跳出循环
            }
            catch (const std::exception &ex)
            {
                // 如果获取失败，等待1微秒后重试
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        }
        if (transform.translation.x == 0.0 && 
        transform.translation.y == 0.0 && 
        transform.translation.z == 0.0 && 
        transform.rotation.x == 0.0 && 
        transform.rotation.y == 0.0 && 
        transform.rotation.z == 0.0 && 
        transform.rotation.w == 1.0)
            {
                cout << "返回了空变换" << endl;
            }
        return transform; // 返回变换信息（包含平移和旋转）
    }
    //计算当前实际角度
    float BuffNode::Angle(geometry_msgs::msg::Transform& buff_to_chassis_yaw)
    {

        //先得到R和t
        buff_tvec.at<float>(0) = buff_to_chassis_yaw.translation.x;
        buff_tvec.at<float>(1) = buff_to_chassis_yaw.translation.y;
        buff_tvec.at<float>(2) = buff_to_chassis_yaw.translation.z;
        tf2::Quaternion quaternion(
        buff_to_chassis_yaw.rotation.x,
        buff_to_chassis_yaw.rotation.y,
        buff_to_chassis_yaw.rotation.z,
        buff_to_chassis_yaw.rotation.w
        );
        // 将四元数转换为旋转矩阵
        tf2::Matrix3x3 rotation_matrix(quaternion);
        // 将旋转矩阵转为 OpenCV 的 cv::Mat 类型
        // 将旋转矩阵转为 OpenCV 的 cv::Mat 类型
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                buff_R.at<float>(i, j) = rotation_matrix[i][j];
            }
        }
        //计算R标真实位置,R标位置在z轴负方向0.7m处
        Mat R_to_buff = (Mat_<float>(3, 1) << 0.0f, 0.0f, -0.7f);
        //R作一个均值滤波
        if(R_flag.at<float>(0, 0)==0)
        R_flag =buff_R*R_to_buff + buff_tvec;
        else
        R_flag =(R_flag+buff_R*R_to_buff + buff_tvec)/2.0f;
        // R_flag =buff_R*R_to_buff + buff_tvec;
        cout<<"buff_tvec"<<buff_tvec<<endl;
        cout<<"R_flag"<<R_flag<<endl;
        //计算当前角度位置
        //世界坐标系向左是y，向前是x，向上是z
        //这里认为从水平向右开始为0度，逆时针增加到360度
        float dy = (buff_tvec.at<float>(2)- R_flag.at<float>(2));
        float dx =-(buff_tvec.at<float>(1)- R_flag.at<float>(1));
        float angle = atan2(dy, dx);
        angle = R2D(angle);
        cout<<"angle"<<angle<<endl;
        /* 限制角度在[0. 360] */
        angle = angle < 0 ? angle + 360 : angle;
        return angle;
  
    }
     //将buff_target先pnp并转到chassis_yaw下，再调用上面那个函数计算实际角度
    float BuffNode::cal_angle(Detection& buff_target,const rclcpp::Time& time_stamp)
    {
        
        geometry_msgs::msg::TransformStamped buff_to_cam;
        buff_to_cam.header.stamp = time_stamp;
        buff_to_cam.header.frame_id = "autoaim_camera";
        buff_to_cam.child_frame_id = "buff";
        //pnp解算方法可能要调整
        pnp_solver_->solve_pnp(buff_target, buff_to_cam.transform);
        tf_broadcaster_->sendTransform(buff_to_cam);

         //进入滤波前得先转到云台坐标系下处理
        auto buff_to_chassis_yaw = try_get_transform("chassis_yaw", "buff", time_stamp);

        //计算出当前角度返回
        return Angle(buff_to_chassis_yaw);
    }
    //预测实际位置的函数
    cv::Point3f BuffNode::pred_target(
        const float real_angle,const float bullet_speed, 
        const float img_to_aim_time,const float t_now
        )
    {
        //和装甲板一样
        const float img_to_hit_time_1 = math::get_distance(buff_tvec) / bullet_speed;
        final_angle = FinalAngle(real_angle,t_now,img_to_hit_time_1);
        Point3f final_positon = FinalPosition(final_angle);
        const float img_to_aim_time_2 = img_to_aim_time + math::get_distance(final_positon) / bullet_speed;
        final_angle = FinalAngle(real_angle,t_now,img_to_aim_time_2);
        final_positon = FinalPosition(final_angle);
        return final_positon;

    }


    bool BuffNode::select_buff(std::vector<Detection> &src, Detection &dst) {

        int zero_label_count = 0;  // 用来统计label为0的Detection的数量
        bool found = false;
        // 遍历src向量，查找label为0的Detection
        for (const auto& detection : src) {
            if ((detection.label == 0)) {
                zero_label_count++;
                if (zero_label_count == 1) {
                    // 找到第一个label为0的Detection，赋值给dst
                    dst = detection;
                    found = true;
                }
            }
        }
        // 如果没有找到或者找到的个数不为1,则认为有问题
        if (zero_label_count != 1 || !found) {
            return false;
        }else{
            return true;
        }
        
    }

    void BuffNode::clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        historyBuffDataList.clear();
        filteredAngleList.clear();
        filter.clear();
        followCounts = 0;
        buff_status = BUFF_STATUS::INITIATION;
    }
    
    void BuffNode::FitCurveAsync(){
    std::lock_guard<std::mutex> lock(mtx);  // 确保复制数据时的线程安全
    auto historyData = historyBuffDataList;
    auto filteredData = filteredAngleList;
    if (!futureResult.valid() || futureResult.wait_for(milliseconds(0)) == future_status::ready)
    {
        // futureResult.get();  // 获取结果以处理可能的异常
        futureResult = std::async(std::launch::async, &BuffNode::FitCurve, this, historyData, filteredData);
    }

}


    void BuffNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        pnp_solver_->set_cam_matrix(
            cv::Mat(3, 3, CV_64F, msg->k.data()),
            cv::Mat(1, 5, CV_64F, msg->d.data()));
        // 相机内参和畸变在运行中不会改变，所以设置后即可取消camera_info订阅
        camera_info_sub_.reset();
        camera_info_sub_ = nullptr;
    }

    cv::Point2f BuffNode::get_pretiction_VTM(const builtin_interfaces::msg::Time& header_stamp)
    {
        geometry_msgs::msg::Transform predicted_to_VTM_transform;
        try {
            predicted_to_VTM_transform = this->try_get_transform(
                "VTM",
                "target_buff",
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

    void BuffNode::get_VTM_camera_param(){
        cout<<"图传已设置"<<endl;
        cv::FileStorage cameraConfig(ament_index_cpp::get_package_share_directory("autoaim_buff") + 
        "/config/VTM_camera_params.yaml", cv::FileStorage::READ);
        cameraConfig["camera_CI_MAT"] >> this->VTM_intrinsic_;
        cameraConfig["camera_D_MAT"] >> this->VTM_distortion_ ;
    }

    void BuffNode::FitCurve(const vector<BuffData> &historyBuffDataList_, const vector<float> &filteredAngleList_)
    {
        // 拟合
        ceres::Problem problem;
        ceres::Solver::Options options;
        // 输出
        ceres::Solver::Summary summary;
        double fit_params[4] = {params[0], params[1], params[2], params[3]};
         // 抛弃前面的数据
        for(int i=ignore_time;i<historyBuffDataList_.size();i++){

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 4>(
                    new CURVE_FITTING_COST(historyBuffDataList_[i].t, direction * filteredAngleList_[i])),
                new ceres::CauchyLoss(0.5),
                fit_params);
        }
        options.minimizer_progress_to_stdout = false;
        // 限制参数范围
        for (int i = 0; i < 4; i++)
        {
            problem.SetParameterLowerBound(fit_params, i, lower_bound[i]);
            problem.SetParameterUpperBound(fit_params, i, upper_bound[i]);
        }
         /* 防止特别离谱的结果出现 */
        float rmse_ep = 1e1;
        int iter = 0, maxIter = iteration;
        do
        {
            iter++;
            ceres::Solve(options, &problem, &summary);
            // 赋值
            VectorXd fit_parmas_vec = VectorXd::Map(fit_params, 4);
            rmse_ep = CalcRMSE(fit_parmas_vec);
        } while (rmse_ep > max_rmse && maxIter > iter);
        params = Eigen::Vector4d(fit_params[0], fit_params[1], fit_params[2], fit_params[3]);
    }

    float BuffNode::CalcRMSE(VectorXd params)
    {

        float rmse = 0;
        for (auto &data : historyBuffDataList)
        {
            float t = data.t;
            float speed = data.speed;
            float predSpeed = params[0] * sin(params[1] * t + params[2]) + params[3];
            rmse += pow2(predSpeed - speed);
        }
        return sqrt(rmse / historyBuffDataList.size());
    }

    float BuffNode::CalcRMSE( )
    {
        float rmse = 0;
        for (auto &data : historyBuffDataList)
        {
            float t = data.t;
            float speed = data.speed;
            float predSpeed = params[0] * sin(params[1] * t + params[2]) + params[3];
            rmse += pow2(predSpeed - speed);
        }
        return sqrt(rmse / historyBuffDataList.size());
    }

    //根据当前角度和预测时间来计算最终角度
    float BuffNode::FinalAngle(float angle, float t, float dt)
    {
        float offset;
        if(mode_==BIG_MODE)
        {
            float a = params[0];
            float w = params[1];
            float theta = params[2];
            float b = params[3];
            //这里都有个C可以消去
            float lower = -a / w * cos(w * t + theta) + b * t;
            float upper = -a / w * cos(w * (t+dt) + theta) + b * (t+dt);
            offset = R2D(upper - lower);
        }else{
            offset = 60*dt;
        }
        angle += direction * offset;
        angle = angle < 0 ? angle + 360 : ( angle > 360 ? angle - 360 :angle);
        return angle;
    }
    //将最终角度通过R标计算出扇叶位置
    cv::Point3f BuffNode::FinalPosition(float fianle_angle)
    {

        cv::Point3f final_position;
        fianle_angle = D2R(fianle_angle);
        final_position.x = R_flag.at<float>(0) ;
        final_position.y = R_flag.at<float>(1) - 0.7*std::cos(fianle_angle);
        final_position.z = R_flag.at<float>(2) + 0.7*std::sin(fianle_angle);
        return final_position;
    }

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_buff::BuffNode)