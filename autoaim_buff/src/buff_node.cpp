#include <buff.hpp>

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
        this->get_parameters();
    }


    //主回调函数
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
                //检查超时 5s
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
    
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_buff::BuffNode)