#ifndef __UNSCENTED_KALMAN_FILTER_HPP__
#define __UNSCENTED_KALMAN_FILTER_HPP__
#include "Eigen/Dense"
#include <math.h>
#include <glog/logging.h>
#include <tf2/convert.h>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <select_utils.hpp>
#include <math_utils.hpp>

using namespace Eigen;
using namespace std;


// 速度参数方程为a*sin(wt+theta)+2.09-a
// 积分得到位移方程为-a/w*cos(wt+theta)+2.09t-a*t+C
//三个状态量 a，w，theta
template<int V_Z = 1, int V_X = 3, int n = 3>
class UkfBuff
{
public:
    using Mat_zz = Matrix<float, V_Z, V_Z>;
    using Mat_xx = Matrix<float, V_X, V_X>;
    using Mat_zx = Matrix<float, V_Z, V_X>;
    using Mat_xz = Matrix<float, V_X, V_Z>;
    using Mat_x1 = Matrix<float, V_X, 1>;
    using Mat_z1 = Matrix<float, V_Z, 1>;
    using Mat_xn = Matrix<float, V_X, 2*n+1>;
    using Mat_zn = Matrix<float, V_Z, 2*n+1>;
    Mat_xx F;
    Mat_x1 x_k1;
    Mat_x1 P_x_k1;
    Mat_z1 z_k1;
    Mat_xz K;
    Mat_zx H;
    Mat_xx R;
    Mat_zz Q;
    Mat_xx P;
    Mat_xx P_;
    Mat_zz Pz;
    Mat_xx Px;
    Mat_xn x;
    Mat_zn z;
 
    float Wc[2*n+1];
    float Wm[2*n+1];
    float alpha = 0.1;
    int belta = 2;
    float lanmuda = 3 - n;
    float r = 0.65;  
    // void init(Mat_x1 x_k1, Mat_zz Q, Mat_xx R, Mat_xx P)
    // {
    //     this->x_k1 = x_k1;
    //     this->Q = Q;
    //     this->R = R;
    //     this->P = P;
    // }

   UkfBuff(const std::string& params_path) {
        load_params(params_path);
    }
    void load_params(const std::string& params_path) {
        cv::FileStorage fs(params_path, cv::FileStorage::READ);
        cv::Mat process_noise, measurement_noise,state_cov;
        fs["UKFBUFF"]["process_noise_cov"] >> process_noise;
        fs["UKFBUFF"]["measurement_noise_cov"] >> measurement_noise;
        fs["UKFBUFF"]["state_cov"] >> state_cov;
        cv::cv2eigen(process_noise, R);
        cv::cv2eigen(measurement_noise, Q);
        cv::cv2eigen(state_cov, P);
        x_k1 << 0.9125, 1.942, 0;
        fs.release();
        std::cout << "ukf滤波器加载正常" << std::endl;
    }

    //状态转移方程
    //状态量->先验预测量
    // 速度参数方程为a*sin(wt+theta)+2.09-a
    // 积分得到位移方程为-a/w*cos(wt+theta)+2.09t-a*t+C
    //三个状态量 a，w，theta
    Mat_x1 F_predict(Mat_x1 x_k1)
    {
        return x_k1;
    }
    //先验预测量->预测的观测值
    Mat_z1 H_(Mat_x1 x_k1, float t)
    {
        // 计算位移公式 位移为-a / omega * cos(omega * t1 + w0)+(2.09-a)*t+c
        Mat_z1 z;
        z[0] = x_k1[0]*sin(x_k1[1]*t+x_k1[2])+2.09-x_k1[0];  
        return z;
    }

    void reset(const std::string& params_path)
    {
        load_params(params_path);
    }

    void update(float t, Mat_z1 z_k)
    {

        //权重初始化
        Wm[0] = lanmuda / (n + lanmuda);
        Wc[0] = Wm[0] + 1 - alpha*alpha + belta;
        for(int i = 1; i <= 2*n; i++)
        {
            Wm[i] = 1 / (2 * (n + lanmuda));
            Wc[i] = 1 / (2 * (n + lanmuda));
        }

        //获得预测sigma点
        Mat_xx P1;
        x.fill(0);
        x.col(0) = F_predict(x_k1);
        P1 = P.llt().matrixL();
        P1 = sqrt(lanmuda + n) * P1;
        //std::cout << "P1: " <<P1<< std::endl;
        for(int i = 1; i <= n; i++)
        {
            x.col(i) = F_predict(x_k1 + P1.col(i-1));
            x.col(i + n) = F_predict(x_k1 - P1.col(i-1));
        }

        //无迹变换,更新先验状态量
        P_x_k1.fill(0);
        for(int i = 0; i <= 2*n ; i++)
        {
            P_x_k1 += Wm[i] * x.col(i);
        }

        //更新协方差
        P_ = R;
        Mat_x1 x_differ;
        for(int i = 0; i <= 2*n ; i++)
        {
            x_differ = x.col(i) - P_x_k1;
            P_ += Wc[i] * x_differ * x_differ.transpose();
        }
       // std::cout << "P_: " <<P_<< std::endl;
        //更新观测均值和协方差
        //获取观测值
        z.fill(0);
        for(int i = 0; i <= 2*n; i++)
        {   
            z.col(i) = H_(x.col(i), t);
        }
       // std::cout << "z: " <<z<< std::endl;
        //计算均值
        z_k1.fill(0);
        for(int i = 0; i <= 2*n; i++)
        {
            z_k1 += Wm[i] * z.col(i);
        }
       // std::cout << "z_k1: " <<z_k1<< std::endl;
        //计算协方差
        Mat_z1 z_differ;
        Pz = Q;
        for(int i = 0; i <= 2*n; i++)
        {
            z_differ = z.col(i) - z_k1;
            Pz +=Wc[i] * z_differ * z_differ.transpose();
        }
        //计算观测值和预测值的交叉协方差
        Mat_xz Pxy;
        Pxy.fill(0);
        for(int i = 0; i <=2*n; i++)
        {
            z_differ = z.col(i) - z_k1;
            x_differ = x.col(i) - P_x_k1;
            // if(i==0)
            // {
            // std::cout << "x_differ: " <<x_differ<< std::endl;
            // std::cout << "x.col(i): " <<x.col(i)<< std::endl;
            // std::cout << "P_x_k1: " <<P_x_k1<< std::endl;
            // }
            Pxy += Wc[i] * x_differ * z_differ.transpose();
        }
        // std::cout << "x: " <<x<< std::endl;
        // std::cout << "P_x_k1: " <<P_x_k1<< std::endl;
       
        // std::cout << "z_differ: " <<z_differ<< std::endl;
        // std::cout << "Pxy: " <<Pxy<< std::endl;
        // std::cout << "Pz: " <<Pz<< std::endl;
        //计算卡尔曼增益
        K = Pxy* Pz.inverse();
        //更新状体估计
        Mat_z1 z_measure_differ = z_k - z_k1;
        x_k1 = P_x_k1 + K * z_measure_differ;

        //更新协方差
        P = P_ - K * Pz * K.transpose();
    }
    
     
    /*下面是预测部分*/

    void predict(int mode_,const float bullet_speed, const float time_delay,
    geometry_msgs::msg::Transform& pre_buff_to_cam,
    const Mat& R, const Mat& tvec,const float& t)
    {   
        //std::cout << "进入predict" << std::endl;
        //计算hit_time
        Mat Pred_to_buff = cal_pred_to_buff(t,time_delay,mode_);
        //std::cout << "predict第一处正常" << std::endl;
        //std::cout << "R" << R<<std::endl;
        //std::cout << "Pred_to_buff" << Pred_to_buff<<std::endl;
        Mat Pred_to_cam = R * Pred_to_buff + tvec;
        float distance = cal_distance(Pred_to_cam);
        //std::cout << "predict第二处正常" << std::endl;
        // cout<<"Pred_to_cam"<<Pred_to_cam<<endl;
        // cout<<"R"<<R<<endl;
        // cout<<"tvec"<<tvec<<endl;
        
        float hit_time = time_delay + distance / bullet_speed;
        // cout<<"time_delay"<<time_delay<<endl;
        // cout<<"distance"<<distance<<endl;
        // cout<<"bullet_speed"<<bullet_speed<<endl; 
        // cout<<"hit_time"<<hit_time<<endl;
        //计算hit_time后符的位置
        Pred_to_buff = cal_pred_to_buff(t,hit_time,mode_);
        Pred_to_cam = R * Pred_to_buff + tvec;
        // std::cout << "R" << R<<std::endl;
        // std::cout << "Pred_to_buff" << Pred_to_buff<<std::endl;
        // std::cout << "tvec" << tvec<<std::endl;
        //赋值给传入的tf变换
        pre_buff_to_cam.translation.x = Pred_to_cam.at<double>(0);
        pre_buff_to_cam.translation.y = Pred_to_cam.at<double>(1);
        pre_buff_to_cam.translation.z = Pred_to_cam.at<double>(2);
        // 设置姿态为单位四元数（没有旋转）  
        pre_buff_to_cam.rotation.x = 0.0;  // 四元数的x分量  
        pre_buff_to_cam.rotation.y = 0.0;  // 四元数的y分量  
        pre_buff_to_cam.rotation.z = 0.0;  // 四元数的z分量  
        pre_buff_to_cam.rotation.w = 1.0;  // 四元数的w分量 

    }

    Mat cal_pred_to_buff(float t,float time_delay,int mode)
    {
        float d_theta = pre_rotation_angle(t,time_delay,mode);
        //std::cout << "d_theta" << d_theta<<std::endl;
        float pred_x = -r*sin(d_theta);
        float pred_y = -r*(1-cos(d_theta));
        float pred_z = 0;
        Mat Pred_to_buff = (Mat_<double>(3, 1) << pred_x, pred_y, pred_z);
        return Pred_to_buff;

    }
    float pre_rotation_angle(float t,float dt,int mode)
    {

        float d_theta;
        if(mode==BIG_MODE)
        {
            // 计算位移公式 位移为-a / omega * cos(omega * t1 + w0)+(2.09-a)*t+c
            // 算角度差可以忽略c
            float theta_now = -x_k1[0] / x_k1[1] * cos(x_k1[1] * t + x_k1[2])+(2.09 - x_k1[0]) * t; 
            float theta_pre = -x_k1[0] / x_k1[1] * cos(x_k1[1] * (t + dt) + x_k1[2])+(2.09 - x_k1[0]) * (t + dt); 
            d_theta = theta_pre-theta_now;
        }
        else
        {
            d_theta = pi/3*dt;
        }
        return d_theta;

    }
    float cal_distance(Mat position)
    {
        float x0 = position.at<double>(0);
        float x1 = position.at<double>(1);
        float x2 = position.at<double>(2);
        return sqrt(x0 * x0 + x1 * x1 + x2 * x2);
    }

};


#endif
