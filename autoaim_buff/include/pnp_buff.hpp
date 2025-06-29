#pragma once

#include <Eigen/Dense>
#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <autoaim_interfaces/msg/detection.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
using namespace cv;
class PnPBuff {
public:
    /*!
        @brief 使用ENPN进行pnp解算，IPPE求解角度差效果有限
        @param buff_detection 输入符4个角点位置
        @param transform 输出的buff坐标系到相机坐标系的变换。
        @return 
        @attention 相机坐标系定义与opencv一致（向右是x，向下是y，向前是z），符的坐标系定义（向右为x，向上为y,朝向相机方向为z）
    */
    bool solve_pnp(
        const autoaim_interfaces::msg::Detection& detection,
        geometry_msgs::msg::Transform& transform
    ) const {
        const auto& world_points = BUFF_POINTS;
        const std::vector<cv::Point2d> img_points {
            {detection.tl.x, detection.tl.y},
            {detection.bl.x, detection.bl.y},
            {detection.br.x, detection.br.y},
            {detection.tr.x, detection.tr.y}
        };
    
        std::vector<cv::Mat> rvecs(2), tvecs(2);
        // IPPE法解平移向量
        const int solutions_IPPE = cv::solvePnPGeneric(
            world_points,
            img_points,
            cam_intrinsic_,
            cam_distortion_,
            rvecs,
            tvecs,
            false,
            cv::SOLVEPNP_IPPE
        );
       

        int solution_index = 0;
        if (solutions_IPPE >= 1 && tvecs[0].at<double>(2) > 0) {
            solution_index = 0;
        } else if (solutions_IPPE >= 2 && tvecs[1].at<double>(2) > 0) {
            solution_index = 1;
        } else {
            solution_index=0;
        }
        for (int i = 0; i < 2; i++) {
        rvecs[i].convertTo(rvecs[i], CV_32F);
        tvecs[i].convertTo(tvecs[i], CV_32F);
        }   

        // 左乘即可把opencv的相机系（右x，下y，前z）转成我们在tf2中的相机系（前x，左y，上z）
        const Eigen::Quaternionf cv_to_tf(-0.5, 0.5, -0.5, 0.5);
        Eigen::Vector3f rvec;
        cv::cv2eigen(rvecs[solution_index], rvec);
        Eigen::Quaternionf rotation(Eigen::AngleAxisf(rvec.norm(),
                                     rvec.normalized()));
        rotation = cv_to_tf * rotation;
        // 将 Eigen::Quaternionf 转换为 tf2::Quaternion
        tf2::Quaternion tf_quaternion(rotation.x(), rotation.y(), rotation.z(), rotation.w());


        transform.rotation.x = tf_quaternion.getX();
        transform.rotation.y = tf_quaternion.getY();
        transform.rotation.z = tf_quaternion.getZ();
        transform.rotation.w = tf_quaternion.getW();

        // 把opencv pnp的tvec转到我们的tf2系下
        Eigen::Vector3f translation;
        cv::cv2eigen(tvecs[solution_index], translation);
        translation = cv_to_tf * translation;

        // 3. 将 Eigen::Vector3d 转换为 geometry_msgs::msg::Transform 里的平移部分
        transform.translation.x = translation.x();  // 使用 Eigen::Vector3d 的成员方法获取 x
        transform.translation.y = translation.y();  // 使用 Eigen::Vector3d 的成员方法获取 y
        transform.translation.z = translation.z();  // 使用 Eigen::Vector3d 的成员方法获取 z
        
        std::cout<<"pnp的X_Y_Z:"
        <<translation.x() <<" "
        << translation.y() <<" "
        << translation.z() <<" "
        << std::endl;
        // //EPNP法解旋转矩阵
        // const int solutions_ENPN = cv::solvePnPGeneric(
        //     world_points,
        //     img_points,
        //     cam_intrinsic_,
        //     cam_distortion_,
        //     rvecs,
        //     tvecs,
        //     false,
        //     cv::SOLVEPNP_EPNP
        // );
        // Mat R;
        // cv::Rodrigues(rvecs[0], R);
        // tf2::Matrix3x3 rotation_matrix(
        //     R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        //     R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        //     R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2)
        // );
        // // std::cout<<"原始的R:"<<R<<std::endl;
        // tf2::Quaternion quaternion;
        // rotation_matrix.getRotation(quaternion);
        // transform.rotation.x = quaternion.getX();
        // transform.rotation.y = quaternion.getY();
        // transform.rotation.z = quaternion.getZ();
        // transform.rotation.w = quaternion.getW();
        return 0;
    }

    /*!
        @brief 设置相机的内参矩阵和畸变矩阵
        @attention 算PnP前一定要先设置这个
    */
    void set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion) {
        cam_intrinsic_ = intrinsic.clone();
        cam_distortion_ = distortion.clone();
        std::cout<<"相机参数已设置"<<std::endl;
    }


    // 单位: 米
    // static constexpr float SMALL_HEIGHT = 0.144;
    // static constexpr float BIG_HEIGHT = 0.173;
    // static constexpr float SMALL_WIDTH = 0.160;
    // static constexpr float BIG_WIDTH = 0.186;
    static constexpr float R_BUFF = 0.125;
    // 相机向左y，向前x，向上z
    const std::vector<cv::Point3f> BUFF_POINTS {
        {0 , 0 ,R_BUFF},
        {0 , R_BUFF ,0},
        {0 , 0 ,-R_BUFF},
        {0 , -R_BUFF ,0},
    };


    cv::Mat cam_intrinsic_ =
(cv::Mat_<double>(3, 3) << 3.036053, 0, 1920, 
                        0, 5.39742756, 1080, 
                        0, 0, 1);
;
    cv::Mat cam_distortion_ = (cv::Mat_<double>(1, 5) << 0 ,0 ,0, 0, 0);
};