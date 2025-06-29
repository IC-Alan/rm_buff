#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <autoaim_interfaces/msg/comm_recv.hpp>

#include "rclcpp_lifecycle/lifecycle_node.hpp"
constexpr float d2r(const float deg) {
    return deg * M_PI / 180.0;
}

namespace autoaim_robot_description {

class RobotDescriptionNode : public rclcpp::Node {
public:
  explicit RobotDescriptionNode(const rclcpp::NodeOptions& options)
  : Node("autoaim_robot_description", options) {
    
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    std::string comm_recv_sub_topic = declare_parameter("comm_recv_sub_topic","/serial/comm_recv");
    comm_recv_sub_ = create_subscription<autoaim_interfaces::msg::CommRecv>(
      comm_recv_sub_topic,
      rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&RobotDescriptionNode::comm_recv_callback, this, std::placeholders::_1)
    );

    send_static_transforms();
  }

  virtual ~RobotDescriptionNode() = default; 

private:
  void send_static_transforms() {
    const double cam_to_gimbal_x = declare_parameter("cam_to_gimbal_x", 0.0);
    const double cam_to_gimbal_y = declare_parameter("cam_to_gimbal_y", 0.0);
    const double cam_to_gimbal_z = declare_parameter("cam_to_gimbal_z", 0.0);
    
    geometry_msgs::msg::TransformStamped cam_to_gimbal;
    cam_to_gimbal.header.stamp = this->now();
    cam_to_gimbal.header.frame_id = "gimbal";
    cam_to_gimbal.child_frame_id = "autoaim_camera";
    cam_to_gimbal.transform.translation.x = cam_to_gimbal_x;
    cam_to_gimbal.transform.translation.y = cam_to_gimbal_y;
    cam_to_gimbal.transform.translation.z = cam_to_gimbal_z;
    cam_to_gimbal.transform.rotation.w = 1.0;
    
    static_tf_broadcaster_->sendTransform(cam_to_gimbal);


    const double VTM_to_cam_x_ = declare_parameter("VTM_to_cam_x", 0.0);
    const double VTM_to_cam_y_ = declare_parameter("VTM_to_cam_y", 0.0);
    const double VTM_to_cam_z_ = declare_parameter("VTM_to_cam_z", 0.0);

    geometry_msgs::msg::TransformStamped VTM_to_cam;
    VTM_to_cam.header.stamp = this->now();
    VTM_to_cam.header.frame_id = "autoaim_camera";
    VTM_to_cam.child_frame_id = "VTM";
    VTM_to_cam.transform.translation.x = VTM_to_cam_x_;
    VTM_to_cam.transform.translation.y = VTM_to_cam_y_;
    VTM_to_cam.transform.translation.z = VTM_to_cam_z_;
    // 这里认为相机坐标系向右是x，向下是y，向前是z。世界坐标系向右是x，向前是y，向上是z。
    VTM_to_cam.transform.rotation.x = 0.5;
    VTM_to_cam.transform.rotation.y = -0.5;
    VTM_to_cam.transform.rotation.z = 0.5;
    VTM_to_cam.transform.rotation.w = -0.5;

    static_tf_broadcaster_->sendTransform(VTM_to_cam);


    const double fric_to_gimbal_x = declare_parameter("fric_to_gimbal_x", 0.0);
    const double fric_to_gimbal_y = declare_parameter("fric_to_gimbal_y", 0.0);
    const double fric_to_gimbal_z = declare_parameter("fric_to_gimbal_z", 0.0);

    geometry_msgs::msg::TransformStamped fric_to_gimbal;
    fric_to_gimbal.header.stamp = this->now();
    fric_to_gimbal.header.frame_id = "gimbal";
    fric_to_gimbal.child_frame_id = "fric";
    fric_to_gimbal.transform.translation.x = fric_to_gimbal_x;
    fric_to_gimbal.transform.translation.y = fric_to_gimbal_y;
    fric_to_gimbal.transform.translation.z = fric_to_gimbal_z;
    fric_to_gimbal.transform.rotation.x = 0;
    fric_to_gimbal.transform.rotation.y = 0;
    fric_to_gimbal.transform.rotation.z = 0;
    fric_to_gimbal.transform.rotation.w = 1;

    static_tf_broadcaster_->sendTransform(fric_to_gimbal);
  }

  void send_dynamic_transforms(const rclcpp::Time& time_stamp, const double roll, const double pitch, const double yaw) {
    geometry_msgs::msg::TransformStamped gimbal_to_chassis;
    gimbal_to_chassis.header.stamp = time_stamp;
    gimbal_to_chassis.header.frame_id = "chassis";
    gimbal_to_chassis.child_frame_id = "gimbal";
    
    tf2::Quaternion q_gimbal_chassis;
    q_gimbal_chassis.setRPY(roll, pitch, yaw);
    
    gimbal_to_chassis.transform.rotation.x = q_gimbal_chassis.getX();
    gimbal_to_chassis.transform.rotation.y = q_gimbal_chassis.getY();
    gimbal_to_chassis.transform.rotation.z = q_gimbal_chassis.getZ();
    gimbal_to_chassis.transform.rotation.w = q_gimbal_chassis.getW();
    
    tf_broadcaster_->sendTransform(gimbal_to_chassis);

    geometry_msgs::msg::TransformStamped gimbal_to_chassis_yaw;
    gimbal_to_chassis_yaw.header.stamp = time_stamp ;
    gimbal_to_chassis_yaw.header.frame_id = "chassis";
    gimbal_to_chassis_yaw.child_frame_id = "chassis_yaw";
    
    tf2::Quaternion q_gimbal_chassis_yaw;
    q_gimbal_chassis_yaw.setRPY(0, 0, yaw);

    gimbal_to_chassis_yaw.transform.rotation.x = q_gimbal_chassis_yaw.getX();
    gimbal_to_chassis_yaw.transform.rotation.y = q_gimbal_chassis_yaw.getY();
    gimbal_to_chassis_yaw.transform.rotation.z = q_gimbal_chassis_yaw.getZ();
    gimbal_to_chassis_yaw.transform.rotation.w = q_gimbal_chassis_yaw.getW();
    
    tf_broadcaster_->sendTransform(gimbal_to_chassis_yaw);

    geometry_msgs::msg::TransformStamped shoot_to_fric;
    shoot_to_fric.header.stamp = time_stamp;
    shoot_to_fric.header.frame_id = "fric";
    shoot_to_fric.child_frame_id = "shoot";

    tf2::Quaternion q_shoot_fric;
    q_shoot_fric.setRPY(roll, pitch, yaw); 

    shoot_to_fric.transform.rotation.x = -q_shoot_fric.getX();
    shoot_to_fric.transform.rotation.y = -q_shoot_fric.getY();
    shoot_to_fric.transform.rotation.z = -q_shoot_fric.getZ();
    shoot_to_fric.transform.rotation.w = q_shoot_fric.getW();

    tf_broadcaster_->sendTransform(shoot_to_fric);
  }

  void comm_recv_callback(const autoaim_interfaces::msg::CommRecv::SharedPtr msg) {
    //if(abs(msg->pitch) < 45.0) 
    send_dynamic_transforms(msg->header.stamp, d2r(msg->roll), d2r(msg->pitch), d2r(msg->yaw));
  }

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
  std::shared_ptr<rclcpp::Subscription<autoaim_interfaces::msg::CommRecv>> comm_recv_sub_;
};

}  // namespace autoaim_robot_description

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_robot_description::RobotDescriptionNode)