std_msgs/Header header

# Prediction node state
int32 target_color
int32 target_label
float32 bullet_speed
bool[] is_enemy_can_shoot

# Tracker state
int32 tracker_status
uint32 appear_frames
uint32 track_frames
uint32 lost_frames
uint32 observing_armor_id
geometry_msgs/Point32 kf_xyz_position
geometry_msgs/Point32 kf_xyz_velocity
geometry_msgs/Point32 ukf_xy_position # 只使用x和y
geometry_msgs/Point32 ukf_xy_velocity # 只使用x和y
float32 kf_yaw
float32 kf_yaw_palstance
float32[] radius
float32[] height

# Buff
float64 filtervalue
int8 direction
float64 final_angle
int32 recognized_counts

# Big Buff
float64 a
float64 w
float64 w0
float64 b
float32 now_angle
float32 last_delta_angle
float32 delta_angle
float32 v
float32 real_angle
float32 com_angle

#curve
float32 time_till_now
float32 predangle
float32 theta0