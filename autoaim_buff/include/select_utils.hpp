#ifndef SELECT_UTILS_HPP
#define SELECT_UTILS_HPP


#define PI 3.1415926
#define MAX_LOST_TIME 5
#define CONVERGING_FRAMES 1000
inline float pow2(float a) {
    return a * a;
}
enum class BUFF_STATUS
{
    INITIATION,
    CONVERGING,
    TRACKING,
    TEMP_LOST,
    LOST
};
enum 
{
    SMALL_MODE=0x03,
    BIG_MODE=0x04
};
enum BUFF_DIRECTION
{
    CW = -1, // 顺时针
    CCW = 1  // 逆时针
};


#endif

