#pragma once

#include <cstdint>
#include <autoaim_interfaces/msg/comm_send.hpp>
#include "crc_checksum.hpp"
#include <serial_utils.hpp>

class InfantrySendPacket {
public:
    InfantrySendPacket() = default;
    ~InfantrySendPacket() = default;

    size_t get_size() {
        return sizeof(SendPacket);
    }

    void from_msg(const autoaim_interfaces::msg::CommSend::SharedPtr& msg) {
        sendPacket.pitchAngle = static_cast<int16_t>(msg->pitch * 100);
        sendPacket.yawAngle = static_cast<int16_t>(msg->yaw * 100);
        sendPacket.mixedData = (msg->shoot_flag) | (msg->target_num << 4) |(msg->target_find <<2);
        sendPacket.VTMx = static_cast<int16_t>(msg->vtm_x);
        sendPacket.VTMy = static_cast<int16_t>(msg->vtm_y);
        sendPacket.CRC8 = Get_CRC8_Check_Sum(reinterpret_cast<uint8_t *>(this) + sizeof(sendPacket.header), sizeof(sendPacket) - sizeof(sendPacket.header) - sizeof(sendPacket.CRC8) - sizeof(sendPacket.tailer), 0);
    }

    void from_debug() {
        sendPacket.pitchAngle = 0;
        sendPacket.yawAngle = 0;
        sendPacket.mixedData = 0;
        sendPacket.VTMx = 0;
        sendPacket.VTMy = 0;
        sendPacket.CRC8 = Get_CRC8_Check_Sum(reinterpret_cast<uint8_t *>(this) + sizeof(sendPacket.header), sizeof(sendPacket) - sizeof(sendPacket.header) - sizeof(sendPacket.CRC8) - sizeof(sendPacket.tailer), 0);
    }
    
    std::vector<uint8_t> get_vector() {
        return serial_utils::to_vector(sendPacket);
    }

private:
    struct SendPacket
    {
        uint8_t header[3] = {0xaa, 0xbb, 0xcc};
        int16_t pitchAngle, yawAngle;
        uint8_t mixedData;
        int16_t VTMx, VTMy;
        uint8_t CRC8;
        uint8_t tailer = 0xff;
    } __attribute__((packed));

    SendPacket sendPacket;
    
} ;
