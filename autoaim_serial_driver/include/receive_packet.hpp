#pragma once

#include <rclcpp/rclcpp.hpp>
#include <autoaim_interfaces/msg/comm_recv.hpp>
#include <serial_utils.hpp>

class InfantryReceivePacket {
public:
    InfantryReceivePacket() = default;
    ~InfantryReceivePacket() = default;

    size_t get_recv_size() {
        return sizeof(RecvPacket) + header.size() + 1;
    }

    std::vector<uint8_t> get_header() {
        return header;
    }
    
    void from_vector(const std::vector<uint8_t>& data) {
        recvPacket = serial_utils::from_vector<RecvPacket>(data);
    }

    void from_debug() {
        recvPacket.mode = 0;
        recvPacket.shootSpeed = 24.5;
        recvPacket.rollAngle = 0;
        recvPacket.pitchAngle = 0;
        recvPacket.yawAngle = 0;
        recvPacket.mixedData = 0;
    }

    autoaim_interfaces::msg::CommRecv get_msg() const {
        autoaim_interfaces::msg::CommRecv msg;
        msg.mode = recvPacket.mode;
        // msg.mode = 4;
        msg.shoot_speed = static_cast<float>(serial_utils::swap_bytes_of_int16(recvPacket.shootSpeed)) / 100.0f;
        msg.roll = static_cast<float>(serial_utils::swap_bytes_of_int16(recvPacket.rollAngle)) / 100.0f;
        msg.pitch = static_cast<float>(serial_utils::swap_bytes_of_int16(recvPacket.pitchAngle)) / 100.0f;
        msg.yaw = static_cast<float>(serial_utils::swap_bytes_of_int16(recvPacket.yawAngle)) / 100.0f;
        msg.target_color = recvPacket.mixedData & 0x03;
        return msg;
    }

private:
    std::vector<uint8_t> header = {0x3F, 0x4F};

    struct RecvPacket
    {
        uint8_t mode;
        int16_t shootSpeed;
        int16_t rollAngle, pitchAngle, yawAngle;
        uint8_t mixedData; 
    }__attribute__((packed));

    RecvPacket recvPacket;
};