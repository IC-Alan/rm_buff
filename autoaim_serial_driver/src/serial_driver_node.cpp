#include <rclcpp/rclcpp.hpp>
#include <serial_driver/serial_driver.hpp>

#include <autoaim_interfaces/msg/comm_recv.hpp>
#include <autoaim_interfaces/msg/comm_send.hpp>

#include <serial_port.hpp>
#include <receive_packet.hpp>
#include <send_packet.hpp>

namespace autoaim_serial_driver {

template<
    typename SendPacketT,
    typename RecvPacketT,
    typename SendMsgT,
    typename RecvMsgT
>
class SerialDriverNode : public rclcpp::Node {
public:
    explicit SerialDriverNode(const rclcpp::NodeOptions& options) 
        : Node("autoaim_serial_driver", options) {

        // Declare parameters
        enable_debug_ = declare_parameter("enable_debug", false);
        enable_send_ = declare_parameter("enable_send_to_electl", true);
        auto device_name = declare_parameter("device_name", "/dev/ttyUSB0");
        auto baud_rate = declare_parameter("baud_rate", 115200);
        
        // Setup publishers and subscribers
        comm_recv_pub_ = create_publisher<RecvMsgT>(
            declare_parameter("comm_recv_topic", "/serial/comm_recv"), 
            rclcpp::SensorDataQoS().keep_last(1)
        );

        comm_send_sub_ = create_subscription<SendMsgT>(
            declare_parameter("comm_send_topic", "/serial/comm_send"),
            rclcpp::SensorDataQoS().keep_last(1),
            [this](const typename SendMsgT::SharedPtr msg) {
                if (!enable_send_) return;
                if (serial_port_ && serial_port_->is_open()) {
                    SendPacketT packet;
                    packet.from_msg(msg);
                    serial_port_->send_packet(packet.get_vector());
                }
            }
        );

        if (enable_debug_) {
            RCLCPP_INFO(get_logger(), "Running in debug mode");
            return;
        }
        
        // Initialize serial port
        serial_port_ = std::make_unique<SerialPort>(
            this,
            device_name,
            baud_rate,
            drivers::serial_driver::FlowControl::NONE,
            drivers::serial_driver::Parity::NONE,
            drivers::serial_driver::StopBits::ONE
        );

        // Start receive thread
        RecvPacketT receiver;
        serial_port_->start_receive_thread(
            receiver.get_header(),
            receiver.get_recv_size(),
            [this](const std::vector<uint8_t>& data) {
                handle_received_packet(data);
            }
        );
    }

    ~SerialDriverNode() = default;

private:
    void handle_received_packet(const std::vector<uint8_t>& data) {
        RecvPacketT packet;
        packet.from_vector(data);
        auto msg = packet.get_msg();
        msg.header.stamp = now();
        comm_recv_pub_->publish(msg);
    }

    std::unique_ptr<SerialPort> serial_port_;
    typename rclcpp::Publisher<RecvMsgT>::SharedPtr comm_recv_pub_;
    typename rclcpp::Subscription<SendMsgT>::SharedPtr comm_send_sub_;

    bool enable_debug_;
    bool enable_send_;
};

using InfantrySerialDriverNode = SerialDriverNode<
    InfantrySendPacket,
    InfantryReceivePacket,
    autoaim_interfaces::msg::CommSend,
    autoaim_interfaces::msg::CommRecv
>;

} // namespace autoaim_serial_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_serial_driver::InfantrySerialDriverNode)