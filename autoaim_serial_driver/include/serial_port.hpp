#pragma once

#include <rclcpp/rclcpp.hpp>
#include <serial_driver/serial_driver.hpp>
#include <crc_checksum.hpp>
// #include "rclcpp_lifecycle/lifecycle_node.hpp"

constexpr int MAX_RECONNECT_ATTEMPTS = 5;
constexpr std::chrono::milliseconds RECONNECT_INTERVAL(1000);

class SerialPort {
public:
    SerialPort(rclcpp::Node* node, 
               const std::string& port_name,
               const int baud_rate,
               const drivers::serial_driver::FlowControl flow_control,
               const drivers::serial_driver::Parity parity,
               const drivers::serial_driver::StopBits stop_bits
            )
        : node_(node), 
          reconnect_attempts_(0) {
        try {
            auto config = std::make_unique<drivers::serial_driver::SerialPortConfig>(
                baud_rate, flow_control, parity, stop_bits);
            owned_ctx_ = std::make_unique<IoContext>(2);
            driver_ = std::make_unique<drivers::serial_driver::SerialDriver>(
                *owned_ctx_);
          
            driver_->init_port(port_name, *config);
            reopen_port();
        } catch (const std::exception& e) {
            RCLCPP_FATAL(node_->get_logger(), "Driver initialization failed: %s", e.what());
        }
    }

    ~SerialPort() {
        shutdown_flag_ = true;
        close_port();
        if (receive_thread_.joinable()) {
            receive_thread_.join();
        }
        if (driver_) {
            driver_.reset();
        }
        if (owned_ctx_) {
            owned_ctx_.reset(); 
        }
    }

    void send_packet(const std::vector<uint8_t>& data) {
        if (!is_open()) {
            RCLCPP_WARN(node_->get_logger(), "Port not open, cannot send data");
            return;
        }
        try {
            driver_->port()->send(data);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(node_->get_logger(), "Send failed: %s", e.what());
            handle_io_error();
        }
    }

    void start_receive_thread(const std::vector<uint8_t>& val_header, 
                              const size_t frame_size, 
                              std::function<void(const std::vector<uint8_t>&)> recv_callback) {
        receive_thread_ = std::thread([this, val_header, frame_size, recv_callback]() {
            receive_loop(val_header, frame_size, recv_callback);
        });
    }

    bool is_open() const { 
        return driver_->port()->is_open();
    }

private:
    std::atomic<bool> shutdown_flag_{false};
    rclcpp::Node* node_;
    std::unique_ptr<drivers::serial_driver::SerialDriver> driver_;
    std::thread receive_thread_;
    std::unique_ptr<IoContext> owned_ctx_;
    std::atomic<int> reconnect_attempts_{0};

    void receive_loop(const std::vector<uint8_t>& val_header, 
                      const size_t frame_size, 
                      std::function<void(const std::vector<uint8_t>&)> recv_callback) {
        std::vector<uint8_t> recv_header(val_header.size());
        std::vector<uint8_t> buffer(frame_size - val_header.size());

        while (!shutdown_flag_ && rclcpp::ok()) {
            try {
                if (!is_open()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                // Read header
                driver_->port()->receive(recv_header);
                if (recv_header != val_header) {
                    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 20000, 
                        "Invalid header: 0x%02X 0x%02X", recv_header[0], recv_header[1]);
                    continue;
                }

                // Read remaining data
                size_t bytes_read = driver_->port()->receive(buffer);
              
                if (bytes_read != buffer.size()) {
                    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 20000, 
                        "Incomplete packet, expected %zu, got %zu", buffer.size(), bytes_read);
                    continue;
                }

                // Validate CRC
                if (!Verify_CRC8_Check_Sum(buffer.data(), buffer.size())) {
                    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 20000, 
                        "CRC check failed");
                    continue;
                }

                recv_callback(std::vector<uint8_t>(
                    buffer.begin(),
                    buffer.begin() + frame_size - recv_header.size() - 1
                ));

            } catch (const std::exception& e) {
                RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 20000,
                                    "Receive error: %s", e.what());
                handle_io_error();
            }
        }
    }
    
    void handle_io_error() {
        if (reconnect_attempts_ < MAX_RECONNECT_ATTEMPTS) {
            RCLCPP_INFO(node_->get_logger(), "Attempting to reconnect... (%d/%d)",
                       reconnect_attempts_ + 1, MAX_RECONNECT_ATTEMPTS);
            reopen_port();
            reconnect_attempts_++;
        } else {
            RCLCPP_ERROR(node_->get_logger(), "Max reconnect attempts reached");
            reconnect_attempts_ = 0;
        }
    }

    void reopen_port() {
        try {
            if (is_open()) driver_->port()->close();
            driver_->port()->open();
            RCLCPP_INFO(node_->get_logger(), "Port reopened successfully");
            reconnect_attempts_ = 0;
        } catch (const std::exception& e) {
            RCLCPP_WARN(node_->get_logger(), "Port reopen failed: %s", e.what());
        }
    }

    void close_port() {
        if (is_open()) {
            try {
                driver_->port()->close();
            } catch (const std::exception& e) {
                if (!shutdown_flag_) {  // Only log if not shutting down
                    RCLCPP_ERROR(node_->get_logger(), "Port close failed: %s", e.what());
                }
            }
        }
    }
};