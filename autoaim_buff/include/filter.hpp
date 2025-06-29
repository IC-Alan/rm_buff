/*
 * @Description:
 * @Version: 1.0
 * @Autor: Julian Lin
 * @Date: 2023-06-22 22:47:41
 * @LastEditors: Julian Lin
 * @LastEditTime: 2023-07-26 23:26:34
 */
#ifndef __FILTER_HPP__
#define __FILTER_HPP__
#include <iostream>
#include <cmath>
#include <vector>
/*
    for (double x : input) {
        double filteredValue = butterworth.filter(x);
        output.push_back(filteredValue);
    }
*/
class IIR
{
public:
    IIR() { clear(); };
    IIR(int order, double cutoffFrequency, double samplingRate)
        : order_(order), cutoffFrequency_(cutoffFrequency), samplingRate_(samplingRate)
    {
        clear();
    }
    IIR(vector<double> den, vector<double> num_)
    {
        order_ = den.size() - 1;
        cutoffFrequency_ = 0;
        samplingRate_ = 0;
        bCoefficients_ = num_;
        aCoefficients_ = den;
        clear();
    }
    ~IIR() = default;
    void SetHypParam(int order, double cutoffFrequency, double samplingRate)
    {
        order_ = order;
        cutoffFrequency_ = cutoffFrequency;
        samplingRate_ = samplingRate;
        PrintHypParam();
        clear();
    }
    void SetHypParam(vector<double> den_, vector<double> num_)
    {
        order_ = den_.size() - 1;
        cutoffFrequency_ = 0;
        samplingRate_ = 0;
        bCoefficients_ = num_;
        aCoefficients_ = den_;
        PrintHypParam();
        clear();
    }
    void PrintHypParam()
    {
        std::cout << "b: ";
        for (int i = 0; i < order_ + 1; i++)
        {
            std::cout << bCoefficients_[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "a: ";
        for (int i = 0; i < order_ + 1; i++)
        {
            std::cout << aCoefficients_[i] << " ";
        }
        std::cout << std::endl;
    }
    void clear()
    {
        inputBuffer_.clear();
        outputBuffer_.clear();
        inputBuffer_.resize(order_ + 1, 0.0);
        outputBuffer_.resize(order_ + 1, 0.0);
    }
    float filter(double input)
    {
        // 更新历史输入缓冲区
        inputBuffer_.push_back(input);
        inputBuffer_.erase(inputBuffer_.begin());

        // outputBuffer_.push_back(0.0);
        // outputBuffer_.erase(outputBuffer_.begin());

        // 应用滤波器
        double output = 0.0;
        for (int i = 0; i <= order_; ++i)
        {
            output += bCoefficients_[i] * inputBuffer_[order_ - i];
        }
        for (int i = 1; i <= order_; ++i)
        {
            output -= aCoefficients_[i] * outputBuffer_[i - 1];
        }
        output /= aCoefficients_[0];
        // 更新历史输出缓冲区
        outputBuffer_.push_back(output);
        outputBuffer_.erase(outputBuffer_.begin());
        return output;
    }

private:
    int order_;
    double cutoffFrequency_;
    double samplingRate_;
    std::vector<double> bCoefficients_;
    std::vector<double> aCoefficients_;
    std::vector<double> inputBuffer_;
    std::vector<double> outputBuffer_;
};

/* 有问题，请勿使用 */
enum class WaveletType
{
    HAAR = 0,
    DB4 = 1,
    DB6 = 2,
    SYM4 = 3,
    SYM6 = 4,
};
struct Wavelet
{
    WaveletType type;
    std::vector<double> lowPassFilter;
    std::vector<double> highPassFilter;
};
class WaveletFilter
{
public:
    WaveletFilter() { clear(); }
    WaveletFilter(int levels, double threshold, int windowSize)
        : levels_(levels), threshold_(threshold), windowSize_(windowSize) {}
    void SetHypParam(int levels, double threshold, int windowSize, WaveletType type)
    {
        levels_ = levels;
        threshold_ = threshold;
        windowSize_ = windowSize;
        switch (type)
        {
        case WaveletType::HAAR:
            wavelet_.type = WaveletType::HAAR;
            wavelet_.lowPassFilter = {0.7071067811865476, 0.7071067811865476};
            wavelet_.highPassFilter = {-0.7071067811865476, 0.7071067811865476};
            break;
        case WaveletType::DB4:
            wavelet_.type = WaveletType::DB4;
            wavelet_.lowPassFilter = {0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604};
            wavelet_.highPassFilter = {-0.1294095225512604, -0.2241438680420134, 0.8365163037378079, -0.4829629131445341};
            break;
        case WaveletType::DB6:
            wavelet_.type = WaveletType::DB6;
            wavelet_.lowPassFilter = {0.3326705529500825, 0.8068915093110925, 0.4598775021184915, -0.1350110200102546, -0.08544127388224149, 0.03522629188570953};
            wavelet_.highPassFilter = {0.03522629188570953, 0.08544127388224149, -0.1350110200102546, -0.4598775021184915, 0.8068915093110925, -0.3326705529500825};
            break;
        case WaveletType::SYM4:
            wavelet_.type = WaveletType::SYM4;
            wavelet_.lowPassFilter = {-0.2303778133088552, 0.7148465705529154, -0.6308807679295904, -0.02798376941698385};
            wavelet_.highPassFilter = {-0.02798376941698385, 0.6308807679295904, 0.7148465705529154, 0.2303778133088552};
            break;
        case WaveletType::SYM6:
            wavelet_.type = WaveletType::SYM6;
            wavelet_.lowPassFilter = {0.01540410932702737, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194};
            wavelet_.highPassFilter = {0.787641141030194, -0.4910559419267466, -0.048311742585633, 0.11799011114819057, 0.0034907120842174702, -0.01540410932702737};
            break;
        }
    }
    float filter(double input)
    {
        // 将新的数据加入信号缓冲区
        signal_.push_back(input);

        // 确保信号缓冲区的长度不超过窗口大小
        if (signal_.size() > windowSize_)
        {
            signal_.erase(signal_.begin());
        }

        // 检查信号缓冲区是否满足窗口大小，满足则进行实时滤波
        if (signal_.size() == windowSize_)
        {
            std::vector<double> windowData(signal_.begin(), signal_.end());

            // 对当前窗口的信号进行小波阈值去噪
            for (int level = 0; level < levels_; ++level)
            {
                std::vector<double> cA, cD;
                dwt(windowData, cA, cD);
                waveletThreshold(cD, threshold_);
                idwt(windowData, cA, cD);
            }

            // 将处理后的信号数据替换回信号缓冲区
            // std::copy(windowData.begin(), windowData.end(), signal_.begin());
            return windowData.back();
        }
        return 0.0;
    }

    void clear()
    {
        signal_.clear();
    }

    const std::vector<double> &getFilteredSignal() const
    {
        return signal_;
    }

private:
    // 离散小波变换
    void dwt(std::vector<double> &signal, std::vector<double> &cA, std::vector<double> &cD)
    {
        int N = signal.size();
        int half_N = N / 2;
        cA.resize(half_N);
        cD.resize(half_N);

        for (int i = 0; i < half_N; ++i)
        {
            cA[i] = 0.0;
            cD[i] = 0.0;
            for (int j = 0; j < 4; ++j)
            {
                int k = (i + j) % N;
                cA[i] += wavelet_.lowPassFilter[j] * signal[k];
                cD[i] += wavelet_.highPassFilter[j] * signal[k];
            }
        }
    }

    // 小波阈值去噪
    void waveletThreshold(std::vector<double> &cD, double threshold)
    {
        for (double &coeff : cD)
        {
            if (std::abs(coeff) <= threshold)
            {
                coeff = 0.0;
            }
        }
    }

    // 逆离散小波变换
    void idwt(std::vector<double> &signal, const std::vector<double> &cA, const std::vector<double> &cD)
    {
        int N = signal.size();
        int half_N = N / 2;

        // 小波重构
        for (int i = 0; i < N; ++i)
        {
            signal[i] = 0.0;
            for (int j = 0; j < 4; ++j)
            {
                int k = (i - j + N) % N;
                signal[i] += wavelet_.lowPassFilter[j] * cA[k] + wavelet_.highPassFilter[j] * cD[k];
            }
        }
    }
    int levels_;
    double threshold_;
    int windowSize_;
    std::vector<double> signal_;
    Wavelet wavelet_;
};
#endif