#pragma once

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

using namespace nvinfer1;

struct keypoint
{
    cv::Point2f pt;
    float kpt_conf = 0; // 相信奇迹的力量
};

struct Object
{
    float bbox[4];
    float conf;
    int color;
    int class_id;
    std::vector<keypoint> kpts;
};

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}
void nms(std::vector<int> &picked, std::vector<Object> &proposals, float nms_thresh)
{
    picked.clear();

    const int n = proposals.size();

    for (int i = 0; i < n; i++)
    {
        Object &a = proposals[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            Object &b = proposals[picked[j]];
            if (a.class_id == b.class_id && a.color == b.color &&  iou(a.bbox, b.bbox) > nms_thresh)
            {
                keep = 0;
            }
        }
        if (keep)
            picked.push_back(i);
    }
}
void qsort_descent_inplace(std::vector<Object> &objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].conf;

    while (i <= j)
    {
        while (objects[i].conf > p)
            i++;

        while (objects[j].conf < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(objects, left, j);
    if (i < right)
        qsort_descent_inplace(objects, i, right);
}

void qsort_descent_inplace(std::vector<Object> &objects)
{
    if (objects.empty())
        return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}
