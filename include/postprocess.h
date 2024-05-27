#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>

#include <opencv2/highgui.hpp>

typedef signed char int8_t;
typedef unsigned int uint32_t;

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
} DetectRect;

// yolov10
class GetResultRectYolov10
{
public:
    GetResultRectYolov10();

    ~GetResultRectYolov10();

    int GenerateMeshGrid();

    int GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects);

    float sigmoid(float x);

private:
    std::vector<float> MeshGrid;

    const int ClassNum = 80;
    int HeadNum = 3;

    int InputWidth = 640;
    int InputHeight = 640;
    int Strides[3] = {8, 16, 32};
    int MapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};

    float NmsThresh = 0.45;
    float ObjectThresh = 0.25;

    int TopK = 50;

    std::vector<float> RegDfl;
    float RegDeq[16] = {0};
};

#endif
