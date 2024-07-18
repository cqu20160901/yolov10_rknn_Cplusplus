#include "postprocess.h"
#include <math.h>

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

static inline float fast_exp(float x)
{
    // return exp(x);
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (12102203.1616540672 * x + 1064807160.56887296);
    return v.f;
}

static float DeQnt2F32(int8_t qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

/****** yolov10 ****/
GetResultRectYolov10::GetResultRectYolov10()
{
}

GetResultRectYolov10::~GetResultRectYolov10()
{
}

float GetResultRectYolov10::sigmoid(float x)
{
    return 1 / (1 + fast_exp(-x));
}

int GetResultRectYolov10::GenerateMeshGrid()
{
    int ret = 0;
    if (HeadNum == 0)
    {
        printf("=== Yolov10 MeshGrid  Generate failed! \n");
    }

    for (int index = 0; index < HeadNum; index++)
    {
        for (int i = 0; i < MapSize[index][0]; i++)
        {
            for (int j = 0; j < MapSize[index][1]; j++)
            {
                MeshGrid.push_back(float(j + 0.5));
                MeshGrid.push_back(float(i + 0.5));
            }
        }
    }

    printf("=== Yolov10 MeshGrid  Generate success! \n");

    return ret;
}

int GetResultRectYolov10::GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects)
{
    int ret = 0;
    if (MeshGrid.empty())
    {
        ret = GenerateMeshGrid();
    }

    int gridIndex = -2;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
    float cls_val = 0;
    float cls_max = 0;
    int cls_index = 0;

    int quant_zp_cls = 0, quant_zp_reg = 0, quant_zp_msk;
    float quant_scale_cls = 0, quant_scale_reg = 0, quant_scale_msk = 0;

    DetectRect temp;
    std::vector<DetectRect> RectResults;

    for (int index = 0; index < HeadNum; index++)
    {
        int8_t *reg = (int8_t *)pBlob[index * 2 + 0];
        int8_t *cls = (int8_t *)pBlob[index * 2 + 1];
        int8_t *msk = (int8_t *)pBlob[index + HeadNum * 2];

        quant_zp_reg = qnt_zp[index * 2 + 0];
        quant_zp_cls = qnt_zp[index * 2 + 1];
        quant_zp_msk = qnt_zp[index + HeadNum * 2];

        quant_scale_reg = qnt_scale[index * 2 + 0];
        quant_scale_cls = qnt_scale[index * 2 + 1];
        quant_scale_msk = qnt_scale[index + HeadNum * 2];

        float sfsum = 0;
        float locval = 0;
        float locvaltemp = 0;

        for (int h = 0; h < MapSize[index][0]; h++)
        {
            for (int w = 0; w < MapSize[index][1]; w++)
            {
                gridIndex += 2;

                if (1 == ClassNum)
                {
                    cls_max = sigmoid(DeQnt2F32(cls[0 * MapSize[index][0] * MapSize[index][1] + h * MapSize[index][1] + w], quant_zp_cls, quant_scale_cls));
                    cls_index = 0;
                }
                else
                {
                    for (int cl = 0; cl < ClassNum; cl++)
                    {
                        cls_val = cls[cl * MapSize[index][0] * MapSize[index][1] + h * MapSize[index][1] + w];

                        if (0 == cl)
                        {
                            cls_max = cls_val;
                            cls_index = cl;
                        }
                        else
                        {
                            if (cls_val > cls_max)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                        }
                    }
                    cls_max = sigmoid(DeQnt2F32(cls_max, quant_zp_cls, quant_scale_cls));
                }

                if (cls_max > ObjectThresh)
                {
                    RegDfl.clear();
                    for (int lc = 0; lc < 4; lc++)
                    {
                        sfsum = 0;
                        locval = 0;
                        for (int df = 0; df < 16; df++)
                        {
                            locvaltemp = exp(DeQnt2F32(reg[((lc * 16) + df) * MapSize[index][0] * MapSize[index][1] + h * MapSize[index][1] + w], quant_zp_reg, quant_scale_reg));
                            RegDeq[df] = locvaltemp;
                            sfsum += locvaltemp;
                        }
                        for (int df = 0; df < 16; df++)
                        {
                            locvaltemp = RegDeq[df] / sfsum;
                            locval += locvaltemp * df;
                        }

                        RegDfl.push_back(locval);
                    }

                    xmin = (MeshGrid[gridIndex + 0] - RegDfl[0]) * Strides[index];
                    ymin = (MeshGrid[gridIndex + 1] - RegDfl[1]) * Strides[index];
                    xmax = (MeshGrid[gridIndex + 0] + RegDfl[2]) * Strides[index];
                    ymax = (MeshGrid[gridIndex + 1] + RegDfl[3]) * Strides[index];

                    xmin = xmin > 0 ? xmin : 0;
                    ymin = ymin > 0 ? ymin : 0;
                    xmax = xmax < InputWidth ? xmax : InputWidth;
                    ymax = ymax < InputHeight ? ymax : InputHeight;

                    if (xmin >= 0 && ymin >= 0 && xmax <= InputWidth && ymax <= InputHeight)
                    {
                        temp.xmin = xmin / InputWidth;
                        temp.ymin = ymin / InputHeight;
                        temp.xmax = xmax / InputWidth;
                        temp.ymax = ymax / InputHeight;
                        temp.classId = cls_index;
                        temp.score = cls_max;
                        RectResults.push_back(temp);
                    }
                }
            }
        }
    }

    if (RectResults.size() > TopK)
    {
        std::sort(RectResults.begin(), RectResults.end(), [](DetectRect &Rect1, DetectRect &Rect2) -> bool
                  { return (Rect1.score > Rect2.score); });
    }

    std::cout << "TopK Before num :" << RectResults.size() << std::endl;

    for (int i = 0; i < RectResults.size() && i < TopK; ++i)
    {
        // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
        DetectiontRects.push_back(float(RectResults[i].classId));
        DetectiontRects.push_back(float(RectResults[i].score));
        DetectiontRects.push_back(float(RectResults[i].xmin));
        DetectiontRects.push_back(float(RectResults[i].ymin));
        DetectiontRects.push_back(float(RectResults[i].xmax));
        DetectiontRects.push_back(float(RectResults[i].ymax));
    }

    return ret;
}
