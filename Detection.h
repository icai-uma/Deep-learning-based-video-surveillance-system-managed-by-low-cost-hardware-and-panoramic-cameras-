#ifndef DETECTION_H
#define DETECTION_H
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include "Ventana.h"
#include <string>

using namespace std;

class Detection
{
public:
	 bool badDetection;
    float pi;
    int x1;
    int x2;
    int x3;
    //Mat image;
    bool active;
    string label;
    Ventana ventana;
    Detection();
    Detection(int xsub1, int xsub2, int xsub3, float p);

    void printDetection();
    void updateDetection(float alpha, float delta);
};

#endif // DETECTION_H
