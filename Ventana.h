#ifndef VENTANA_H
#define VENTANA_H
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>


using namespace cv;
using namespace std;

class Ventana
{
public:
    int posX;
    int posY;
    int tamanoHorizontal;
    int tamanoVertical;
    string label;
    Mat image;
    int R;
    int G;
    int B;


    Ventana();
    Ventana(int pX, int pY, int tHorizontal, int tVertical, int red, int green, int blue, string label);
    void imprimirInformacionVentana();
    void moverRectangulo(int x, int y);
    void dibujarVentana(Mat);
    void extraerRecorte(Mat imageBuffer);

};

#endif // VENTANA_H
