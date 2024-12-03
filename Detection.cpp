#include "Detection.h"

Detection::Detection()
{
	badDetection = true;
    active = false;
    pi = 1;

}

Detection::Detection(int xsub1, int xsub2, int xsub3, float p){
	badDetection = false;
    x1 = xsub1;
    x2 = xsub2;
    x3 = xsub3;
    pi = p;
    label = "No label";
    ventana = Ventana(x1,x2,x3,x3,255,255,0,label);
    //int xSupIzq = (int)(x1-x3/2);
    //int ySupIzq = (int)(x2-x3/2);
//    int xInfDch = (int)(x1+x3/2);
//    int yInfDch = (int)(x2+x3/2);


////    Rect auxRectangle(xSupIzq,ySupIzq,x3,x3);
//    Rect auxRectangle(x1,x2,x3,x3);
//
////    //Rect roi(40,100,100,120);
//
//    image = imagen(auxRectangle);

}



void Detection::printDetection(){
    cout<< "pi=" << pi << endl;
    cout << "x1="<< x1 << endl;
    cout << "x2="<< x2 << endl;
    cout << "x3=" << x3 << endl;
    cout << "active = " << active << endl;
}

void Detection::updateDetection(float alpha, float delta){
    if (active){
        //cout << "Dentro de update detection" << endl;

        pi = alpha*pi;
        //x1 = x1+delta;
        ventana.posX = x1;
    }

    //not necessary to modify x1 or x3 as they are the same as in frame n-1

}

