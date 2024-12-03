#include "Ventana.h"
Ventana::Ventana()
{

}

Ventana::Ventana(int pX, int pY, int tHorizontal, int tVertical, int red, int green, int blue, string lbel){
    posX = pX;
    posY = pY;
    tamanoHorizontal = tHorizontal;
    tamanoVertical = tVertical;
    label = lbel;
    R = red;
    G = green;
    B = blue;
}

//Constructor Ventana
//Parameters:
//anchoImagen: video stream width
//altoImagen: video stream height
//red: integer value of the red component
//green: integer value of the green component
//blue: integer value for the blue component
//generates window initial position using random number generators

//Ventana::Ventana(int anchoImagen, int altoImagen,int tHorizontal, int tVertical, int red, int green, int blue){

//    tamanoHorizontal = rand() % tHorizontal;
//    tamanoVertical = rand() % tVertical;
//    posX = rand() % (anchoImagen - tHorizontal);
//    posY = rand() % (altoImagen - tVertical);
//    R = red;
//    G = green;
//    B = blue;
//}

void Ventana::imprimirInformacionVentana(){
    std::cout<< "posX" << "=" << posX<<std::endl;
    std::cout<< "posY" << "=" << posY<<std::endl;
    std::cout<< "tamanoHoriontal" << "=" << tamanoHorizontal<<std::endl;
    std::cout<< "tamanoVertical" << "=" << tamanoVertical<<std::endl;
}

void Ventana::extraerRecorte(Mat imageBuffer){
    //cout << "Antes del cropRectangle posx = " << posX << " posY = "<< posY << " posX+tamanoHorizontal = " << posX+tamanoHorizontal <<  " posY+tamanoVertical = " << posY+tamanoVertical <<  endl;
    image = cv::Mat::zeros(Size(tamanoHorizontal,tamanoVertical), imageBuffer.type());
    imageBuffer(cv::Rect(posX,posY, tamanoHorizontal,tamanoVertical)).copyTo(image(cv::Rect(0,0,tamanoHorizontal,tamanoVertical)));
    //cout << "despues del cropRectangle posx = " << posX << " posY = "<< posY << " posX+tamanoHorizontal = " << posX+tamanoHorizontal <<  " posY+tamanoVertical = " << posY+tamanoVertical <<  endl;

}

void Ventana::dibujarVentana(Mat imageBuffer){
//	Rect cropRectangle(posX, posY, posX+tamanoHorizontal,posY+tamanoVertical);
//	cout << "Antes del cropRectangle posx = " << posX << " posY = "<< posY << " posX+tamanoHorizontal = " << posX+tamanoHorizontal <<  " posY+tamanoVertical = " << posY+tamanoVertical <<  endl;
//
//	image = cv::Mat::zeros(Size(tamanoHorizontal,tamanoVertical), imageBuffer.type());
//	imageBuffer(cv::Rect(posX,posY, tamanoHorizontal,tamanoVertical)).copyTo(image(cv::Rect(0,0,tamanoHorizontal,tamanoVertical)));
//	cout << "despues del cropRectangle posx = " << posX << " posY = "<< posY << " posX+tamanoHorizontal = " << posX+tamanoHorizontal <<  " posY+tamanoVertical = " << posY+tamanoVertical <<  endl;

    rectangle(imageBuffer, Point(posX,posY),Point(posX+tamanoHorizontal,posY+tamanoVertical), Scalar(B,G,R),5,0);




    //putText(imageBuffer, label,  Point(posX, posY+20),FONT_HERSHEY_PLAIN, 1.2,Scalar(B,G,R),1.0);
    // putText(imageBuffer, label,  Point(posX, posY+20),CV_FONT_HERSHEY_DUPLEX, 0.5,Scalar(B,G,R),1.0);
    putText(imageBuffer, label,  Point(posX, posY+20),cv::FONT_HERSHEY_DUPLEX, 0.5,Scalar(B,G,R),1.0);

}

void Ventana::moverRectangulo(int x, int y){
    posX = x;
    posY = y;
}
