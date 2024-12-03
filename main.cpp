#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include "Ventana.h"
#include <stdlib.h>
#include <array>
//#include "boost/filesystem.hpp"
//#include "boost/regex.hpp"

#include "Detection.h"

#include <stdio.h>
#include <random>
#include <fstream>
#include <sstream>
#include <chrono>
#include <opencv2/opencv.hpp>

// Include the model interface file for the compiled ELL model
#include "model.h"

// Include helper functions
#include "tutorialHelpers.h"
#include "gnuplot.h"


//#include <set>
#define NUMERODEFICHEROS 300
#define MAXDETECTIONS 10000
//#define NUMROWSINICIAL 960
//#define NUMCOLSINICIAL 1920
#define NUMROWS 800
#define NUMCOLS 1920
#define DETECTIONMAXSIZE 400
#define MINIMUMFRAME 10000
#define MAXIMUMFRAME 60000
//#define MAXIMUMFRAME 60000
//#define DETECTIONTHRESHOLD 10.0
#define EUCLIDEANDISTANCETHRESHOLD 20
//#define GAUSSIAN true  //Tocar aqui
#define NUMRANDOMWINDOWS 1
//#define NAMECATEGORIESFILE "/home/icai23/IWINAC2019/categories.txt"
#define NAMECATEGORIESFILE "../categories.txt"

//#define NAMEANOMALOUSCATEGORIESFILE "/home/icai23/integracion/anomalousCategories3.txt"
//#define NAMEANOMALOUSCATEGORIESFILE "/home/icai23/IWINAC2019/anomalousCategories4.txt"
#define NAMEANOMALOUSCATEGORIESFILE "../anomalousCategories4.txt"

//#define FILEBASEPATH "/home/icai23/barridoTestPositivoBottle/imagen"
//#define FILEBASEPATH "/home/icai23/barridosIndividuales/barrido36011ObjetosAnim/imagen"
//#define FILEBASEPATH "/home/icai23/IWINAC2019/framesAnimado/frame-"
//#define FILEBASEPATH "../framesSoloChainsaw/frame-"
//#define FILEBASEPATH "../framesSoloBalonChainsaw/frame-"
//#define FILEBASEPATH "../framesSoloBalon/frame-"
#define FILEBASEPATH "../frames5Objetos/frame-"


#define NUMTESTS 10 //Tocar aqui
#define NUMANOMALOUSOBJECTS 11 //Tocar aqui
//#define RECORDFILEBASENAME "/home/icai23/recordsTotalAnim/"
//#define RECORDFILEBASENAME "/home/icai23/IWINAC2019/records/"
#define RECORDFILEBASENAME "../records/"
# define NUMCOLSRECORDFILE 8
#define MAXWINDOWSFORTEST 10 //Tocar aqui
#define DISPLAYRESULTSWINDOW false
#define RUTAVIDEOSALIDA "../outputVideo/outputVideo_"
#define PROBABILITYDISTRIBUTION "GAUSSIAN"
//#define PROBABILITYDISTRIBUTION "STUDENT"
//#define PROBABILITYDISTRIBUTION "TRIANGLE"

struct objectFound{
	string name;
	int numberOfDetections;
};


using namespace cv;
using namespace std;



class Tutorial 
{
public:

    Tutorial() 
    {
        // Read the category names
        this->categories = ReadLinesFromFile(NAMECATEGORIESFILE);
		////std::cout << "nmero de categorias = " << categories.size() << std::endl;

        // Get the model's input shape. We will use this information later to resize images .
        TensorShape inputShape;
        model_GetInputShape(0, &inputShape);
        this->inputShape = inputShape;
        this->inputSize = model_GetInputSize();
        printf("Input shape=[%d,%d,%d]\n", inputShape.rows, inputShape.columns,
            inputShape.channels);

        TensorShape outputShape;
        model_GetOutputShape(0, &outputShape);
        this->outputShape = outputShape;
        printf("Output shape=[%d,%d,%d]\n", outputShape.rows, outputShape.columns,
            outputShape.channels);

        // Create a vector to hold the model's output predictions

        this->outputSize = model_GetOutputSize();
        this->input.resize(this->inputSize);


        this->start = std::chrono::steady_clock::now();
    }

    /*void PrepareNextImage(cv::VideoCapture& camera)
    {
        // Get an image from the camera. (Alternatively, call GetImageFromFile to read from file)
        cv::Mat image = GetImageFromCamera(camera);
        this->image = image;

        // Prepare an image for processing
        // - Resize and center-crop to the required width and height while preserving aspect ratio.
        // - OpenCV gives the image in BGR order. If needed, re-order the channels to RGB.
        // - Convert the OpenCV result to a std::vector<float>
        this->input = tutorialHelpers::PrepareImageForModel(image, this->inputShape.columns,
            this->inputShape.rows);
    }*/
    
     void PrepareNextImage(cv::Mat & imageToTest)
    {
        // Get an image from the camera. (Alternatively, call GetImageFromFile to read from file)

        this->image = imageToTest;

        // Prepare an image for processing
        // - Resize and center-crop to the required width and height while preserving aspect ratio.
        // - OpenCV gives the image in BGR order. If needed, re-order the channels to RGB.
        // - Convert the OpenCV result to a std::vector<float>
        this->input = tutorialHelpers::PrepareImageForModel(image, this->inputShape.columns,
            this->inputShape.rows);
    }

    int8_t InputCallback(float* buffer)
    {
        size_t size = this->input.size();
        assert(size == this->inputSize);
        ::memcpy(buffer, &this->input[0], this->inputSize * sizeof(float));

        return true;
    }

    void OutputCallback(float* buffer)
    {
        this->predictions = std::vector<float>(&buffer[0], &buffer[this->outputSize]);
    }

    bool ValidateCategories()
    {
        return (categories.size() == this->outputSize);
    }

    int GetOutputSize() {
        return this->outputSize;
    }

    void Step() 
    {
        auto now = std::chrono::steady_clock::now();
        double ticksFromStart = std::chrono::duration<double>(now - start).count();
        double ticks = model_GetTicksUntilNextInterval(ticksFromStart);
        model_Predict(ticks, this->input.data());
    }

    bool HasPrediction() 
    {
        return this->predictions.size() > 0;
    }

    std::vector<std::pair<size_t, float>> GetTopN(size_t topN = 5, double threshold = 0.20)
    {
        return tutorialHelpers::GetTopN(this->predictions, 5);
    }

    std::string GetCategory(int index) 
    {
        return categories[index];
    }
    
    cv::Mat GetImage() 
    {
        return this->image;
    }


private:
    // Read an image from the camera
    static cv::Mat GetImageFromCamera(cv::VideoCapture& camera)
    {
        cv::Mat frame;
        camera >> frame;
        return frame;
    }

    // Read an image from a file
    static cv::Mat GetImageFromFile(const std::string& filename)
    {
        return cv::imread(filename);
    }

    // Read a file of strings
    static std::vector<std::string> ReadLinesFromFile(const std::string& filename)
    {
        std::vector<std::string> lines;
        std::string line;

        std::ifstream file(filename);

        while (std::getline(file, line))
        {
            if (line.length() > 0) lines.emplace_back(line);
        }

        return lines;
    }


    std::vector<std::string> categories;
    TensorShape inputShape;
    TensorShape outputShape;
    size_t inputSize;
    std::vector<float> input;
    cv::Mat image;
    size_t outputSize;
    std::vector<float> predictions;
    std::chrono::time_point<std::chrono::steady_clock> start;
};



Tutorial tutorial;

int8_t model_InputCallback(float* buffer)
{
    return tutorial.InputCallback(buffer);
}

void model_OutputCallback(float* buffer)
{
    tutorial.OutputCallback(buffer);
}

array<Mat, NUMERODEFICHEROS> loadFrames(string rutaBaseFicheros){
    array<Mat, NUMERODEFICHEROS> frames;
    string rutaCompletaFichero;
    for(int i = 0; i<NUMERODEFICHEROS; i++){
        rutaCompletaFichero = rutaBaseFicheros + to_string(i) + ".jpg";
        std::cout<< "La ruta del fichero es: " << rutaCompletaFichero << endl;
        Mat imagenLeida = imread(rutaCompletaFichero, IMREAD_COLOR);
        
			Mat imagenExtraida = cv::Mat::zeros(Size(NUMCOLS,NUMROWS), imagenLeida.type());
			imagenLeida(cv::Rect(0,0, NUMCOLS,NUMROWS)).copyTo(imagenExtraida(cv::Rect(0,0,NUMCOLS,NUMROWS)));					
				        
        
        
        //frames[i] = imread(rutaCompletaFichero, IMREAD_COLOR); // Read the file
        frames[i]= imagenExtraida;
    }

    return frames;
}

//Update all active detections
/*void updateDetections(array<Detection, MAXDETECTIONS> &A, float alpha, float delta, int currentNumberOfDetections){
        for(array<Detection, MAXDETECTIONS>::iterator it = A.begin(); it != A.end();++it){
            if((*it).active){
                (*it).updateDetection(alpha, delta);
                if(((*it).x1 > NUMCOLS) || ((*it).x2 > NUMROWS) || (((*it).x1 + (*it).x3) > NUMCOLS) || (((*it).x2 + (*it).x3)>NUMROWS)){
                        (*it).active = false;
                }
            }
        }
}*/

void normalizeDetections (array<Detection, MAXDETECTIONS> &A, int currentNumberOfDetections){
	////std::cout<<"Inside normalizeDetections" << endl;
	float cummulativePi = 0;
	for (int i = 0; i< currentNumberOfDetections;i++){
		cummulativePi += A[i].pi;
	}
	for (int i = 0; i< currentNumberOfDetections;i++){
		A[i].pi = A[i].pi/cummulativePi;
	}
}

void updateDetections(array<Detection, MAXDETECTIONS> &A, float alpha, float delta, int currentNumberOfDetections, int & numberOfOptimizedDetection, Mat & imagenActual, std::array<std::string,1000> & anomalousCategories, int detectionThreshold){
		for (int i = 0; i< currentNumberOfDetections;i++){
			 A[i].updateDetection(alpha,delta);
		}
		normalizeDetections (A,currentNumberOfDetections);
}


/*void updateDetections(array<Detection, MAXDETECTIONS> &A, float alpha, float delta, int currentNumberOfDetections, int & numberOfOptimizedDetection, Mat & imagenActual, std::array<std::string,1000> & anomalousCategories, int detectionThreshold){
         
		 for(array<Detection, MAXDETECTIONS>::iterator it = A.begin(); it != A.end();++it){
            if((*it).active){
                (*it).updateDetection(alpha, delta);
               // if(((*it).x1 > NUMCOLS) || ((*it).x2 > NUMROWS) || (((*it).x1 + (*it).x3) > NUMCOLS) || (((*it).x2 + (*it).x3)>NUMROWS)){
                        //(*it).active = false;
               // }
           	}
		 }
		
        if (A[numberOfOptimizedDetection].active){
	        Mat imagenExtraida = cv::Mat::zeros(Size(A[numberOfOptimizedDetection].ventana.tamanoHorizontal,A[numberOfOptimizedDetection].ventana.tamanoVertical), imagenActual.type());
			  imagenActual(cv::Rect(A[numberOfOptimizedDetection].ventana.posX,A[numberOfOptimizedDetection].ventana.posY, A[numberOfOptimizedDetection].ventana.tamanoHorizontal,A[numberOfOptimizedDetection].ventana.tamanoVertical)).copyTo(imagenExtraida(cv::Rect(0,0,A[numberOfOptimizedDetection].ventana.tamanoHorizontal,A[numberOfOptimizedDetection].ventana.tamanoVertical)));					
											
											//cout << "Despues de almacenImagenes" << endl;
																			
											//auto input = tutorialHelpers::PrepareImageForModel(imagenExtraida, inputShape.columns, inputShape.rows);
											//cout << "Despues de input" << endl;
			        						// Send the image to the compiled model and fill the predictions vector with scores, measure how long it takes
			        						
			        						//model_predict(input, predictions);
			        						
			        						// Get the value of the top 5 predictions
			        						//auto top5 = tutorialHelpers::GetTopN(predictions, 5);
			        						
			        						
				tutorial.PrepareNextImage(imagenExtraida);
	  						
	  			//model_Predict(input, predictions);
	  			tutorial.Step();
	  						
	  			// Get the value of the top 5 predictions
	  			auto top5 = tutorial.GetTopN(5);		        						
	  						
	  						
	  						
				//Test whether the category is some of our anomalous categories
				
				int indiceTop5 = 0;
				bool encontrado = false;
				while ((!encontrado)&&(indiceTop5 < top5.size())){
					if(anomalousCategories[top5.at(indiceTop5).first].compare("Empty")!=0){
						encontrado = true;									
					
					}else{
					
						indiceTop5++;
					}								
				}
				
				//if (indiceTop5<top5.size()){ //Al menos uno esta en la lista de anomalos y merece la pena sacar la ventana
				if (!((encontrado)&&(((top5.at(indiceTop5).second) *100.0 ) >= detectionThreshold))){
					A[numberOfOptimizedDetection].active = false;
				}
			}
}*/



/*float closedIntervalRand(float x0, float x1)
{
    return x0 + (x1 - x0) * rand() / ((float) RAND_MAX);
}*/

array<std::string,NUMCOLSRECORDFILE> splitString(string line){

    array<std::string, NUMCOLSRECORDFILE>returnArray;
    std::string delimiter = " ";
    size_t pos = 0;
    std::string token;
    int indexReturnArray = 0;
    while ((pos = line.find(delimiter)) != std::string::npos) {
        
        returnArray[indexReturnArray] = line.substr(0, pos);
        ////std::cout << returnArray[indexReturnArray] << " ";//  std::endl;
        ////std::cout << indexReturnArray << endl;
        ////std::cout << line << endl;
        line.erase(0, pos + delimiter.length());
        indexReturnArray++;
    }
    returnArray[indexReturnArray] = line;
    ////std::cout << std::endl;

    for (int i = 0; i<NUMCOLSRECORDFILE;i++){
        //std::cout << returnArray[i] << " ";
    }
    //std::cout << endl;

    return returnArray;


}






float closedIntervalRand(float x0, float x1)
{

    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_int_distribution<int>  distr((int)x0, (int)x1);
    return (float) distr(generator);
}

float randomFloatGenerator(float x0, float x1){

	float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	return r;
}


// Detection adaptDetection(int x1, int x2, int x3){
// 	cout << "La prepostdeteccion generada es... " << to_string(x1) << ";" << to_string(x2) << ";" <<  to_string(x3)<< endl;

// 	Detection detection;
// 	if ((x1 > NUMCOLS)||(x2 > NUMROWS)){
//     	detection = Detection();
    	
//   	 }else if (((x1+x3)>NUMCOLS)&&((x2+x3)>NUMROWS)) {
//       detection = Detection(x1-(x1+x3-NUMCOLS), x2-(x2+x3-NUMROWS), x3, randomFloatGenerator(0,1));
//   	 }else if((x1+x3)>NUMCOLS){
// 		detection = Detection(x1-x3, x2, x3, randomFloatGenerator(0,1));	 
//   	 }else if ((x2+x3) > NUMROWS){
// 		detection = Detection(x1, x2-x3, x3, randomFloatGenerator(0,1));  	 
//   	 }else if(((x1+x3) < 0)||((x2+x3)<0)){
// 		detection = Detection();  	 
//   	 }else if((x1<0)&&(x2<0)){
//   	 	detection = Detection(x1+x3, x2+x3, x3, randomFloatGenerator(0,1));	
//   	 }else if(x1<0){
// 		detection = Detection(x1+x3, x2, x3, randomFloatGenerator(0,1));	
//   	 }else if(x2<0){
// 		detection = Detection(x1, x2+x3, x3, randomFloatGenerator(0,1));	  	 
//   	 }else{
// 		detection = Detection(x1, x2, x3, randomFloatGenerator(0,1));  	 
//   	 }


//     return detection;
// }

Detection adaptDetection(int x1, int x2, int x3){
	//std::cout << "La prepostdeteccion generada es... " << to_string(x1) << ";" << to_string(x2) << ";" <<  to_string(x3)<< endl;

	Detection detection;
	// if ((x1 > NUMCOLS)||(x2 > NUMROWS)){
    // 	detection = Detection();
    	
  	//  }else if (((x1+x3)>NUMCOLS)&&((x2+x3)>NUMROWS)) {
    //   detection = Detection(x1-(x1+x3-NUMCOLS), x2-(x2+x3-NUMROWS), x3, randomFloatGenerator(0,1));
  	//  }else if((x1+x3)>NUMCOLS){
	// 	detection = Detection(x1-x3, x2, x3, randomFloatGenerator(0,1));	 
  	//  }else if ((x2+x3) > NUMROWS){
	// 	detection = Detection(x1, x2-x3, x3, randomFloatGenerator(0,1));  	 
  	//  }else if(((x1+x3) < 0)||((x2+x3)<0)){
	// 	detection = Detection();  	 
  	//  }else if((x1<0)&&(x2<0)){
  	//  	detection = Detection(x1+x3, x2+x3, x3, randomFloatGenerator(0,1));	
  	//  }else if(x1<0){
	// 	detection = Detection(x1+x3, x2, x3, randomFloatGenerator(0,1));	
  	//  }else if(x2<0){
	// 	detection = Detection(x1, x2+x3, x3, randomFloatGenerator(0,1));	  	 
  	//  }else{
	// 	detection = Detection(x1, x2, x3, randomFloatGenerator(0,1));  	 
  	//  }

	if ((x3>0)&&(x1 > 0)&&(x1<=NUMCOLS) && ((x1+x3) > 0) && ((x1+x3) <=NUMCOLS)&&
	 (x2>0)&&(x2<=NUMROWS)&&((x2+x3)>0) && ((x2+x3)<=NUMROWS)){
		 	detection = Detection(x1, x2, x3, randomFloatGenerator(0,1)); 
	}else{
		detection = Detection();
	}

		//cout << "La prepostpostdeteccion generada es... " << to_string(detection.x1) << ";" << to_string(detection.x2) << ";" <<  to_string(detection.x3)<< endl;

    return detection;
}


Detection generateRandomDetection(){
    int x1 = (int)closedIntervalRand(1,NUMCOLS);
    int x2 = (int)closedIntervalRand(1,NUMROWS);
    int x3 = (int)closedIntervalRand(1,DETECTIONMAXSIZE);
//	x1 = x1 - x3/2;
//	x2 = x2 - x3/2;
    int superficie = x3*x3;
    Detection detection;
	return adaptDetection(x1,x2,x3);

     /*if((x1 < NUMCOLS) && (x1 > 0) && (x2 < NUMROWS) &&(x2 >0) && ((x1 + x3) < NUMCOLS)  && ((x2 + x3)<NUMROWS)){
                 if ((superficie>=MINIMUMFRAME)|| (superficie <= MAXIMUMFRAME)){
                        detection = Detection(x1, x2, x3, closedIntervalRand(0,1));
                 }else{
                     detection = Detection();
                 }
        }else{
            detection = Detection();
        }*/
        
//     if ((x1 > NUMCOLS)||(x2 > NUMROWS)){
//     	detection = Detection();
    	
//   	 }else if (((x1+x3)>NUMCOLS)&&((x2+x3)>NUMROWS)) {
//       detection = Detection(x1-(x1+x3-NUMCOLS), x2-(x2+x3-NUMROWS), x3, randomFloatGenerator(0,1));
//   	 }else if((x1+x3)>NUMCOLS){
// 		detection = Detection(x1-(x1+x3-NUMCOLS), x2, x3, randomFloatGenerator(0,1));	 
//   	 }else if ((x2+x3) > NUMROWS){
// 		detection = Detection(x1, x2-(x2+x3-NUMROWS), x3, randomFloatGenerator(0,1));  	 
//   	 }else if(((x1+x3) < 0)||((x2+x3)<0)){
// 		detection = Detection();  	 
//   	 }else if((x1<0)&&(x2<0)){
//   	 	detection = Detection(x1+x3, x2+x3, x3, randomFloatGenerator(0,1));	
//   	 }else if(x1<0){
// 		detection = Detection(x1+x3, x2, x3, randomFloatGenerator(0,1));	
//   	 }else if(x2<0){
// 		detection = Detection(x1, x2+x3, x3, randomFloatGenerator(0,1));	  	 
//   	 }else{
// 		detection = Detection(x1, x2, x3, randomFloatGenerator(0,1));  	 
//   	 }

//    //detection = Detection(x1, x2, x3 , closedIntervalRand(0,1), imagen);
//     return detection;
}

/*int calculateDetectionIndex(array<Detection, MAXDETECTIONS> & A, int currentNumberOfDetections){
    int indexDetection;
    array<float,MAXDETECTIONS + 1> histc;
    array<float, MAXDETECTIONS> cumsum;
    float numeroRandom = closedIntervalRand(0,1);
    for(int i = 0; i<MAXDETECTIONS; i++){
        float sumaParcial =0;
        for(int j = 0; j<MAXDETECTIONS; j++){
            sumaParcial += A[j].pi;
        }
        cumsum[i] = sumaParcial;
    }
    histc[0] =0;
    for (int i = 1; i<MAXDETECTIONS+1; i++){
        histc[i] = cumsum[i-1];
    }
    int indice = 0;
    while((numeroRandom < histc[indice])&&(indice < MAXDETECTIONS+1)){
        indice++;
    }
    if (indice <  MAXDETECTIONS+1){
        indexDetection = indice;
    }else{
        indexDetection = MAXDETECTIONS;
    }

    return indexDetection;
}*/


// int calculateDetectionIndex(array<Detection, MAXDETECTIONS> & A, int currentNumberOfDetections){
//     int indexDetection;
//     array<float,MAXDETECTIONS + 1> histc;
//     array<float, MAXDETECTIONS> cumsum;
//     float numeroRandom = randomFloatGenerator(0,1);
//     for(int i = 0; i<currentNumberOfDetections; i++){
//         float sumaParcial =0;
//         for(int j = 0; j<currentNumberOfDetections; j++){
//             sumaParcial += A[j].pi;
//         }
//         cumsum[i] = sumaParcial;
//     }
//     histc[0] =0;
//     for (int i = 1; i<currentNumberOfDetections+1; i++){
//         histc[i] = cumsum[i-1];
//     }
//     int indice = 0;
//     while((numeroRandom < histc[indice])&&(indice < currentNumberOfDetections+1)){
//         indice++;
//     }
//     if (indice <  currentNumberOfDetections+1){
//         indexDetection = indice;
//     }else{
//         indexDetection = currentNumberOfDetections;
//     }

//     return indexDetection;
// }



int calculateDetectionIndex(array<Detection, MAXDETECTIONS> & A, int currentNumberOfDetections){
    int indexDetection;
    array<float,MAXDETECTIONS + 1> histc;
    array<float, MAXDETECTIONS> cumsum;
    float numeroRandom = randomFloatGenerator(0,1);
    for(int i = 0; i<currentNumberOfDetections; i++){
        float sumaParcial =0;
        for(int j = 0; j<=i; j++){
            sumaParcial += A[j].pi;
        }
        cumsum[i] = sumaParcial;
    }
    histc[0] =0;
    for (int i = 1; i<currentNumberOfDetections+1; i++){
        histc[i] = cumsum[i-1];
    }
    int indice = 0;
    while((numeroRandom < histc[indice])&&(indice < currentNumberOfDetections+1)){
        indice++;
    }
    if (indice <  currentNumberOfDetections+1){
        indexDetection = indice;
    }else{
        indexDetection = currentNumberOfDetections;
    }

    return indexDetection;
}


// Detection adaptDetection(int x1, int x2, int x3){
// 	Detection detection;
// 	if ((x1 > NUMCOLS)||(x2 > NUMROWS)){
//     	detection = Detection();
    	
//   	 }else if (((x1+x3)>NUMCOLS)&&((x2+x3)>NUMROWS)) {
//       detection = Detection(x1-(x1+x3-NUMCOLS), x2-(x2+x3-NUMROWS), x3, randomFloatGenerator(0,1));
//   	 }else if((x1+x3)>NUMCOLS){
// 		detection = Detection(x1-(x1+x3-NUMCOLS), x2, x3, randomFloatGenerator(0,1));	 
//   	 }else if ((x2+x3) > NUMROWS){
// 		detection = Detection(x1, x2-(x2+x3-NUMROWS), x3, randomFloatGenerator(0,1));  	 
//   	 }else if(((x1+x3) < 0)||((x2+x3)<0)){
// 		detection = Detection();  	 
//   	 }else if((x1<0)&&(x2<0)){
//   	 	detection = Detection(x1+x3, x2+x3, x3, randomFloatGenerator(0,1));	
//   	 }else if(x1<0){
// 		detection = Detection(x1+x3, x2, x3, randomFloatGenerator(0,1));	
//   	 }else if(x2<0){
// 		detection = Detection(x1, x2+x3, x3, randomFloatGenerator(0,1));	  	 
//   	 }else{
// 		detection = Detection(x1, x2, x3, randomFloatGenerator(0,1));  	 
//   	 }


//     return detection;
// }

std::piecewise_linear_distribution<double> triangularDistribution(double min, double peak, double max)
{
    std::array<double, 3> i{min, peak, max};
    std::array<double, 3> w{0, 1, 0};
    return std::piecewise_linear_distribution<double>{i.begin(), i.end(), w.begin()};
}

// Detection generateTriangularDetection(array<Detection, MAXDETECTIONS> & A, float sigma, int currentNumberOfDetections){
// 	float sumAccX1 = 0.0;
// 	float sumAccX2 = 0.0;
// 	float sumAccX3 = 0.0;
// 	std::default_random_engine generator;

// 	//calculamos x1:
// 	int x1,x2,x3;
// 	for (int i = 0; i<currentNumberOfDetections; i++){
// 		std::piecewise_linear_distribution<double> distributionX1 = triangularDistribution (1,A[i].x1,NUMCOLS);
// 		std::piecewise_linear_distribution<double> distributionX2 = triangularDistribution (1,A[i].x2,NUMROWS);
// 		std::piecewise_linear_distribution<double> distributionX3 = triangularDistribution (1,A[i].x3,DETECTIONMAXSIZE);
// 		sumAccX1 +=A[i].pi*distributionX1(generator);
// 		sumAccX2 +=A[i].pi*distributionX2(generator);
// 		sumAccX3 +=A[i].pi*distributionX3(generator);
// 	}
// 	x1 = sumAccX1/currentNumberOfDetections;
// 	x2 = sumAccX2/currentNumberOfDetections;
// 	x3 = sumAccX3/currentNumberOfDetections;

// 	return adaptDetection(x1,x2,x3);


// }


// Detection generateGaussianDetection(array<Detection, MAXDETECTIONS> & A, float sigma, int currentNumberOfDetections){
// 	float probabilityPositiveThreshold = 0.5;
// 	float probabilityPositive;
// 	probabilityPositive = randomFloatGenerator(0,1);
//     Detection detection;
// 	 int sigmaCols = (int)(sigma*NUMCOLS);
// 	 int sigmaRows = (int)(sigma*NUMROWS);
// 	 int sigmaSize = (int)(sigma*DETECTIONMAXSIZE);
//     int indexDetection = calculateDetectionIndex(A, currentNumberOfDetections);
// 	int x1, x2, x3;
// 	if (probabilityPositive >= probabilityPositiveThreshold){
// 		x1 = (int)(A[indexDetection].x1 + sigmaCols * randomFloatGenerator(0,1)); //CAmbiar la generacion de numeros por una normal.
// 		x2 = (int)(A[indexDetection].x2 + sigmaRows * randomFloatGenerator(0,1));
// 		x3 = (int)(A[indexDetection].x3 + sigmaSize * randomFloatGenerator(0,1));
// 	}else{
// 	 	// x1 = (int)((A[indexDetection].x1 + sigmaCols * randomFloatGenerator(0,1))-A[indexDetection].x3 * randomFloatGenerator(0,1));
// 	 	// x2 = (int)((A[indexDetection].x2 + sigmaRows * randomFloatGenerator(0,1))-A[indexDetection].x3 * randomFloatGenerator(0,1));
// 	 	// x3 = (int)((A[indexDetection].x3 + sigmaSize * randomFloatGenerator(0,1))-A[indexDetection].x3 * randomFloatGenerator(0,1));
// 		x1 = (int)((A[indexDetection].x1 + sigmaCols * randomFloatGenerator(0,1))-A[indexDetection].x3);
// 	 	x2 = (int)((A[indexDetection].x2 + sigmaRows * randomFloatGenerator(0,1))-A[indexDetection].x3);
// 	 	x3 = (int)((A[indexDetection].x3 + sigmaSize * randomFloatGenerator(0,1))-A[indexDetection].x3);
// 		 //cout << "<<<<<<<<<<<<< x1 = " << x1 << " x2 = " << x2 << " x3 = " << x3 << endl;
// 		if (x3 < 0){
// 			x3 = -x3;
// 		}
// 	}
//     int superficie = x3*x3;
//     return adaptDetection(x1,x2,x3);
//}


// Detection generateTriangularDetection(array<Detection, MAXDETECTIONS> & A, float sigma, int currentNumberOfDetections){
// 	float probabilityPositiveThreshold = 0.5;
// 	float probabilityPositive;
// 	//probabilityPositive = randomFloatGenerator(0,1);
// 	//std::default_random_engine generator;
// 	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//    	std::default_random_engine generator(seed);

//     Detection detection;
//     int indexDetection = calculateDetectionIndex(A, currentNumberOfDetections);
// 	int x1, x2, x3;
// 	std::piecewise_linear_distribution<double> distributionX1 = triangularDistribution (A[indexDetection].x1-sigma, A[indexDetection].x1, A[indexDetection].x1+sigma);
// 	std::piecewise_linear_distribution<double> distributionX2 = triangularDistribution (A[indexDetection].x2-sigma, A[indexDetection].x2, A[indexDetection].x2+sigma);
// 	std::piecewise_linear_distribution<double> distributionX3 = triangularDistribution (A[indexDetection].x3-sigma, A[indexDetection].x3, A[indexDetection].x3+sigma);
// 	x1 = (int)(distributionX1(generator)); //Cambiar la generacion de numeros por una normal.
// 	x2 = (int)(distributionX2(generator)); 
// 	x3 = (int)(distributionX3(generator)); 
	
//     int superficie = x3*x3;
//     return adaptDetection(x1,x2,x3);
// }


Detection generateTriangularDetection(array<Detection, MAXDETECTIONS> & A, float sigma, int currentNumberOfDetections){
    Detection detection;
	float sigmaCols = sigma*NUMCOLS;
	float  sigmaRows = sigma*NUMROWS;
	float sigmaSize = sigma*DETECTIONMAXSIZE;


    int indexDetection = calculateDetectionIndex(A, currentNumberOfDetections);
	int x1, x2, x3;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  	std::default_random_engine generator(seed);
	//std::default_random_engine generator;
	std::piecewise_linear_distribution<double> distribution = triangularDistribution (-1, 0, 1);
		// x1 = (int)(A[indexDetection].x1 + sigmaCols * randomFloatGenerator(0,1)); //CAmbiar la generacion de numeros por una normal.
		// x2 = (int)(A[indexDetection].x2 + sigmaRows * randomFloatGenerator(0,1));
		// x3 = (int)(A[indexDetection].x3 + sigmaSize * randomFloatGenerator(0,1));
		float numberX1 = distribution(generator);
		float numberX2 = distribution(generator);
		float numberX3 = distribution(generator);
		//cout << "Los numeros generados son... " << to_string(numberX1) << ";" << to_string(numberX2) << ";" <<  to_string(numberX3)<< endl;

	x1 = (int)(A[indexDetection].x1 + sigmaCols * numberX1);
	x2 = (int)(A[indexDetection].x2 + sigmaRows * numberX2);
	x3 = (int)(A[indexDetection].x3 + sigmaSize * numberX3);

	//cout << "La predeteccion generada es... " << to_string(x1) << ";" << to_string(x2) << ";" <<  to_string(x3)<< endl;

	// if (x3 == 0){
	// 	x3 = 300;
	// }else if (x3 < 0){
 	// 	x3 = -x3;
 	// }
	
    int superficie = x3*x3;
    return adaptDetection(x1,x2,x3);
}




Detection generateGaussianDetection(array<Detection, MAXDETECTIONS> & A, float sigma, int currentNumberOfDetections){
    Detection detection;
	float sigmaCols = sigma*NUMCOLS;
	float  sigmaRows = sigma*NUMROWS;
	float sigmaSize = sigma*DETECTIONMAXSIZE;


    int indexDetection = calculateDetectionIndex(A, currentNumberOfDetections);
	int x1, x2, x3;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  	std::default_random_engine generator(seed);
	//std::default_random_engine generator;
  	//std::normal_distribution<double> distribution(0,0.1);
	std::normal_distribution<double> distribution(0,1);

		// x1 = (int)(A[indexDetection].x1 + sigmaCols * randomFloatGenerator(0,1)); //CAmbiar la generacion de numeros por una normal.
		// x2 = (int)(A[indexDetection].x2 + sigmaRows * randomFloatGenerator(0,1));
		// x3 = (int)(A[indexDetection].x3 + sigmaSize * randomFloatGenerator(0,1));
		float numberX1 = distribution(generator);
		float numberX2 = distribution(generator);
		float numberX3 = distribution(generator);
		//cout << "Los numeros generados son... " << to_string(numberX1) << ";" << to_string(numberX2) << ";" <<  to_string(numberX3)<< endl;

	x1 = (int)(A[indexDetection].x1 + sigmaCols * numberX1);
	x2 = (int)(A[indexDetection].x2 + sigmaRows * numberX2);
	x3 = (int)(A[indexDetection].x3 + sigmaSize * numberX3);

	//cout << "La predeteccion generada es... " << to_string(x1) << ";" << to_string(x2) << ";" <<  to_string(x3)<< endl;

	// if (x3 == 0){
	// 	x3 = 300;
	// }else if (x3 < 0){
 	// 	x3 = -x3;
 	// }
	
    int superficie = x3*x3;
    return adaptDetection(x1,x2,x3);
}


Detection generateStudentTDetection(array<Detection, MAXDETECTIONS> & A, float sigma, int currentNumberOfDetections){
    Detection detection;
	float sigmaCols = sigma*NUMCOLS;
	float  sigmaRows = sigma*NUMROWS;
	float sigmaSize = sigma*DETECTIONMAXSIZE;


    int indexDetection = calculateDetectionIndex(A, currentNumberOfDetections);
	int x1, x2, x3;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  	std::default_random_engine generator(seed);
  	std::student_t_distribution<double> distribution(2.1);

	float numberX1 = distribution(generator);
	float numberX2 = distribution(generator);
	float numberX3 = distribution(generator);
		
		//cout << "Los numeros generados son... " << to_string(numberX1) << ";" << to_string(numberX2) << ";" <<  to_string(numberX3)<< endl;

	x1 = (int)(A[indexDetection].x1 + sigmaCols * numberX1);
	x2 = (int)(A[indexDetection].x2 + sigmaRows * numberX2);
	x3 = (int)(A[indexDetection].x3 + sigmaSize * numberX3);

	//cout << "La predeteccion generada es... " << to_string(x1) << ";" << to_string(x2) << ";" <<  to_string(x3)<< endl;

	
    int superficie = x3*x3;
    return adaptDetection(x1,x2,x3);
}


Detection generatePossibleDetection(array<Detection, MAXDETECTIONS> & A, float sigma, float q, Mat & imagen, bool mixture,int currentNumberOfDetections, string probabilityDistribution){
    //float q = 0.01;
    //float sigma = 0.5;
    Detection detection;
    if (mixture){
		if ((randomFloatGenerator(0,1)<q)||(currentNumberOfDetections == 0)){
			//cout << "generateRandomDetection" << endl;
			detection = generateRandomDetection();
		}else{
			//cout << "generateChungDetection" << endl;
			//detection = generateChungDetection(A,sigma,currentNumberOfDetections);
			if (probabilityDistribution == "GAUSSIAN"){
				//cout << "entering gaussian..." << endl;
				detection = generateGaussianDetection(A,sigma,currentNumberOfDetections);
				//cout << "La deteccion generada es... " << to_string(detection.x1) << ";" << to_string(detection.x2) << ";" <<  to_string(detection.x3)<< endl;

			}else if (probabilityDistribution == "TRIANGLE"){
				//cout << "entering triangle..." << endl;

				detection = generateTriangularDetection(A,sigma,currentNumberOfDetections);
				//cout << "La deteccion generada es... " << to_string(detection.x1) << ";" << to_string(detection.x2) << ";" <<  to_string(detection.x3)<< endl;

			}else{
				//cout << "entering student..." << endl;

				detection = generateStudentTDetection(A,sigma,currentNumberOfDetections);
				//cout << "La deteccion generada es... " << to_string(detection.x1) << ";" << to_string(detection.x2) << ";" <<  to_string(detection.x3)<< endl;

			}
			//detection = generateGaussianDetection(A,sigma,currentNumberOfDetections);
		}
    }else{
    	detection = generateRandomDetection();
    }

    return detection;
}

// Detection generatePossibleDetection(array<Detection, MAXDETECTIONS> & A, float sigma, float q, Mat & imagen, bool mixture,int currentNumberOfDetections){
//     //float q = 0.01;
//     //float sigma = 0.5;
//     Detection randomDetection;
// 	Detection triangularDetection;
// 	Detection detection;
//     if (mixture){
// 		randomDetection = generateRandomDetection();
// 		cout<< "generando deteccion aleatoria" << endl;
// 		randomDetection.x1 = (int)q*randomDetection.x1;
// 		randomDetection.x2 = (int)q*randomDetection.x2;
// 		randomDetection.x3 = (int)q*randomDetection.x3;

// 		triangularDetection = generateTriangularDetection(A,sigma,currentNumberOfDetections);
// 		cout<< "generando deteccion triangular" << endl;
// 		triangularDetection.x1 = (int)((1-q)*triangularDetection.x1);
// 		triangularDetection.x2 = (int)((1-q)*triangularDetection.x2);
// 		triangularDetection.x3 = (int)((1-q)*triangularDetection.x3);

// 		detection = adaptDetection(randomDetection.x3 + triangularDetection.x3, randomDetection.x3 + triangularDetection.x3, randomDetection.x3 + triangularDetection.x3);
//     }else{
//     	detection = generateRandomDetection();
//     }

//     return detection;
// }



/* int lookForWeakestDetection(array<Detection, MAXDETECTIONS>&A){
    int weakestDetection = 0;
    float currentPiValue = A[0].pi;
    for (int i = 0; i<MAXDETECTIONS; i++){
        if (A[i].pi <currentPiValue){
            //cout << "Dentro del if" << endl;
            weakestDetection = i;
            currentPiValue = A[i].pi;
        }
    }
    return weakestDetection;
}*/

bool compare4Cuadrants(Detection & detectionA, Detection & detectionB){
	bool resultado = false; 
	//primer cuadrante
	if (((detectionB.x1+detectionB.x3)<(detectionA.x1))||(detectionB.x1>(detectionA.x1+detectionA.x3))||((detectionB.x2+detectionB.x3)<detectionA.x2)||(detectionB.x2>(detectionA.x2+detectionA.x3))){
		resultado = true;
	}else{
		float centroXA = detectionA.x1 + detectionA.x3/2;
		float centroYA = detectionA.x2 + detectionA.x3/2;
		float centroXB = detectionB.x1 + detectionB.x3/2;
		float centroYB = detectionB.x2 + detectionB.x3/2;
		
		float distanciaEuclidea = sqrt(pow(centroXA-centroXB,2)+pow(centroYA-centroYB,2));
		
		if (distanciaEuclidea >= EUCLIDEANDISTANCETHRESHOLD){
			resultado = true;
		}
	}
	return resultado;
}


bool matchesExistent(array<Detection, MAXDETECTIONS>&A, Detection & detection, int & currentNumberOfDetections){
	//int tol = 100;
	bool matches = false;
	int i = 0;	
	while((i<currentNumberOfDetections)&&(matches==false)){
		if(A[i].active){
			if (compare4Cuadrants(A[i], detection)==false){
				matches = true;
			}
		}
		i++;
	}
	
	
	
	/*for (int i = 0; i<MAXDETECTIONS; i++){
		if(A[i].active){
			if (compare4Cuadrants(A[i], detection)==false){
				matches = true;
			}
		}
	}*/
	return matches;
}

void addDetectionAA(array<Detection, MAXDETECTIONS>&A, Detection detection, int & currentNumberOfDetections){
	//std::cout << "inside AddDetectionAA" << endl;
	A[currentNumberOfDetections] = detection;
	A[currentNumberOfDetections].active = true;
    //std::cout<< "Posible deteccion anadida. " << "x1= " << A[currentNumberOfDetections].x1 << " x2= " << A[currentNumberOfDetections].x2 << " x3= " << A[currentNumberOfDetections].x3 << "posicion en array: " << currentNumberOfDetections << "Pi= " << A[currentNumberOfDetections].pi << endl;
	currentNumberOfDetections++;
	/*
    int i = 0;
    while ((A[i].active == true) && (i<MAXDETECTIONS)){
        i++;
    }
    if(i<MAXDETECTIONS){
        //cout << "i menor que MAXDETECTIONS" << endl;
        A[i] = detection;

        A[i].active = true;
        cout<< "Posible deteccion anadida. " << "x1= " << A[i].x1 << " x2= " << A[i].x2 << " x3= " << A[i].x3 << "posicion en array: " << i << endl;
		
    }else{

        int weakestDetection = lookForWeakestDetection(A);

        //cout << "La weakest detection es: " << weakestDetection <<endl;
        A[weakestDetection]= detection;
        A[weakestDetection].active = true;
        //cout<< "Posible deteccion anadida. " << "x1= " << A[weakestDetection].x1 << " x2= " << A[weakestDetection].x2 << " x3= " << A[weakestDetection].x3 << endl;
    }*/

}


// Read a file of strings
static std::vector<std::string> ReadLinesFromFile(const std::string& filename)
{
    std::vector<std::string> lines;
    std::string line;

    std::ifstream file(filename);

    while (std::getline(file, line))
    {
        if (line.length() > 0) lines.emplace_back(line);
    }

    return lines;
}


std::array<std::string,1000> orderAnomalousCategories(std::vector<std::string>& categories, std::vector<std::string>& anomalousC){
	std::array<std::string,1000> anomalousCategories;
		
		for(int i = 0; i<1000; i++){
			anomalousCategories[i] = "Empty";
			
				
		}
	
	
	/*cout << "Anomalous categories" << endl;
	
	for(int i = 0; i< anomalousCategories.size(); i++){
		cout << anomalousCategories[i] << endl;
				
	}*/
	for(int i = 0; i<1000;i++){
		//cout << "bucle externo = " << i << endl;
		bool encontrado = false;
		int j = 0;
		while ((!encontrado)&&(j<anomalousC.size())){
			//cout << "Dentro = " << j <<   endl;
			if (categories.at(i).compare(anomalousC.at(j))==0){
				encontrado = true;
				//cout << "Encontrado " << endl;
			}else{
				//cout << "No Encontrado" << endl;
				j++;
			}
		
			//cout << "Despues de primera comparacion" << endl;
			
			
		}
		if (j<anomalousC.size()){
				//strcpy(anomalousCategories[i], categories[i]);
				anomalousCategories[i] =  categories[i];
		} 
			
	}
	
	return anomalousCategories;

}

void initializeAnomalousObjectsFound(std::array<int, 1000> & anomalousObjectsFound){
	for(int i = 0; i<1000;i++){
		anomalousObjectsFound[i]=0;	
	
	}

}

/*int calculateNumberOfAnomalousObjectsFound(std::array<int, 1000> & anomalousObjectsFound){
	int numberOfAnomalousObjectsFound = 0;
	for(int i = 0; i< 1000; i++){
			if(anomalousObjectsFound[i]>0){
						numberOfAnomalousObjectsFound++;	
						cout << "sumamos uno, categoria: " << i+1 << endl;
			}
	}
	
	return numberOfAnomalousObjectsFound;
}*/

int calculateNumberOfAnomalousObjectsFound(std::array<int, 1000> & anomalousObjectsFound){
	int numberOfAnomalousObjectsFound = 0;
	for(int i = 0; i< 1000; i++){
			//if(anomalousObjectsFound[i]>0){
						numberOfAnomalousObjectsFound+=anomalousObjectsFound[i];	
					//	cout << "sumamos uno, categoria: " << i+1 << endl;
			//}
	}
	
	return numberOfAnomalousObjectsFound;
}

int calculateNumberOfDifferentAnomalousObjectsFound(std::array<int, 1000> & anomalousObjectsFound){
	int numberOfDifferentAnomalousObjectsFound = 0;
	for(int i = 0; i< 1000; i++){
			//if(anomalousObjectsFound[i]>0){
				if (anomalousObjectsFound[i] != 0){
						numberOfDifferentAnomalousObjectsFound++;
				}	
					//	cout << "sumamos uno, categoria: " << i+1 << endl;
			//}
	}
	
	return numberOfDifferentAnomalousObjectsFound;
}

float calculatePercentageOfAnomalousObjectsFound(int numberOfAnomalousObjectsFound){
	float percentageOfAnomalousObjectsFound = 0;
	percentageOfAnomalousObjectsFound = 	(numberOfAnomalousObjectsFound*100)/(NUMANOMALOUSOBJECTS*NUMERODEFICHEROS);
	
	return percentageOfAnomalousObjectsFound;
	
}

std::array<struct objectFound, NUMANOMALOUSOBJECTS>initializedifferentObjectsFound(){
	std::array<struct objectFound, NUMANOMALOUSOBJECTS> differentObjectsFound;
	for (int i = 0; i< NUMANOMALOUSOBJECTS; i++){
		differentObjectsFound[i].numberOfDetections =0;

	}
	return differentObjectsFound;
}






void pruebasConNumeroVentanas(int numRandomWindows, int currentTestNumber, float alpha, float delta, float sigma, float q, TensorShape inputShape,std::vector<float> & predictions, array<Detection,MAXDETECTIONS> & A, array<Mat, NUMERODEFICHEROS> & almacenImagenes, std::array<std::string,1000> & anomalousCategories, std::vector<std::string> & categories, std::array<int, 1000> & anomalousObjectsFound, double & frameProcessingTime, bool displayResultsWindow, int detectionThreshold, bool mixture, string probabilityDistribution, int progressCounter, int totalExecutions){
	// Declare a variable to hold the prediction times
    std::vector<double>  predictionTimes;
    double meanTimeToPredict = 0.0;
	 string posibleCategoria;
	 bool teclaSalir = false;
	 int indiceImagen = 0;
    char tecla = 'l';
    int M = numRandomWindows;
    int currentNumberOfDetections = 0;
    int numberOfOptimizedDetection = 0;
	/* int rowsRows = almacenImagenes[1].rows;
	int colsCols = almacenImagenes[1].cols;
	cout << "numero de columnas = " << colsCols << " numero de filas = " << rowsRows << endl; */
	//string videoName = RUTAVIDEOSALIDA;
	// if (mixture){
	// 	videoName += "mixture" + to_string(numRandomWindows) + to_string(currentTestNumber)+ ".avi";
	// }else{
	// 	videoName += "random" + to_string(numRandomWindows) + to_string(currentTestNumber)+ ".avi";
	// }
	//string videoName = RUTAVIDEOSALIDA + to_string(numRandomWindows) + to_string(currentTestNumber)+ ".avi";
	// VideoWriter video (videoName, CV_FOURCC('X','V','I','D'),1, Size(colsCols,rowsRows));
	//VideoWriter video (videoName, CV_FOURCC('X','V','I','D'),15, Size(NUMCOLS,NUMROWS));
    
		while ((!teclaSalir)&&(indiceImagen < NUMERODEFICHEROS)){
		    	//medicion del tiempo
		    	auto start = std::chrono::steady_clock::now();
		//         tecla = waitKey(0); // Wait for a keystroke in the window
		        //std::cout<<"key="<<tecla<<endl;
		        if (tecla == 's') {
		            teclaSalir = true;
		            //std::cout<<"Pulse intro para salir..."<<endl;
		
		        }else if(tecla == 'l'){
		            //Actalizamos el array de detecciones
		            //cout<< "Actualizamos las detecciones... " << endl;
		            updateDetections(A,alpha,delta,currentNumberOfDetections, numberOfOptimizedDetection, almacenImagenes[indiceImagen],  anomalousCategories, detectionThreshold);
		            //cout<< "Detecciones actualizadas. " << endl;
					std::cout << "current progress: " << (progressCounter*100)/totalExecutions << " %" <<endl;
		
		                int i = 0;
		                int numeroBrutoDeteccionesGeneradas = 0;
		                while (i< M){
		                    //cout<< "Generando posible deteccion... " << endl;
		
		                    Detection possibleDetection = generatePossibleDetection(A, sigma, q, almacenImagenes[indiceImagen], mixture, currentNumberOfDetections, probabilityDistribution);
		                    int superficie = possibleDetection.x3*possibleDetection.x3;
		
		                    //if((possibleDetection.x1 < NUMCOLS) && (possibleDetection.x1 > 0) && (possibleDetection.x2 < NUMROWS) &&(possibleDetection.x2 >0) && ((possibleDetection.x1 + possibleDetection.x3) < NUMCOLS)  &&  ((possibleDetection.x2 + possibleDetection.x3)<NUMROWS)){
		                    	if(!possibleDetection.badDetection){
		                        //if (superficie>=MINIMUMFRAME){
		                       	if ((superficie>=MINIMUMFRAME) && (superficie <= MAXIMUMFRAME) && (matchesExistent(A, possibleDetection, currentNumberOfDetections)==false)){
								//if ((superficie>=MINIMUMFRAME) && (superficie <= MAXIMUMFRAME)){

		                           // cout<< "Posible deteccion generada. " << "x1= " << possibleDetection.x1 << " x2= " << possibleDetection.x2 << " x3= " << possibleDetection.x3 << endl;
		                           // cout << "Superficie = " << superficie << " pixeles" << endl;
		
		                            //addDetectionAA(A, possibleDetection);
		//*****************************************************************************Trozo de codigo destinado a recortar la imagen y pasarla  por la red neuronal***************************
							
										Mat imagenExtraida = cv::Mat::zeros(Size(possibleDetection.ventana.tamanoHorizontal,possibleDetection.ventana.tamanoVertical), almacenImagenes[indiceImagen].type());
		    							almacenImagenes[indiceImagen](cv::Rect(possibleDetection.ventana.posX,possibleDetection.ventana.posY, possibleDetection.ventana.tamanoHorizontal,possibleDetection.ventana.tamanoVertical)).copyTo(imagenExtraida(cv::Rect(0,0,possibleDetection.ventana.tamanoHorizontal,possibleDetection.ventana.tamanoVertical)));					
										
										possibleDetection.ventana.label = "";
										possibleDetection.ventana.R = 77;
										possibleDetection.ventana.G = 166;
										possibleDetection.ventana.B = 255;

										possibleDetection.ventana.dibujarVentana(almacenImagenes[indiceImagen]);
										//cout << "Despues de almacenImagenes" << endl;
																		
										//auto input = tutorialHelpers::PrepareImageForModel(imagenExtraida, inputShape.columns, inputShape.rows);
										//cout << "Despues de input" << endl;
		        						// Send the image to the compiled model and fill the predictions vector with scores, measure how long it takes
		        						
		        						//model_predict(input, predictions);
		        						
		        						// Get the value of the top 5 predictions
		        						//auto top5 = tutorialHelpers::GetTopN(predictions, 5);
		        						
		        						
										tutorial.PrepareNextImage(imagenExtraida);
		        						
		        						//model_Predict(input, predictions);
		        						tutorial.Step();
		        						
		        						// Get the value of the top 5 predictions
		        						auto top5 = tutorial.GetTopN(5);		        						
		        						
		        						
		        						
										//Test whether the category is some of our anomalous categories
										
										int indiceTop5 = 0;
										bool encontrado = false;
										while ((!encontrado)&&(indiceTop5 < top5.size())){
											if(anomalousCategories[top5.at(indiceTop5).first].compare("Empty")!=0){
												encontrado = true;									
											
											}else{
											
												indiceTop5++;
											}								
										}
										
											//if (indiceTop5<top5.size()){ //Al menos uno esta en la lista de anomalos y merece la pena sacar la ventana
											if ((encontrado)&&(((top5.at(indiceTop5).second) *100.0 ) >= detectionThreshold)){ //Al menos uno esta en la lista de anomalos y merece la pena sacar la ventana
											//We write down the new element found in the registry of found elements:
												anomalousObjectsFound[top5.at(indiceTop5).first]++;
												// Generate header text that represents the top5 predictions
		        								std::stringstream headerText;
		        								float accuracy =0;
		        								for (auto element : top5)
		        									{
		        											//cout << "Antes de headerText" << categories[1] << endl;
		        											//cout << "categories[element.first] = " << categories[element.first] << endl;
		        											if (floor(element.second *100.0)>=detectionThreshold){
		        												if (floor(element.second *100.0)>=accuracy){
		        													accuracy = floor(element.second *100.0);
		        												}
		            										headerText << "(" << std::floor(element.second * 100.0) << "%) " << categories[element.first] << "  ";
		            									}
		            															
		        									}
		        									//cout << "Antes de posibleCategoria" << endl;
		        									posibleCategoria = headerText.str();
		        									//cout<< "posibleCategoria = " << posibleCategoria << endl; 
													//posibleCategoria = passSampleThroughCNN(possibleDetection, almacenImagenes[indiceImagen], inputShape.columns, inputShape.rows, categories);
		//************************	**************************************************************************************************************************************************************
		                        		if (posibleCategoria != ""){
		                        			//cout << "Deteccion anadida " << endl;
		                        			 possibleDetection.ventana.label = posibleCategoria;
		                        			 if ((accuracy>20.0)&&(accuracy<=40.0)){
		                        			 	possibleDetection.ventana.R = 255;
		                        			 	possibleDetection.ventana.G = 0;
		                        			 	possibleDetection.ventana.B = 0;
		                        			 	
		                        			 }else if((accuracy>40.0)&&(accuracy<=70.0)){
		                        			 	possibleDetection.ventana.R = 255;
		                        			 	possibleDetection.ventana.G = 255;
		                        			 	possibleDetection.ventana.B = 0;
		                        			 	
		                        			 }else{
		                        			 	possibleDetection.ventana.R = 0;
		                        			 	possibleDetection.ventana.G = 255;
		                        			 	possibleDetection.ventana.B = 0;
		                        			 	
		                        			 }
		                          		  addDetectionAA(A, possibleDetection, currentNumberOfDetections);
		                        		}
		                          		
		                       		}
		                       		i++;
		                        }else{
		                        	if ((superficie<MINIMUMFRAME)|| (superficie > MAXIMUMFRAME)){
		                            //cout << "Posible deteccion fallida. Superficie = " << possibleDetection.x3*possibleDetection.x3 << " pixeles" << endl;
		                         }else{
											 // cout << "Posible deteccion fallida. Superposicion" << endl;                       
		                         }
		                        }
		                    }
		                    numeroBrutoDeteccionesGeneradas++;
		                     
		                }
		
								//std::cout << "El numero de detecciones generadas es = " << numeroBrutoDeteccionesGeneradas << endl;
		                /*string cadenaDetection = "Detection ";
		                string cadenaEncabezado;
		                for(int i = 0;i<MAXDETECTIONS;i++){
		                    cadenaEncabezado = cadenaDetection + to_string(i);
		                    if (A[i].active){
		                        //cout << "Nuevo recorte generado: " << "posx=" << A[i].ventana.posX << " posy=" << A[i].ventana.posY << endl;
		                        A[i].ventana.extraerRecorte(almacenImagenes[indiceImagen]);
		
		                        namedWindow( cadenaEncabezado, WINDOW_AUTOSIZE ); // Create a window for display.
		                        imshow( cadenaEncabezado, A[i].ventana.image);
		                    }else{
		                        //cout << "deteccion no activa" << endl;
		                        namedWindow( cadenaEncabezado, WINDOW_AUTOSIZE ); // Create a window for display.
		                        imshow( cadenaEncabezado, noImage);
		                    }
		                }*/
		
		                for(int i = 0;i<currentNumberOfDetections;i++){
		                    //if (A[i].active){
		                        //cout << "Nueva ventana generada: " << "posx=" << A[i].ventana.posX << " posy=" << A[i].ventana.posY << endl;

		                        A[i].ventana.dibujarVentana(almacenImagenes[indiceImagen]);
		                    //}
		                }
		
		                //std::cout << "***********Informacion de las detecciones****************** " << endl;
		                for(int i = 0; i<currentNumberOfDetections; i++){
		                    //if (A[i].active){
		                        //std::cout<< "Deteccion " << i << ": " << " x1=" << A[i].x1 << " x2= " << A[i].x2 << " x3= " << A[i].x3 << "Pi= " << A[i].pi << " Label = " << A[i].ventana.label <<endl;
		                    //}
		                }
		                //std::cout<<"**************************************************************" << endl;
		
							if (displayResultsWindow){
								namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
								imshow( "Display window", almacenImagenes[indiceImagen] );
								
		                 	}
							//video.write(almacenImagenes[indiceImagen]);
		                }else{
		        //            v1.dibujarVentana(imageBuffer);
		        //            v2.dibujarVentana(imageBuffer);
		                    //v1.dibujarVentana(almacenImagenes[indiceImagen]);
		                    //v2.dibujarVentana(almacenImagenes[indiceImagen]);
		                    
		                    
		                    if (displayResultsWindow){
			                    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
			                    imshow( "Display window", almacenImagenes[indiceImagen] );                // Show our image inside it.
								
		                    }
							//video.write(almacenImagenes[indiceImagen]);
		
		
		                }
		                //if (indiceImagen < NUMERODEFICHEROS-1){
		                    indiceImagen++;
		                    //std::cout << "el numero de la imagen es " << indiceImagen << endl;
		                //}
		                //imshow( "Display window", almacenImagenes[indiceImagen] );                // Show our image inside it.
		               // waitKey();
		                tecla = waitKey(1); // Wait for a keystroke in the window
		                tecla = 'l';
		                
		                    //i++;
		                auto end = std::chrono::steady_clock::now();
		                
		                meanTimeToPredict = std::floor(tutorialHelpers::GetMeanDuration(predictionTimes, std::chrono::duration<double>(end - start).count()) * 1000);
		        			//std::cout << meanTimeToPredict << "ms/frame -> " << 1000/meanTimeToPredict << " frames per second " << endl;
							numberOfOptimizedDetection = (numberOfOptimizedDetection + 1) % MAXDETECTIONS;
		            }
					//video.release();
		            //std::cout << meanTimeToPredict << "ms/frame -> " << 1000/meanTimeToPredict << " frames per second"<< endl;    
		            
		            frameProcessingTime = 1000.0/meanTimeToPredict;

    
    
    


}

array<Mat, NUMERODEFICHEROS> copyArrayframes(array<Mat, NUMERODEFICHEROS> & almacenImagenes){
	array<Mat, NUMERODEFICHEROS> resultado;
	for (int i = 0; i<NUMERODEFICHEROS; i++){
			resultado[i] = almacenImagenes[i].clone();
	}
	return resultado;
}

void initializeDetections(array<Detection,MAXDETECTIONS> & A){
	for (int i = 0; i<MAXDETECTIONS; i++){
		A[i].x1 = 0;
		A[i].x2 = 0;
		A[i].x3 = 0;
		A[i].pi = 0;
		A[i].badDetection = true;
		A[i].active = false;
	
	}
}

array<int,3> secondsToHours(int sec){
	array<int,3>duration;
	int hours = sec/3600;
	int minutes = (sec%3600)/60;
	int seconds = (sec%3600)%60;
	duration[0] = hours;
	duration[1] = minutes;
	duration[2]= seconds;
	return duration;
}


void plotResults(string mixtureFileName, string randomFileName){
	float arrayMixture[MAXWINDOWSFORTEST][NUMCOLSRECORDFILE];
    float arrayRandom[MAXWINDOWSFORTEST][NUMCOLSRECORDFILE];

    std::string line;
    ifstream fileMixture (mixtureFileName);
    ifstream fileRandom (randomFileName);

    //std::cout << "reading mixture..." << endl;
    if (fileMixture.is_open()){
        //std::cout << "ficheroAbierto" <<  endl;
        int row = 0;
        while (getline (fileMixture,line))
        {
            array<std::string,NUMCOLSRECORDFILE> arrayOfTokens =  splitString(line);
            for (int i = 0; i<NUMCOLSRECORDFILE; i++){
                arrayMixture[row][i] = std::stof(arrayOfTokens[i]);
            }
            row++;
        }
    }
    
    //std::cout << "reading random..." << endl;
    if (fileRandom.is_open()){
        int row = 0;
        while (getline (fileRandom,line))
        {
            array<std::string,NUMCOLSRECORDFILE> arrayOfTokens =  splitString(line);
            for (int i = 0; i<NUMCOLSRECORDFILE; i++){
                arrayRandom[row][i] = std::stof(arrayOfTokens[i]);
            }
            row++;
        }
    }
    fileRandom.close();

    string ficheroSalidaMixture = "ficheroSalidaMixture.dat";
   
    ofstream outputFileMixture;
	outputFileMixture.open(ficheroSalidaMixture);
    for (int i = 0; i<MAXWINDOWSFORTEST;i++){
        outputFileMixture << i+1 << " " << arrayMixture[i][0] << endl;
    }
    
    outputFileMixture.close();

    string ficheroSalidaRandom = "ficheroSalidaRandom.dat";

    ofstream outputFileRandom;
	outputFileRandom.open(ficheroSalidaRandom);
    for (int i = 0; i<MAXWINDOWSFORTEST;i++){
        outputFileRandom << i+1 << " " << arrayRandom[i][0] << endl;
    }
    outputFileRandom.close();


	GnuplotPipe gp;
    gp.sendLine("set title 'Comparative results'");
    gp.sendLine("set xlabel 'Mumber of potential detections'");
    gp.sendLine("set ylabel 'Mean number of objects detected per frame'");
   // gp.sendLine("plot [-pi/2:pi] cos(x),-(sin(x) > sin(x+1) ? sin(x) : sin(x+1))");
   string stringFileAux = mixtureFileName.erase(mixtureFileName.find("mixture"),7);
   stringFileAux = stringFileAux.erase(stringFileAux.find(".txt"),4);
   string lineFileName = "set terminal png size 800,600; set output '" + stringFileAux + "'; plot 'ficheroSalidaMixture.dat' w lp title 'Mixture', 'ficheroSalidaRandom.dat' w lp title 'Random'";
   gp.sendLine(lineFileName);

}





int main(int argc, char *argv[])
{
	
	auto comienzo = std::chrono::steady_clock::now();
    Mat noImage = imread("/home/icai23/noImage.png", IMREAD_COLOR);
    Mat waterBottle = imread("/home/icai23/waterBottle3.png", IMREAD_COLOR);
	    Mat wrench = imread("/home/icai23/wrench.png", IMREAD_COLOR);
    Mat dog = imread("/home/icai23/dog.png", IMREAD_COLOR);
    Mat bird = imread("/home/icai23/eagle.png", IMREAD_COLOR);
    Mat bag = imread("/home/icai23/bag.png", IMREAD_COLOR);
    Mat horse = imread("/home/icai23/horse.png", IMREAD_COLOR);
    Mat clock = imread("/home/icai23/clock.png", IMREAD_COLOR);
    Mat gasTank = imread("/home/icai23/gasTank.png", IMREAD_COLOR);
    Mat pliers = imread("/home/icai23/pliers.jpg", IMREAD_COLOR);
    Mat motorbike = imread("/home/icai23/daytona.jpg", IMREAD_COLOR);
    Mat bull = imread("/home/icai23/bull.jpg", IMREAD_COLOR);
    //bool gaussian = false;
    //std::cout << "comenzando programa pez" << endl;
	
    srand(time(NULL));
    //alpha value
    
    float alpha = 0.7;
    float delta = 5;
    //Initialize set of active detections
    //array<Detection,MAXDETECTIONS> A;
    //array<Mat, MAXDETECTIONS> recortes;
    //int currentNumberOfDetections = 0;
    int M = NUMRANDOMWINDOWS;
    //set<Detection> A;
    bool teclaSalir = false;
    //string rutaBaseFicheros = "/home/icai23/barrido0Grados/imagen";
    //string rutaBaseFicheros = "/home/icai23/barridoTestPositivoBottle/imagen";
        //string rutaBaseFicheros = "/home/icai23/barridosIndividuales/barrido360Bag/imagen";
        string rutaBaseFicheros = FILEBASEPATH;
    
        // array<Mat, NUMERODEFICHEROS> almacenImagenesNoTocar = loadFrames(rutaBaseFicheros);
     

    int indiceImagen = 0;
    char tecla = 'l';
    
    bool displayResultsWindow = DISPLAYRESULTSWINDOW;
    
//************************Tutorial code *************************************************
    
 // Read the category names
    auto categories = ReadLinesFromFile(NAMECATEGORIESFILE); 
    // Read the anomalous category names
    auto anomalousC = ReadLinesFromFile(NAMEANOMALOUSCATEGORIESFILE); 
	 auto anomalousCategories = orderAnomalousCategories(categories,anomalousC);
	 //std::array<int, 1000> anomalousObjectsFound;
	 //initializeAnomalousObjectsFound(anomalousObjectsFound);
	 double frameProcessingTime;
	 double cummulativeFrameProcessingTime;
	 double meanFrameProcessingTime;
	 int numberOfAnomalousObjectsFound;
	int  numberOfDifferentAnomalousObjectsFound;
	 float percentageOfAnomalousObjectsFound;
	 float meanNumberOfAnomalousObjectsFound;
	 float meanPercentageOfAnomalousObjectsFound;
	 int cummulativeNumberOfAnomalousObjectsFound;
	 float cummulativePercentageOfAnomalousObjectsFound;
	 string recordFileBaseName = RECORDFILEBASENAME;
	 string probabilityDistribution = PROBABILITYDISTRIBUTION;
	 int maxWindowsForTest = MAXWINDOWSFORTEST;
	 //int detectionThreshold = 90;
	 bool mixture;
	 int maxNumberOfAnomalousObjectsFound = 0;
	 int minNumberOfAnomalousObjectsFound = 50;
	 float maxPercentageOfAnomalousObjectsFound = 0.0;
	 float minPercentageOfAnomalousObjectsFound = 120.0;
	 float sigma = 0.3;
	 float q = 0.7;
	 //std::default_random_engine generator;
	 std::array<std::string,3> arrayDistributions = {"GAUSSIAN", "STUDENT", "TRIANGLE"};
	// std::array<std::string,6> arrayRutaInputs = {"../frames1Objetos/frame-", "../frames2Objetos/frame-", "../frames3Objetos/frame-", "../frames5Objetos/frame-", "../frames7Objetos/frame-", "../frames10Objetos/frame-"};
	// std::array<std::string,6> arrayRutaOutputs = {"../records/V1F/" , "../records/V2F/", "../records/V3F/", "../records/V5F/", "../records/V7F/", "../records/V10F/"};
	//  std::array<std::string,1> arrayRutaInputs = {"../frames7Objetos/frame-"};
	//  std::array<std::string,1> arrayRutaOutputs = {"../records/V7F/"};
	//std::array<std::string,1> arrayRutaInputs = {"../frames10Objetos/frame-"};
	//std::array<std::string,1> arrayRutaOutputs = {"../records/V10F/"};
	 std::array<std::string,1> arrayRutaInputs = {"../frames5Objetos/frame-"};
	 std::array<std::string,1> arrayRutaOutputs = {"../records/V5F/"};
	
	 int progressCounter = 0;
	 int totalExecutions = arrayRutaInputs.size()*( 4 *(MAXWINDOWSFORTEST * NUMTESTS));
	 bool executeRandom = true;
	

	
	for (int superObjectsIndex = 0; superObjectsIndex < arrayRutaInputs.size(); superObjectsIndex++){	
		string rutaBaseFicheros = 	arrayRutaInputs	[superObjectsIndex];
		array<Mat, NUMERODEFICHEROS> almacenImagenesNoTocar = loadFrames(rutaBaseFicheros);
		cout << rutaBaseFicheros << endl;
		
		recordFileBaseName = arrayRutaOutputs[superObjectsIndex];
		cout << recordFileBaseName << endl;
		executeRandom = true;

		for (int superIndex = 0; superIndex< 3; superIndex++){
			//std::cout << "Ejecucion del bucle*****************************************************" << std::endl;
			probabilityDistribution = arrayDistributions[superIndex];
		
		
			// Get the model's input shape. We will use this information later to resize images appropriately.
			TensorShape inputShape;
			model_GetInputShape(0, &inputShape);   
			
			// Create a vector to hold the model's output predictions
			std::vector<float> predictions(model_GetOutputSize());
			//std::cout << "Trabajo para ICAE2019" << endl;
			//getchar();
			//for (int sigmaIndex = 0; sigmaIndex <3; sigmaIndex ++){
				//q = 0.1;
				//for (int qIndex = 0; qIndex <3; qIndex++ ){
					for(int detectionThreshold = 20; detectionThreshold>10; detectionThreshold = detectionThreshold-10){
						
						string randomOutputFile;

						if (executeRandom){
							randomOutputFile = recordFileBaseName + "random" + "_" + probabilityDistribution + "_" + std::to_string(detectionThreshold) + "_" + std::to_string(maxWindowsForTest) + "_" + to_string(NUMERODEFICHEROS) + "_" + to_string(DETECTIONMAXSIZE) + "_" + to_string(MINIMUMFRAME) + "_" + to_string(MAXIMUMFRAME) + "_" +to_string(EUCLIDEANDISTANCETHRESHOLD) + "_" + to_string(sigma) + "_" + to_string(q) + ".txt";
						}else{
							randomOutputFile = recordFileBaseName + "random" + "_" + "GAUSSIAN" + "_" + std::to_string(detectionThreshold) + "_" + std::to_string(maxWindowsForTest) + "_" + to_string(NUMERODEFICHEROS) + "_" + to_string(DETECTIONMAXSIZE) + "_" + to_string(MINIMUMFRAME) + "_" + to_string(MAXIMUMFRAME) + "_" +to_string(EUCLIDEANDISTANCETHRESHOLD) + "_" + to_string(sigma) + "_" + to_string(q) + ".txt";
						}

						string mixtureOutputFile;
						mixtureOutputFile = recordFileBaseName + "mixture" + "_" + probabilityDistribution + "_" + std::to_string(detectionThreshold) + "_" + std::to_string(maxWindowsForTest) + "_" + to_string(NUMERODEFICHEROS) + "_" + to_string(DETECTIONMAXSIZE) + "_" + to_string(MINIMUMFRAME) + "_" + to_string(MAXIMUMFRAME) + "_" + to_string(EUCLIDEANDISTANCETHRESHOLD) + "_" + to_string(sigma) + "_" + to_string(q) + ".txt";
								
						for(int indiceCaso = 0;indiceCaso <2; indiceCaso++){ 
						if(indiceCaso == 0){
								mixture = true; 
								
										
								ofstream outputFile;
								outputFile.open(mixtureOutputFile);
											
								for(int i = 1; i<=maxWindowsForTest; i++){
									
									frameProcessingTime = 0.0;
									cummulativeFrameProcessingTime = 0.0;
									meanFrameProcessingTime = 0.0;
									
									numberOfAnomalousObjectsFound = 0.0;
									cummulativeNumberOfAnomalousObjectsFound = 0.0;
									meanNumberOfAnomalousObjectsFound = 0.0;
									
									percentageOfAnomalousObjectsFound = 0.0;
									cummulativePercentageOfAnomalousObjectsFound = 0.0;
									meanPercentageOfAnomalousObjectsFound = 0.0;
									
									maxNumberOfAnomalousObjectsFound = 0;
									minNumberOfAnomalousObjectsFound = 50;
									maxPercentageOfAnomalousObjectsFound = 0.0;
									minPercentageOfAnomalousObjectsFound = 120.0;
												
									
									for(int j=1; j<=NUMTESTS; j++){
										array<Mat, NUMERODEFICHEROS> almacenImagenes = copyArrayframes(almacenImagenesNoTocar);
										array<Detection,MAXDETECTIONS> A;
										initializeDetections(A);
										//getchar();
										std::array<int, 1000> anomalousObjectsFound;
										initializeAnomalousObjectsFound(anomalousObjectsFound);
										
										pruebasConNumeroVentanas(i,j,alpha, delta, sigma, q,  inputShape,predictions, A, almacenImagenes,anomalousCategories, categories, anomalousObjectsFound, frameProcessingTime, displayResultsWindow, detectionThreshold, mixture, probabilityDistribution, progressCounter, totalExecutions);
										cummulativeFrameProcessingTime += frameProcessingTime;
							
										meanFrameProcessingTime = cummulativeFrameProcessingTime/j;	
										//std::cout << "Mean frame processing time = " << meanFrameProcessingTime << " fps." << endl;
								
										
										numberOfAnomalousObjectsFound = calculateNumberOfAnomalousObjectsFound(anomalousObjectsFound);
										numberOfDifferentAnomalousObjectsFound = calculateNumberOfDifferentAnomalousObjectsFound(anomalousObjectsFound);
										if (numberOfAnomalousObjectsFound>maxNumberOfAnomalousObjectsFound)	{
											maxNumberOfAnomalousObjectsFound = numberOfAnomalousObjectsFound;
										}		
										
										if (numberOfAnomalousObjectsFound<minNumberOfAnomalousObjectsFound)	{
											minNumberOfAnomalousObjectsFound = numberOfAnomalousObjectsFound;
										}
										
										cummulativeNumberOfAnomalousObjectsFound += numberOfAnomalousObjectsFound;
										meanNumberOfAnomalousObjectsFound = cummulativeNumberOfAnomalousObjectsFound/j;
										
													//std::cout << "number of anomalous objects found " << numberOfAnomalousObjectsFound << endl;
										
										//std::cout << "Mean number of anomalous objects found " << meanNumberOfAnomalousObjectsFound << endl;
										
										percentageOfAnomalousObjectsFound = calculatePercentageOfAnomalousObjectsFound(numberOfAnomalousObjectsFound);
										if (percentageOfAnomalousObjectsFound>maxPercentageOfAnomalousObjectsFound)	{
										maxPercentageOfAnomalousObjectsFound = percentageOfAnomalousObjectsFound;
										}		
										
										if (percentageOfAnomalousObjectsFound<minPercentageOfAnomalousObjectsFound)	{
											minPercentageOfAnomalousObjectsFound = percentageOfAnomalousObjectsFound;
										}	
																			
										cummulativePercentageOfAnomalousObjectsFound += percentageOfAnomalousObjectsFound;
										meanPercentageOfAnomalousObjectsFound = cummulativePercentageOfAnomalousObjectsFound/j;
										
										//std::cout << "Mean percentage of anomalous objects found " << meanPercentageOfAnomalousObjectsFound << endl;
										//getchar();
										progressCounter++;
							
									}
									outputFile << meanNumberOfAnomalousObjectsFound << " " << maxNumberOfAnomalousObjectsFound <<  " "  << minNumberOfAnomalousObjectsFound << " " << meanPercentageOfAnomalousObjectsFound << " " << maxPercentageOfAnomalousObjectsFound << " " << minPercentageOfAnomalousObjectsFound << " " << meanFrameProcessingTime << " " << numberOfDifferentAnomalousObjectsFound << endl;
								}
								outputFile.close();				
								}else{
									if (executeRandom){
										executeRandom = false;	
										mixture = false;  
															
										ofstream outputFile;
										outputFile.open(randomOutputFile);
											
										for(int i = 1; i<=maxWindowsForTest; i++){
											
											frameProcessingTime = 0.0;
											cummulativeFrameProcessingTime = 0.0;
											meanFrameProcessingTime = 0.0;
											
											numberOfAnomalousObjectsFound = 0.0;
											cummulativeNumberOfAnomalousObjectsFound = 0.0;
											meanNumberOfAnomalousObjectsFound = 0.0;
											
											percentageOfAnomalousObjectsFound = 0.0;
											cummulativePercentageOfAnomalousObjectsFound = 0.0;
											meanPercentageOfAnomalousObjectsFound = 0.0;
											
											maxNumberOfAnomalousObjectsFound = 0;
											minNumberOfAnomalousObjectsFound = 50;
											maxPercentageOfAnomalousObjectsFound = 0.0;
											minPercentageOfAnomalousObjectsFound = 120.0;
											
											for(int j=1; j<=NUMTESTS; j++){
												array<Mat, NUMERODEFICHEROS> almacenImagenes = copyArrayframes(almacenImagenesNoTocar);
												array<Detection,MAXDETECTIONS> A;
												initializeDetections(A);
												//getchar();
												std::array<int, 1000> anomalousObjectsFound;
												initializeAnomalousObjectsFound(anomalousObjectsFound);
												
												pruebasConNumeroVentanas(i,j,alpha, delta, sigma, q, inputShape,predictions, A, almacenImagenes,anomalousCategories, categories, anomalousObjectsFound, frameProcessingTime, displayResultsWindow, detectionThreshold, mixture,probabilityDistribution, progressCounter, totalExecutions);
												cummulativeFrameProcessingTime += frameProcessingTime;
									
												meanFrameProcessingTime = cummulativeFrameProcessingTime/j;	
												//std::cout << "Mean frame processing time = " << meanFrameProcessingTime << " fps." << endl;
										
												
												numberOfAnomalousObjectsFound = calculateNumberOfAnomalousObjectsFound(anomalousObjectsFound);
												numberOfDifferentAnomalousObjectsFound = calculateNumberOfDifferentAnomalousObjectsFound(anomalousObjectsFound);

												if (numberOfAnomalousObjectsFound>maxNumberOfAnomalousObjectsFound)	{
													maxNumberOfAnomalousObjectsFound = numberOfAnomalousObjectsFound;
												}		
												
												if (numberOfAnomalousObjectsFound<minNumberOfAnomalousObjectsFound)	{
													minNumberOfAnomalousObjectsFound = numberOfAnomalousObjectsFound;
												}
												
												cummulativeNumberOfAnomalousObjectsFound += numberOfAnomalousObjectsFound;
												meanNumberOfAnomalousObjectsFound = cummulativeNumberOfAnomalousObjectsFound/j;
												
															//std::cout << "number of anomalous objects found " << numberOfAnomalousObjectsFound << endl;
												
												//std::cout << "Mean number of anomalous objects found " << meanNumberOfAnomalousObjectsFound << endl;
												
												percentageOfAnomalousObjectsFound = calculatePercentageOfAnomalousObjectsFound(numberOfAnomalousObjectsFound);
												cummulativePercentageOfAnomalousObjectsFound += percentageOfAnomalousObjectsFound;
												meanPercentageOfAnomalousObjectsFound = cummulativePercentageOfAnomalousObjectsFound/j;
												
												//std::cout << "Mean percentage of anomalous objects found " << meanPercentageOfAnomalousObjectsFound << endl;
												//getchar();
												progressCounter++;
								
									
											}
											outputFile << meanNumberOfAnomalousObjectsFound << " " << maxNumberOfAnomalousObjectsFound <<  " "  << minNumberOfAnomalousObjectsFound << " " << meanPercentageOfAnomalousObjectsFound << " " << maxPercentageOfAnomalousObjectsFound << " " << minPercentageOfAnomalousObjectsFound << " " << meanFrameProcessingTime << " " << numberOfDifferentAnomalousObjectsFound << endl;
										}
										outputFile.close();	
								}			
							}
						}
						plotResults(mixtureOutputFile, randomOutputFile);
					}
			//		q +=0.4;
			//	}
			//	sigma +=0.4;
			//}
		}
	}
	auto final = std::chrono::steady_clock::now();
	

	std::chrono::duration<double> diff = final-comienzo;
	
	//cout << "El tiempo que tarda el programa en ejecutarse es de " << diff.count() << " segundos. Es decir, " << (diff.count())/60 << " minutos. "  << endl;
	array<int,3> duration = secondsToHours(diff.count());
	std::cout << "Time elapsed is " << duration[0] << " hours " << duration[1] << " minutes and " << duration[2] << " seconds." << endl;
	
	
}



