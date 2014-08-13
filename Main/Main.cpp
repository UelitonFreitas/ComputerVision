
#include "math.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <time.h>
#include <sys/stat.h>
#include <pthread.h>
#include <errno.h>

#include "../BoW/BoW.h"
#include "../SVM/SVM.h"
#include "../ColorHistogram/ColorHistogram.h"

using namespace cv;
using namespace std;

string          TrainFolder             = "TrainImages";
string          TestFolder              = "TestImages";
string          TrainImages             = "TrainImages.txt";
string          fileWithClassesNames    = "Classes.txt";
vector<string>  classSet;

string  detectorType        = "SURF";
string  descriptorType      = "SURF";
string  matcherType         = "FlannBased";
float   hessianThreshold    = 0.1;
int     dictionarySize      = 16;

string getImageClass(string imagePath);
int getClassNumber(String fileImageName);

void readImage(vector<string>& images,string folder, string className);
void readImagesPaths(vector<string>& imagesPaths,vector<string>& imagesClasses, bool isTest);
void printStringVector(vector<string>& vector);
void loadTrainImages(vector<Mat>& grayTrainImages, vector<Mat>& colorTrainImages,vector<string>& imagesClasses);
void loadTestImages(vector<Mat>& grayTestImages, vector<Mat>& colorTestImages);
bool createDetectorDescriptorMatcher(Ptr<FeatureDetector>& featureDetector,Ptr<DescriptorExtractor>& descriptorExtractor,Ptr<DescriptorMatcher>& descriptorMatcher );


enum Fish{
  Dourado,Oscar,Tucunare  
};

int main(int argc, char** argv){
    
    BoW bow(detectorType,descriptorType,matcherType,hessianThreshold,dictionarySize);
    
    ColorHistogram ch(8,"HSV",false,false,true);
    
    //Train Images
    vector<Mat>                 grayTrainImages;
    vector<Mat>                 colorTrainImages;
    //Test Images
    vector<Mat>                 grayTestImages;
    vector<Mat>                 colorTestImages;
    
    vector<string>              imagesClasses;

    
    loadTrainImages(grayTrainImages,colorTrainImages,imagesClasses);
    
    //ch.createHistograms(colorTrainImages,imagesClasses);
    //vector<vector<float> > imagesHist = ch.getHistograms();
    
    //BOW
    
      bow.loadTrainImages(grayTrainImages,imagesClasses);
      bow.runTraining();
    
    /*bow.trainFeaturesDetect();
    bow.trainKeyPointsDescriptors();
    bow.createVocabulary();
    bow.setVocabularyOnImageDescriptor();
    bow.saveDictionary();
    bow.createImagesAttributes();
    */
    //bow.loadDictionary("Dictionary-16.xml");
    
    
    //test
    //bow.createImageAttribute(grayTrainImages[8]);
    
    //Mat testImage = bow.getImagesAttributesOfTestImage();
    
    
    //SVM bow
    
    SVMClass svm(classSet);
    svm.train(bow.getImagesAttributes(),imagesClasses);
    svm.saveModel("BoWsvmModel.xml");
    
    
    //SVM hist
    
      SVMClass svmh(classSet);
      //svmh.train(imagesHist,imagesClasses);
      //svmh.saveModel("HsvmModel.xml");
      //svmh.loadModel("HsvmModel.xml");
      
      //vector<float> hf = ch.createHistogram(colorTrainImages[2],false,false,true,"HSV");
      
      //cout << "class: " << svmh.predict(hf) << endl;

      
          

    //cout << endl << "Class: " << svm2.predict(testImage) << endl;
    
    //loadTestImages(grayTestImages,colorTestImages);
    //bow.loadTestImages(grayTestImages);
    
    //bow.createImageAttribute(grayTestImages.at(0),"lol");
    //drawKeypoints( grayTestImages[0], keyPoints, grayTestImages[0], Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //imshow("Key Points", grayTestImages[0]);
    
    //bow.loadTestImages(grayTestImages);
    //featureDetector->detect(grayTrainImages,trainKeyPoints);
    
   
    
    //vector<vector<KeyPoint> > kp = bow.getTrainKeyPoints();
    
    //for (size_t i = 0; i < colorTrainImages.size(); i++){
    //    //drawKeypoints( colorTrainImages[i], kp[i], colorTrainImages[i], Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
   //     imshow("Key Points", grayTrainImages[i]);
    //    waitKey(0);
    //}

    
}




void loadTestImages(vector<Mat>& grayTestImages, vector<Mat>& colorTestImages,vector<string>& imagesClasses){
    
    vector<string> imagesPaths;
    readImagesPaths(imagesPaths,imagesClasses,true);
    
    cout << endl << "Loading test images...." << endl;
    
    for(size_t i = 0; i < imagesPaths.size(); i++){
        
        string path = imagesPaths[i];
        Mat grayImg = imread(path,CV_LOAD_IMAGE_GRAYSCALE);
        Mat colorImg  = imread(path,CV_LOAD_IMAGE_COLOR);
        
        if(grayImg.empty() || colorImg.empty())
            cout << "Test image " << path.c_str() << "can not be read.";
        else{
            grayTestImages.push_back(grayImg);
            colorTestImages.push_back(colorImg);
        }
    }
    
    cout << "Load complete!." << endl;
    
    
}

void loadTrainImages(vector<Mat>& grayTrainImages, vector<Mat>& colorTrainImages,vector<string>& imagesClasses){
    
    vector<string> imagesPaths;
    vector<string> imagesClass;
    readImagesPaths(imagesPaths,imagesClasses,false);
    
    cout << endl << "Loading train images...." << endl;
    
    for(size_t i = 0; i < imagesPaths.size(); i++){
        
        string path         = imagesPaths[i];
        Mat grayImg         = imread(path,CV_LOAD_IMAGE_GRAYSCALE);
        Mat colorImg        = imread(path,CV_LOAD_IMAGE_COLOR);
        
        if(grayImg.empty() || colorImg.empty())
            cout << "Train image " << path.c_str() << "can not be read.";
        else{
            cout << path << " loaded!!" << endl;
            grayTrainImages.push_back(grayImg);
            colorTrainImages.push_back(colorImg);
        }
    }
    
    cout << "Load complete!." << endl;
    
}

void readImage(vector<string>& imagesPath,vector<string>& imagesClasses,string rootFolder, string classFolder){
    
    string command = "ls "+rootFolder+"/"+classFolder+" > temp.txt";
    system(command.c_str());
    
    ifstream file("temp.txt");
    
    for(string line; getline(file,line);){
        string path = rootFolder+"/"+classFolder+"/"+line;
        imagesPath.push_back(path);
        imagesClasses.push_back(classFolder);
    }
}

void readImagesPaths(vector<string>& imagesPaths,vector<string>& imagesClasses, bool isTest){
    
    string folder = isTest ? TestFolder : TrainFolder;
    
    string command = "ls " + folder + " > " + fileWithClassesNames;
    system(command.c_str());
    
    cout << endl << "The File:" << fileWithClassesNames << " contains all class names." << endl; 
    
    ifstream file(fileWithClassesNames.c_str());
    
    if(!file.is_open())
        return;
    
    for(string line; getline(file,line);){
        classSet.push_back(line);
        readImage(imagesPaths,imagesClasses,folder,line);
    }

    //Sort images names.
    sort( imagesPaths.begin() , imagesPaths.end());
    
}



void printStringVector(vector<string>& vector){
    
    cout << "String Vector:" <<  endl;
    
    for (size_t i = 0; i < vector.size() ; i++){
        cout << "   " << vector[i].c_str() << endl;
    }
}



// Return the index of the class from the name of the training  image
int getClassNumber(string className) {

    for (int i = 0; i < classSet.size(); ++i) {
        if (classSet[i] == className) {
            return i;
        }
    }
    cout << "Problem finding the class name inside trainImageName";
    return -1;
}





























































