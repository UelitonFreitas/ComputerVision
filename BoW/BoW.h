#ifndef __BOW_H
#define __BOW_H

#include "cv.h"
#include "cxcore.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/ml/ml.hpp"
#include <stdlib.h>

using namespace cv;
using namespace std;


class BoW{
    
    private:
        int                         dictionarySize;     //BoW dictionary size.
        BOWKMeansTrainer*           bowTrainer;         //Bow Trainer.
        Mat                         dictionary;         //BoW dictionary.
        BOWImgDescriptorExtractor   *bowDE;             //BoW image descriptor. For testing.
        
        vector<Mat>*                trainImages;        //Vector of images.
        vector<Mat>                 trainDescriptors;   //Vector of keypoints descriptors.
        vector< vector<KeyPoint> >  trainKeyPoints;     //Vector of keypoints of each image.
        vector<string>*             imagesClass;
        
        
        vector<Mat>*                testImages;         //Vector of test images.
        vector<Mat>                 testDescriptors;    //Vector of test Descriptors;
        vector< vector<KeyPoint> >  testKeyPoints;      //Vector fo test Keypoints;
        
        vector<Mat>                 imageAttributes;
        
        Ptr<FeatureDetector>        featureDetector;    //Default: Surf
        Ptr<DescriptorExtractor>    descriptorExtractor;
        Ptr<DescriptorMatcher>      descriptorMatcher;  
        
        
        string  detectorType;       //Default SURF.
        string  descriptorType;     //Default SURF.
        string  matcherType;        //Default FLANN
        float    hessianThreshold;  //Theshold, the larger value, less keypoints.
        
        
        string tag;
        char fileName[100];         //Name of dictionary to be saved;
    
    public:
        
        BoW(string  detectorType = "SURF", string  descriptorType = "SURF",string  matcherType  = "FlannBased",float   hessianThreshold = 0.1,int dicSize = 64){
            
            this->detectorType = detectorType;
            this->descriptorType =descriptorType;
            this->matcherType = matcherType;
            this->hessianThreshold =hessianThreshold;
            
            this->dictionarySize = dicSize;
            TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
            int retries=1;
            int flags=KMEANS_PP_CENTERS;
            
            this->createDetectorDescriptorMatcher();
            
            this->bowTrainer = new BOWKMeansTrainer(this->dictionarySize, tc, retries, flags );
            
            this->bowDE = new BOWImgDescriptorExtractor(this->descriptorExtractor,this->descriptorMatcher);
            
            this->tag = "Bag Of Words: ";
        }
        
        
        void runTraining(){
            
            if(this->trainImages != NULL and this->imagesClass != NULL){
                this->trainFeaturesDetect();
                this->trainKeyPointsDescriptors();
                this->createVocabulary();
                this->setVocabularyOnImageDescriptor();
                this->saveDictionary();
                this->createImagesAttributes();
            }
            else{
                cout << endl<< this->tag << "You have to load train imagens and class names.";
            }
        }
        
        void createImageAttribute(Mat& image){
            
            vector<KeyPoint> keyPoints;
            Mat bowDescriptor;
            
            this->featureDetector->detect(image,keyPoints);
            
            
            this->bowDE->compute(image,keyPoints,bowDescriptor);
           
            this->imageAttributes.push_back(bowDescriptor);
        }
        
        void setVocabularyOnImageDescriptor(){
            this->bowDE->setVocabulary(this->dictionary);
        }
        
        void createImagesAttributes(){
            
           
            //this->setVocabulary();
            
            cout << endl << this->tag << "Creating images Attributes..." << endl;
            for(int i = 0; i < this->trainImages->size() ; i++){
                this->createImageAttribute( this->trainImages->at(i));
            }
            cout << "Complete!!" << endl;
            
        }
        
        void loadTrainImages(vector<Mat>& images,vector<string>& imagesClass){
            cout << endl << this->tag << "Loading images..." << endl;
            this->trainImages = new vector<Mat>(images);
            this->imagesClass = new vector<string>(imagesClass);
            cout << "Complete!!!" << endl;
        }
        
        void loadTestImages(vector<Mat>& images){
            this->testImages = new vector<Mat>(images);
        }
        
        void trainFeaturesDetect(){
            cout << endl << this->tag << "Detecting keypoints of " << this->trainImages->size() << " images." << endl;
            this->featureDetector->detect(*(this->trainImages),this->trainKeyPoints);
            cout << "Complete!!!" << endl;
        }
        
        void trainKeyPointsDescriptors(){
            
            cout << endl << this->tag << "Describing Key Points...." << endl;
            this->descriptorExtractor->compute(*(this->trainImages),this->trainKeyPoints,this->trainDescriptors);
            cout << "Complete!!!" << endl;    
            
        }
        
        
        void createVocabulary(){
            
            cout << endl << this->tag << "Creating vocabulary...." << endl;
            
            for(size_t i = 0 ; i < this->trainDescriptors.size(); i++){
                
                Mat descriptor = this->trainDescriptors[i];
                for(int j = 0; j < descriptor.rows; j++){
                    this->bowTrainer->add(descriptor.row(j));
                }
            }
            cout << this->tag <<  "Appliyng k-means..." << endl;
            this->dictionary = this->bowTrainer->cluster();
            
            
            cout << this->tag << "Dictionary created with size " << this->dictionarySize << endl;
            
            
        }
        
        void loadDictionary(string fileName){
            
            //this->fileName = fileName;
            FileStorage file(fileName, FileStorage::READ);
            
            if(file.isOpened()){
                cout << tag << "Loading Dictionary with size: " << this->dictionarySize << " ...." <<endl;;
                file["dictionary"] >> this->dictionary;
            }
            else
                cout << this->tag << "Cannot load the dictionary file." << endl;
            
        }
        
        
        void saveDictionary(){
            
            cout << tag << "Saving Dictionary with size: " << this->dictionarySize << " ...." <<endl;
            sprintf(this->fileName,"Dictionary-%02d.xml",this->dictionarySize);
            
            FileStorage file(this->fileName, FileStorage::WRITE);
            
            if(file.isOpened()){
                file << "dictionary" << this->dictionary;
                cout << tag << "Dictionary saved!! " << endl;
            }
            else
                cout << "Can create dictionary file!!" << endl;
        }
        
        bool createDetectorDescriptorMatcher(){
            initModule_nonfree();
            this->featureDetector     = FeatureDetector::create(detectorType);
            this->descriptorExtractor = DescriptorExtractor::create(descriptorType);
            this->descriptorMatcher   = DescriptorMatcher::create(matcherType);
        
            //shows the parameters of a given algorithm
            //printParams(this->featureDetector);
            this->featureDetector->set("hessianThreshold", this->hessianThreshold);
        }

        
        vector<vector<KeyPoint> >& getTrainKeyPoints(){
            return this->trainKeyPoints;
        }
        
        /*
        vector<Mat>& getImagesAttributes(){
            return this->imageAttributes;
        }
        */
        
        // TESTAR PARA IMPLEMENTAR PADRÃO DE COMUNICAÇÃO
        vector<vector<float> >& getImagesAttributes(){
            
            int     numberOfImages  = this->imageAttributes.size();
            vector<vector<float> >*  features = new  vector<vector<float> >(numberOfImages,vector<float>(this->dictionarySize));
            
            for(int i = 0; i <  numberOfImages; i++){
                Mat row = this->imageAttributes[i].row(0);
                for(int k = 0 ; k < row.cols; k++){
                    (features->at(i)).at(k) = (float)row.data[k];
                }
            }
            
            return *features;
        }
        
        Mat& getImagesAttributesOfTestImage(){
            return this->imageAttributes[0];
        }
    
};




#endif
