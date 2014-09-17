#ifndef __BOFC_H
#define __BOFC_H

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/ml/ml.hpp"
#include <stdlib.h>

#include "ColorMoments.h"

using namespace cv;
using namespace std;



class CV_EXPORTS BoFC{

    private:
        int                             dictionarySize;     //BoFC dictionary size.
        int                             colorFeatures;      //Size of color informations.
        BOWKMeansTrainer*               bowTrainer;         //Bow Trainer.
        Mat                             dictionary;         //BoFC dictionary.
        BOWImgDescriptorExtractor       *bowDE;             //BoFC image descriptor. For testing.

        vector<Mat>*                    trainImages;        //Vector of images.
        vector<Mat>*                    colorTrainImages;    //Vector of color images.
        vector<Mat>                     trainDescriptors;   //Vector of keypoints descriptors.
        vector<Mat>                     trainColorMomentsDescriptors;   //Descriptros with SURF and color moments.
        vector< vector<KeyPoint> >      trainKeyPoints;     //Vector of keypoints of each image.
        vector<vector<ColorMoments> >   trainColorMoments;   //Vector of color moments of each train image.
        vector<string>*                 imagesClass;
        vector<Mat>                     finalDescriptors;   //Final Descriptors

        vector<Mat>*                    testImages;         //Vector of test images.
        vector<Mat>                     testDescriptors;    //Vector of test Descriptors;
        vector< vector<KeyPoint> >      testKeyPoints;      //Vector fo test Keypoints;

        vector<Mat>                     imageAttributes;

        Ptr<FeatureDetector>            featureDetector;    //Default: Surf
        Ptr<DescriptorExtractor>        descriptorExtractor;
        Ptr<DescriptorMatcher>          descriptorMatcher;


        string  detectorType;       //Default SURF.
        string  descriptorType;     //Default SURF.
        string  matcherType;        //Default FLANN
        float    hessianThreshold;  //Theshold, the larger value, less keypoints.


        string tag;
        char fileName[100];         //Name of dictionary to be saved;

    public:

        BoFC(string  detectorType = "SURF", string  descriptorType = "SURF",string  matcherType  = "FlannBased",float   hessianThreshold = 0.1,int dicSize = 64, int colorFeatures = 6){

            this->detectorType = detectorType;
            this->descriptorType =descriptorType;
            this->matcherType = matcherType;
            this->hessianThreshold =hessianThreshold;

            this->dictionarySize = dicSize + colorFeatures;
            this->colorFeatures = colorFeatures;
            TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
            int retries=1;
            int flags=KMEANS_PP_CENTERS;

            this->createDetectorDescriptorMatcher();

            this->bowTrainer = new BOWKMeansTrainer(dicSize + colorFeatures, tc, retries, flags );

            this->bowDE = new BOWImgDescriptorExtractor(this->descriptorExtractor,this->descriptorMatcher);

            this->tag = "Bag Of Features and Colors: ";

            this->trainColorMoments = vector<vector<ColorMoments> >();
        }


        void runTraining(){

            if(this->trainImages != NULL and this->imagesClass != NULL and this->colorTrainImages != NULL){
                //Detect KeyPoints
                this->trainFeaturesDetect();
                //Describe keypoints.
                this->trainKeyPointsDescriptions();

                //Compute color moments of each keypoint of images.
                this->computeColorMoments();

                //Create Descriptors with colors.
                this->createColorDescriptors();

                //Create VOcabulary of BoW.
                this->createVocabulary();


                this->computeVisualDescriptions();
                this->setVocabularyOnImageDescriptor();
                this->saveDictionary();
                //this->createImagesAttributes();
            }
            else{
                cout << endl<< this->tag << "You have to load train imagens and class names.";
            }
        }


        void computeVisualDescription(const Mat& image, vector<KeyPoint>& keypoints, Mat& imgDescriptor,Mat& colorImage){


            if( this->trainKeyPoints.empty() )
                return;

            vector<vector<int> >* pointIdxsOfClusters = 0;

            int clusterCount = this->dictionary.rows; // = vocabulary.rows
            // Compute descriptors for the image.
            Mat descriptors = Mat();
            this->descriptorExtractor->compute( image, keypoints, descriptors );               //Extract descriptors

            // Match keypoint descriptors to cluster center (to vocabulary)
            vector<DMatch> matches;

            vector<ColorMoments> colorMoments;
            ColorMoments* cm = new ColorMoments();
            cm->getColorMoments(colorImage,keypoints,colorMoments);

            Mat newDescriptor( descriptors.rows,(descriptors.cols+this->colorFeatures), descriptors.type());
            this->joinDescriptors(descriptors,colorMoments,newDescriptor);

           // cout<<"[ "<<newDescriptor.row(0)<<"]==>["<<newDescriptor.cols<<"]"<<endl;

            this->descriptorMatcher->match( newDescriptor , matches );

            imgDescriptor = Mat( 1, clusterCount, CV_32FC1, Scalar::all(0.0) );
            float *dptr = (float*)imgDescriptor.data;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                int queryIdx = matches[i].queryIdx;
                int trainIdx = matches[i].trainIdx; // cluster index
                CV_Assert( queryIdx == (int)i );

                dptr[trainIdx] = dptr[trainIdx] + 1.f;
                if( pointIdxsOfClusters )
                    (*pointIdxsOfClusters)[trainIdx].push_back( queryIdx );
            }

            // Normalize image descriptor.
            imgDescriptor /= descriptors.rows;

        }


        void computeVisualDescriptions(){

            this->descriptorMatcher->clear();
            this->descriptorMatcher->add( vector<Mat>(1, this->dictionary) );

            //Compute visual description of each image.
            for (int i = 0; i < this->trainImages->size(); i++){
                //Image keypoints.
                vector<KeyPoint> keypoints;

                //Detect Keypoints.
                this->featureDetector->detect(this->trainImages->at(i),keypoints);                    //Detect Keypoints

                //Descriptors of Keypoints.
                Mat descriptor;

                //Compute visual description and color moments of a image and put it in descriptor variable.
                this->computeVisualDescription(this->trainImages->at(i),keypoints,descriptor,this->colorTrainImages->at(i));

                this->finalDescriptors.push_back(descriptor);
            }
        }

        void setVocabularyOnImageDescriptor(){
            this->bowDE->setVocabulary(this->dictionary);
        }

        /*
        void createImagesAttributes(){

            cout << endl << this->tag << "Creating images Attributes..." << endl;
            for(int i = 0; i < this->trainImages->size() ; i++){
                this->createImageAttribute( this->trainImages->at(i),this->colorTrainImages->at(i));
            }
            cout << "Complete!!" << endl;

        }
        */
        void loadTrainImages(vector<Mat>& images,vector<Mat>& colorImages, vector<string>& imagesClass){
            cout << endl << this->tag << "Loading images..." << endl;
            this->trainImages = new vector<Mat>(images);
            this->imagesClass = new vector<string>(imagesClass);
            this->colorTrainImages = new vector<Mat>(colorImages);

            cout << "Complete!!!" << endl;
        }

        void trainFeaturesDetect(){
            cout << endl << this->tag << "Detecting keypoints of " << this->trainImages->size() << " images." << endl;
            this->featureDetector->detect(*(this->trainImages),this->trainKeyPoints);
            cout << "Complete!!!" << endl;
        }

        void trainKeyPointsDescriptions(){

            cout << endl << this->tag << "Describing Key Points...." << endl;
            this->descriptorExtractor->compute(*(this->trainImages),this->trainKeyPoints,this->trainDescriptors);
            cout << "Complete!!!" << endl;

        }


        void createVocabulary(){

            cout << endl << this->tag << "Creating vocabulary...." << endl;

            for(int i = 0 ; i < this->trainColorMomentsDescriptors.size(); ++i){

                Mat descriptor = this->trainColorMomentsDescriptors[i];
                for(int j = 0; j < descriptor.rows; ++j){
                    this->bowTrainer->add(descriptor.row(j));
                }
            }
            cout << this->tag <<  "Appliyng k-means..." << endl;
            this->dictionary = this->bowTrainer->cluster();


            cout << this->tag << "Dictionary created with size " << this->dictionary.cols << endl;


        }

        void loadDictionary(string fileName){

            //this->fileName = fileName;
            FileStorage file(fileName, FileStorage::READ);

            if(file.isOpened()){
                cout << tag << "Loading Dictionary with size: " << this->dictionarySize << " ...." <<endl;;
                file["dictionary"] >> this->dictionary;
                setVocabularyOnImageDescriptor();
            }
            else
                cout << this->tag << "Cannot load the dictionary file." << endl;

        }


        void saveDictionary(){

            cout << tag << "Saving Dictionary with size: " << this->dictionarySize << " ...." <<endl;
            sprintf(this->fileName,"BOFC-Dictionary-%02d.xml",this->dictionarySize);

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

            this->featureDetector->set("hessianThreshold", this->hessianThreshold);
        }


        vector<vector<KeyPoint> >& getTrainKeyPoints(){
            return this->trainKeyPoints;
        }


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

        vector<float>& getImagesAttributesOfTestImage(){

            vector<float>* feature = new vector<float>(this->dictionarySize);
            Mat row = this->imageAttributes[0].row(0);
            for(int k = 0 ; k < row.cols; k++){
                feature->at(k) = (float)row.data[k];
            }

            return *feature;
        }


        //Color Processing.
        //Compute a single color moment and put him in the list of color moments (in order).
        void computeColorMoment(Mat& image,vector<KeyPoint>& kp){

            vector<ColorMoments>* cms = new vector<ColorMoments>(kp.size());

            for (int i = 0 ; i < kp.size(); i++){
                cms->at(i).computeMoments(image,kp[i]);              // Show our image inside it.
            }
            this->trainColorMoments.push_back(*cms);

        }

        //Get color information of train images.
        void computeColorMoments(){

            for(int i = 0; i < this->colorTrainImages->size() ; i++){
                this->computeColorMoment(this->colorTrainImages->at(i),this->trainKeyPoints[i]);
            }
        }

        void createColorDescriptors(){

	    for ( int i = 0; i < this->trainDescriptors.size(); i++) {

                Mat newDescriptor( this->trainDescriptors[i].rows, (this->trainDescriptors[i].cols+this->colorFeatures), this->trainDescriptors[i].type());
                this->joinDescriptors(this->trainDescriptors[i],this->trainColorMoments[i],newDescriptor);
                this->trainColorMomentsDescriptors.push_back(newDescriptor);
            }
        }

        void joinDescriptors(Mat& descriptor,vector<ColorMoments> colorMoments,Mat& newDescriptor){

            Mat colorMoment( descriptor.rows, this->colorFeatures, descriptor.type());

            //Insert color moments information for each descriptor.
            for ( int j = 0; j < descriptor.rows; j++) {

                float* cmMean = colorMoments[j].getMean();
                float* cmVariance = colorMoments[j].getVariance();

                //Fill color moment matrix whit mean and variance.
                for(int k = 0; k < 3; k++){
                    colorMoment.at<float>(j,k) = cmMean[k];
                    colorMoment.at<float>(j,3+k) = cmVariance[k];
                }
            }

            //Concat descritptors with color moments in de same matrix newDescriptor.
            hconcat(descriptor,colorMoment,newDescriptor);
            //cout << "@@" << newDescriptor.cols << endl;
        }



};


#endif
