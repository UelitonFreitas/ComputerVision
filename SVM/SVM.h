

#ifndef __SVM_H
#define __SVM_H
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <stdlib.h>
#include <utility>



using namespace std;
using namespace cv;

class SVMClass{
    
    private:
        int             numberOfFeatures;
        int             numberOfClasses;
        int             numberOfAttributes;
        string          tag;
    
        map<string,int> labelsMap;
        
        
        // Set up SVM's parameters
        CvSVMParams     params;
        CvSVM           SVM;
        
        
    public:
        
        SVMClass(vector<string>& setOfClasses, int numberOfAttributes = 0){
            
            this->numberOfClasses       = setOfClasses.size();
            this->numberOfAttributes    = numberOfAttributes;
            //Initialize labels map.
            for (int i = 0; i < this->numberOfClasses    ; i++){
                labelsMap[setOfClasses[i]] = i;
            }
            
            params.svm_type    = SVM::C_SVC;
            params.C           = 0.1;
            params.kernel_type = SVM::LINEAR;
            params.term_crit   = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
            
            this->tag = "SVM: ";
            
        }
        
        
        void train(vector<Mat>& featuresDescriptors, vector<string>& featuresClass){
            
            cout << endl << tag << "Training SVM...." << endl;
            this->numberOfFeatures      = featuresClass.size();
            this->numberOfAttributes    = featuresDescriptors[0].cols;
            
            // Set up training data
            //int labels[4] = {1, -1, -1, -1};
            //Mat labelsMat(4, 1, CV_32SC1, labels);
        
            //float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
            //Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
            
            
            //Initialize vector of features class.
            int labels[this->numberOfFeatures];
            
            for(int i = 0 ;i < this->numberOfFeatures   ; i++){
                labels[i] = this->labelsMap[featuresClass[i]];
            }
            
            Mat labelsMat(this->numberOfFeatures, 1,CV_32SC1, labels);
        
            //Class labels formats dor opencv svm formats.
            float trainingData[this->numberOfFeatures][this->numberOfAttributes];
            
            cout << "print" << endl;
            for(int i = 0; i < this->numberOfFeatures ; i++){
                Mat row = featuresDescriptors[i].row(0);
                for(int k = 0 ; k < row.cols; k++){
                    trainingData[i][k] = (double)row.data[k];
                }
            }
            
            
            //Train data format for opencv svm.
            Mat trainingDataMat(this->numberOfFeatures, this->numberOfAttributes,CV_32FC1, trainingData);
            
            SVM.train(trainingDataMat, labelsMat , Mat(), Mat(), params);
            cout << "Complete!!" << endl;

            
        }
        
        float predict(Mat& feature){
            
            float testData[1][this->numberOfAttributes];
            
            Mat row = feature.row(0);
            for(int k = 0 ; k < row.cols; k++){
                testData[0][k] = (double)row.data[k];
            }
            
            Mat testDataMat(1, this->numberOfAttributes, CV_32FC1,testData);
            
            return this->SVM.predict(testDataMat);
            
        }
        
        void saveModel(){
            
            cout << endl << tag << " Saving model on file: svmModel.xml..." << endl;
            this->SVM.save("svmModel.xml");
            cout << "Complete!!" << endl;
            
        }
        
        void loadModel(string file){
            this->SVM.load(file.c_str());
        }
        
        void printData(Mat& bowDescriptor){
            for ( int i = 0; i < bowDescriptor.rows; ++i) {
	            Mat row = bowDescriptor.row(i);
	            for ( int j = 0; j < row.cols; ++j) {
	                int binValue = row.data[j];
	                cout << binValue << ",";
	            }
	        }
        }

        

};




#endif