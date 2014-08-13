


#ifndef __COLORHISTOGRAM_H
#define __COLORHISTOGRAM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageHistogram.h"

using namespace cv;
using namespace std;


enum colorSpace{
    _HSV,
    _RGB
};

class ColorHistogram{
    
    private:
        string                  tag;
        string*                 colorSpace;
        int                     binSize;
        vector<Mat>*            trainImages;
        vector<ImageHistogram*>* trainImageHistogram;
        
        vector<string>* imagesClass;
        
        bool            _1DHistogram;
        bool            _2DHistogram;
        bool            _3DHistogram;
        
        
    public:
        ColorHistogram(int binSize = 16, string colorSpace = "HSV", bool _1D = false, bool _2D = false, bool _3D = true){
            this->binSize       = binSize;
            this->colorSpace    = new String(colorSpace);
            
            this->_1DHistogram  = _1D;
            this->_2DHistogram  = _2D;
            this->_3DHistogram  = _3D;
            
            this->tag = "Color Histogram: ";
            
        }
        
        vector<float>& createHistogram(Mat& testImage, bool _1D, bool _2D, bool _3D,string colorSpace){
            ImageHistogram* testFeature  = new ImageHistogram(testImage,this->binSize,_1D,_2D,_3D,colorSpace);
            
            if(_1D){
                
            }
            else if(_2D){
                
            }
            else if(_3D){
                return createFeatureMat(*testFeature);
            }
            //vector<vector<vector<int> > > v = this->trainImageHistogram->at(i)->get3DHistogram();
        }
        
        void createHistograms(vector<Mat>& images,vector<string>& imagesClass){
            
            cout << endl << this->tag.c_str() << "Loading Images..." << endl;
            
            this->trainImageHistogram = new vector<ImageHistogram*>(images.size());
            
            for (int i = 0; i < images.size(); i++){
                
                this->trainImageHistogram->at(i) = new ImageHistogram(images[i],this->binSize,this->_1DHistogram,this->_2DHistogram,this->_3DHistogram,*this->colorSpace);
                
                vector<vector<vector<int> > > v = this->trainImageHistogram->at(i)->get3DHistogram();
            }
            
            this->trainImages = new vector<Mat>(images);
            this->imagesClass = new vector<string>(imagesClass);
            
            cout << this->tag.c_str() << "Complete!!" << endl;
        }
        
        vector<vector<float> >& getHistograms(){
            //vector<Mat>*    features = new vector<Mat>(this->trainImageHistogram->size());
            
            int     numberOfImages  = this->trainImageHistogram->size();
            int     vecSize = this->binSize*this->binSize*this->binSize;
            vector<vector<float> >*  features = new  vector<vector<float> >(numberOfImages,vector<float>(vecSize));
            
            for (int i = 0; i < this->trainImageHistogram->size(); i++){
                features->at(i) = this->createFeatureMat(*this->trainImageHistogram->at(i));
            }
            
            return *features;
        }
        
        vector<float>&    createFeatureMat(ImageHistogram& imgHist){
           
            vector<vector<vector<int> > > hist =  imgHist.get3DHistogram();
            
            int w = hist.size();
            int h = hist[0].size();
            int d = hist[0][0].size();
            
            vector<float>* feature = new vector<float>(w*h*d);
            for(int i = 0 ;i < w; i++){
                for(int j = 0 ;j < h; j++){
                    for(int k = 0 ;k < d; k++){
                        feature->at(i*w*h + j*w + k)= hist[i][j][k];
                    }
                }
            }
            return *feature;
        }
        
        void computeTrainColorHistograms(){
            
            for (int i = 0; i < this->trainImages->size(); i++){
                //this->computeColorHistogram(this->trainImages->at(i));
            }
        }

    
};
#endif