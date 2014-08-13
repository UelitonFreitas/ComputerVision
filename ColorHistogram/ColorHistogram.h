


#ifndef __COLORHISTOGRAM_H
#define __COLORHISTOGRAM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageHistogram.h"

using namespace cv;
using namespace std;

class ColorHistogram{
    
    private:
        
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
            
        }
        
        void loadTrainImages(vector<Mat>& images,vector<string>& imagesClass){
            this->trainImageHistogram = new vector<ImageHistogram*>(images.size());
            for (int i = 0; i < images.size(); i++){
                this->trainImageHistogram->at(i) = new ImageHistogram(images[i],this->binSize,this->_1DHistogram,this->_2DHistogram,this->_3DHistogram,*this->colorSpace);
            }
            
            this->trainImages = new vector<Mat>(images);
            this->imagesClass = new vector<string>(imagesClass);
        }
        
        
        
        void computeTrainColorHistograms(){
            
            for (int i = 0; i < this->trainImages->size(); i++){
                //this->computeColorHistogram(this->trainImages->at(i));
            }
        }

    
};
#endif