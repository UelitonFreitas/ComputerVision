


#ifndef __IMAGEHISTOGRAM_H
#define __IMAGEHISTOGRAM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "OneDimensionColorHistogram.h"
#include "TwoDimensionColorHistogram.h"
#include "TreeDimensionColorHistogram.h"

using namespace cv;
using namespace std;



class ImageHistogram{
    private:
        
        string*         colorSpace;
        int             binSize;
        
        //Color Histograms.
        OneDimensionColorHistogram*     oneDHistogram;
        TwoDimensionColorHistogram*     twoDHistogram;
        TreeDimensionColorHistogram*    treeDHistogram;
        
    
    public:    
        ImageHistogram(Mat& image, int binSize, bool _1D, bool _2D , bool _3D ,string& colorSpace){
            this->binSize       = binSize;
            this->colorSpace    = new String(colorSpace);
            
            oneDHistogram   = _1D ? new OneDimensionColorHistogram(binSize) :  NULL;
            
            twoDHistogram   = _2D ? new TwoDimensionColorHistogram(binSize) : NULL;
            
            treeDHistogram  = _3D ? new TreeDimensionColorHistogram(binSize) : NULL;
            
            this->computeColorHistogram(image);
        }
        
        void computeColorHistogram(Mat& image){
            for (int i = 0; i < image.rows; i++){
                for (int j = 0; j < image.cols; j++){
                    Vec3b color = image.at<Vec3b>(i,j);
                    this->getColorInformation(color);
                }
            }
        }
        
        
        //compute the color information of color given.
        void getColorInformation(Vec3b& color){
            
            int aChannel    = int( color.val[0] );
            int bChannel    = int( color.val[1] );
            int cChannel    = int( color.val[2] );
            
            if(oneDHistogram != NULL){
                this->oneDHistogram->addColor(aChannel,bChannel,cChannel);
            }
            
            if(twoDHistogram != NULL){
                this->twoDHistogram->addColor(aChannel,bChannel,cChannel);
            }
            
            if(treeDHistogram != NULL){
                this->treeDHistogram->addColor(aChannel,bChannel,cChannel);
                
            }
        }
        
        
        Mat& get3DHistogram(){
            vector<vector<vector<int> > > h = this->treeDHistogram->getABCChannelHistogram(); 
        }
};
#endif