#ifndef __COLORMOMENTS_H
#define __COLORMOMENTS_H


#include "cv.h"
#include <stdlib.h>


using namespace cv;
using namespace std;

//A auxiliary class that contains the color moments informations.
class ColorMoments{
    
    private:
        float mean[3];
        float variance[3];
        
    public:
        ColorMoments(){
        }
        
        ColorMoments(float* mean,float* var){
            
            for (int i = 0 ; i < 3 ; i++){
                this->mean[i]       = mean[i];
                this->variance[i]   = var[i];
            }
        }
    
        float* getMean(){
            return this->mean;
        }
        
        float* getVariance(){
            return this->variance;
        }
        
        void computeMoments(Mat& image, KeyPoint& kp){
            
            int x = kp.pt.x;
            int y = kp.pt.y;
            
            float   mean[3] = {0.0,0.0,0.0};
            float   var[3]  = {0.0,0.0,0.0};
            
            //Number Of Pixels that was in 5x5 range.
            int     numberOfPixels = 0;
            
            //Get Mean of 5x5 pixel around key point kp.
            for (int i = x-3; i < x+2; i++){
                for(int j = y-3; j < y+2 ; j++){
                    
                    if(isInRange(i,j,image.cols,image.rows)){
                        Vec3b pixel = image.at<Vec3b>(i,j);
                        int aChannel    = int( pixel.val[0] );
                        int bChannel    = int( pixel.val[1] );
                        int cChannel    = int( pixel.val[2] );
                        
                        mean[0] += aChannel;
                        mean[1] += bChannel;
                        mean[2] += cChannel;
                        
                        numberOfPixels++;
                    }
                }
            }
            
            //Mean
            this->mean[0] = mean[0]/numberOfPixels;
            this->mean[1] = mean[1]/numberOfPixels;
            this->mean[2] = mean[2]/numberOfPixels;
            
            for (int i = x-3; i < x+2; i++){
                for(int j = y-3; j < y+2 ; j++){
                    
                    if(isInRange(i,j,image.cols,image.rows)){
                        Vec3b pixel = image.at<Vec3b>(i,j);
                        int aChannel    = int( pixel.val[0] );
                        int bChannel    = int( pixel.val[1] );
                        int cChannel    = int( pixel.val[2] );
                        
                        var[0] += pow(aChannel - this->mean[0],2);
                        var[1] += pow(bChannel - this->mean[1],2);
                        var[2] += pow(cChannel - this->mean[2],2);
                    }
                }
            }
            
            this->variance[0] = pow( (var[0]/numberOfPixels) ,0.5);
            this->variance[1] = pow( (var[1]/numberOfPixels) ,0.5);
            this->variance[2] = pow( (var[1]/numberOfPixels) ,0.5);
        }
        
        bool isInRange(int x, int y,int h,int w){
            if( (x < 0) or (y < 0) )
                return false;
            if(x >= h)
                return false;
            if(y >= w)
                return false;
            
            return true;
        }
};





#endif