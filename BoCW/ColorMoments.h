#ifndef __COLORMOMENTS_H
#define __COLORMOMENTS_H


#include <opencv/cv.h>
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

            //cout << "Image:h-" << image.rows << " w-" << image.cols << endl;
            //cout << "Keypoint: x-" << x << " y-" << y << endl;

            //Get Mean of 5x5 pixel around key point kp.
            for (int i = x-3; i < x+2; i++){
                for(int j = y-3; j < y+2 ; j++){

                    if(isInRange(i,j,image.rows,image.cols)){
                        Vec3b pixel = image.at<Vec3b>(i,j);

                        int aChannel    = int( pixel.val[0] );
                        int bChannel    = int( pixel.val[1] );
                        int cChannel    = int( pixel.val[2] );

                        //cout << "RGB:  "<< aChannel << " " << bChannel << " " << cChannel << " " << endl;

                        mean[0] += aChannel;
                        mean[1] += bChannel;
                        mean[2] += cChannel;
                

                    }
                    numberOfPixels++;
                    //else
                      //  cout << "notrange - x:" << x << " y:" <<y << " w:" <<image.cols<<  " h:" <<image.rows << endl;
                }
            }


            //Mean
            this->mean[0] = mean[0]/numberOfPixels;
            this->mean[1] = mean[1]/numberOfPixels;
            this->mean[2] = mean[2]/numberOfPixels;


            for (int i = x-3; i < x+2; i++){
                for(int j = y-3; j < y+2 ; j++){

                    if(isInRange(i,j,image.rows,image.cols)){
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

        void getColorMoments(Mat& image,vector<KeyPoint>& kp,vector<ColorMoments>& colorMoments){

            for(int i = 0; i < kp.size(); i++){
                ColorMoments* cm = this->createColorMoments(image,kp[i]);
                colorMoments.push_back(*cm);
            }
        }

        ColorMoments* createColorMoments(Mat& image, KeyPoint& kp){

            this->computeMoments(image,kp);
            return new ColorMoments(this->mean,this->variance);

        }

        void printValues(){
            cout << endl << "ColorMoment" << endl;
            cout << mean[0] << ' ' << mean[1] << ' ' << mean[2] << endl;
            cout << variance[0] << ' ' << variance[1] << ' ' << variance[2] << endl;
        }

        inline bool isInRange(int i, int j,int rows,int cols){
            if( (i < 0) or (j < 0) ){
                return false;
            }
            if(j >= cols){
                return false;
            }
            if(i >= rows){
                return false;
            }

            return true;
        }
};





#endif
