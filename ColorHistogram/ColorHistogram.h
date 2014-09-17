


#ifndef __COLORHISTOGRAM_H
#define __COLORHISTOGRAM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageHistogram.h"

using namespace cv;
using namespace std;


enum colorSpace{
    _HSVColorSpace,
    _RGBColorSpace
};

class ColorHistogram{

    private:
        string                  tag;
        int                     colorSpace;
        int                     binSize;
        vector<Mat>*            trainImages;
        vector<ImageHistogram*>* trainImageHistogram;

        vector<string>* imagesClass;

        bool            _1DHistogram;
        bool            _2DHistogram;
        bool            _3DHistogram;


    public:
        ColorHistogram(int binSize = 2, int colorSpace = _HSVColorSpace, bool _1D = false, bool _2D = false, bool _3D = true){
            this->binSize       = binSize;
            this->colorSpace    = colorSpace;//= new String(colorSpace);

            this->_1DHistogram  = _1D;
            this->_2DHistogram  = _2D;
            this->_3DHistogram  = _3D;

            this->tag = "Color Histogram: ";

        }

        //Precisa melhorar
        vector<float>& createHistogram(Mat& testImage){

            if(this->colorSpace == _HSVColorSpace)
                cvtColor(testImage,testImage,CV_RGB2HSV);

            ImageHistogram* testFeature  = new ImageHistogram(testImage,this->binSize,this->_1DHistogram,this->_2DHistogram,this->_3DHistogram);
            return createFeatureMat(*testFeature);

        }

        void createHistograms(vector<Mat>& images,vector<string>& imagesClass){

            cout << endl << this->tag.c_str() << "Loading Images..." << endl;

            this->trainImageHistogram = new vector<ImageHistogram*>(images.size());

            for (int i = 0; i < images.size(); i++){

                if(this->colorSpace == _HSVColorSpace)
                    cvtColor(images[i],images[i],CV_RGB2HSV);

                this->trainImageHistogram->at(i) = new ImageHistogram(images[i],this->binSize,this->_1DHistogram,this->_2DHistogram,this->_3DHistogram);

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

        inline int getBinSize(){
            return this->binSize;
        }



};
#endif
