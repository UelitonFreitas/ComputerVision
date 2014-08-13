
#ifndef __TWODIMENSIONCOLORHISTOGRAM_H
#define __TWODIMENSIONCOLORHISTOGRAM_H

#include <stdlib.h>

using namespace std;

class TwoDimensionColorHistogram{
    
    private:
        int nBias;
        int minChannelValue;
        int maxChannelValue;
        
        vector<vector<int> > abChannelHistogram;
        vector<vector<int> > acChannelHistogram;
        vector<vector<int> > bcChannelHistogram;
        
    public:
        
        TwoDimensionColorHistogram(int nBias = 16,int minChannelValue = 0, int maxChannelValue = 256):
            nBias(nBias),
            abChannelHistogram(nBias,vector<int>(nBias,0)),
            acChannelHistogram(nBias,vector<int>(nBias,0)),
            bcChannelHistogram(nBias,vector<int>(nBias,0)),
            minChannelValue(minChannelValue),
            maxChannelValue(maxChannelValue)
        {
            //do nothing
        }
        
        //Put a channel value in a range(Bias).
        //Divide the channel values in nBias sets, indexed from 0 to nBias-1.
        int channelValueBinIndex(int aValue){
            int range = (this->maxChannelValue- this->minChannelValue)/this->nBias;
            int index = 0;
            int factor = 1;
             
            while(aValue >= factor*range){
                index++;
                factor++;
            }
            return index;
        }
        
        void addColor(int aChannel, int bChannel, int cChannel){
            
            aChannel = channelValueBinIndex(aChannel);
            bChannel = channelValueBinIndex(bChannel);
            cChannel = channelValueBinIndex(cChannel);
            
            this->computeABChannelColorHistogram(aChannel,bChannel);
            this->computeACChannelColorHistogram(aChannel,cChannel);
            this->computeBCChannelColorHistogram(bChannel,cChannel);
            
        }
        
        //2D Histograms.
        void computeABChannelColorHistogram(int aChannel, int bChannel){
            this->abChannelHistogram[aChannel][bChannel] = this->abChannelHistogram[aChannel][bChannel] + 1;
        }
        
        void computeACChannelColorHistogram(int aChannel, int cChannel){
            this->acChannelHistogram[aChannel][cChannel] = this->acChannelHistogram[aChannel][cChannel] + 1;
        }
        
        void computeBCChannelColorHistogram(int bChannel, int cChannel){
            this->bcChannelHistogram[bChannel][cChannel] = this->bcChannelHistogram[bChannel][cChannel] + 1;
        }
 
};

#endif //TwoDimensionColorHistogram.h