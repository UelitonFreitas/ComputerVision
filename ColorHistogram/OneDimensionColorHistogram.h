
#ifndef __ONEDIMENSIONCOLORHISTOGRAM_H
#define __ONEDIMENSIONCOLORHISTOGRAM_H

#include <stdlib.h>

using namespace std;

class OneDimensionColorHistogram{
    
    
    private:
        int nBias;
        int minChannelValue;
        int maxChannelValue;
        vector<int> aChannelHistogram;
        vector<int> bChannelHistogram;
        vector<int> cChannelHistogram;
        
        
        
    public:
        
        OneDimensionColorHistogram(int nBias = 16,int minChannelValue = 0, int maxChannelValue = 256):
            nBias(nBias),
            aChannelHistogram(nBias,0),
            bChannelHistogram(nBias,0),
            cChannelHistogram(nBias,0),
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
            
            this->computeAChannelColorHistogram(aChannel);
            this->computeBChannelColorHistogram(bChannel);
            this->computeCChannelColorHistogram(cChannel);
            
        }
        
        //1D Histogram.
        void computeAChannelColorHistogram(int aChannel){
            this->aChannelHistogram[aChannel] = this->aChannelHistogram[aChannel] + 1;
        }
       
        void computeBChannelColorHistogram(int bChannel){
            this->bChannelHistogram[bChannel] = this->bChannelHistogram[bChannel] + 1;
        }
       
        void computeCChannelColorHistogram(int cChannel){
            this->cChannelHistogram[cChannel] = this->cChannelHistogram[cChannel] + 1;
        }
        
        
};

#endif //OneDimensionColorHistogram.h