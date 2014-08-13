
#ifndef __TREEDIMENSIONCOLORHISTOGRAM_H
#define __TREEDIMENSIONCOLORHISTOGRAM_H

#include <stdlib.h>

using namespace std;

class TreeDimensionColorHistogram{

    private:
        int nBias;
        int minChannelValue;
        int maxChannelValue;
        
        //3D-Histograms
        vector<vector<vector<int> > > abcChannelHistogram;
        
    public:
        
        TreeDimensionColorHistogram(int nBias = 16,int minChannelValue = 0, int maxChannelValue = 256):
            nBias(nBias),
            abcChannelHistogram(nBias,vector<vector<int> >(nBias,vector<int>(nBias,0))),
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
         
            this->computeABCChannelHistogram(aChannel,bChannel,cChannel);
        }
     
        //3D Histograms.
        void computeABCChannelHistogram(int aChannel, int bChannel, int cChannel){
            this->abcChannelHistogram[aChannel][bChannel][cChannel] = this->abcChannelHistogram[aChannel][bChannel][cChannel] + 1;
        }
        
        vector<vector<vector<int> > >& getABCChannelHistogram(){
            return abcChannelHistogram;
        }
    
    
};

#endif //TreeDimensionColorHistogram.h