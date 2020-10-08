#include <stdio.h>
#include <fstream>

#include "perf.h"
#include "mosse.h"

using namespace std;

void loadFrame(char* filename, char* buf, size_t w, size_t h)
{
    ifstream infile;
    infile.open(filename, ios::binary);
    infile.read(buf, w * h);
    infile.close();
}

int main() 
{
    PFU_START("Total");

    size_t picW = 640, picH = 360;
    RoiRect ri = { 270, 160, 53, 33 };

    Mosse tracker;

    char* frame = new char[picW * picH];
    loadFrame("input2\\tmp.001.yuv", frame, picW, picH);

    tracker.init(frame, picW, picH, ri);
    //tracker.dump2txt();

    char* frame2 = new char[picW * picH];
    for (size_t i = 2; i <= 250; i++) {
        char filename[256] = {};
        sprintf_s(filename, "input2\\tmp.%03d.yuv", i);
        loadFrame(filename, frame2, picW, picH);

        RoiRect ro = {};
        tracker.update(frame2, picW, picH, ro);
        //printf("OutROI: %d, %d, %d, %d\n", ro.x, ro.y, ro.w, ro.h);

        //tracker.dump2txt();
        //tracker.dump2bin();
    }

    PFU_STOP("Total");

    delete[] frame, frame2;
    return 0;
}
