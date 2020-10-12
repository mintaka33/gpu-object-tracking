#include <stdio.h>
#include <fstream>

#include "perf.h"
#include "mosse.h"

using namespace std;

int loadFrame(char* filename, char* buf, size_t w, size_t h)
{
    ifstream infile;
    infile.open(filename, ios::binary);
    if (!infile.is_open()) {
        printf("ERROR: failed to open file %s!\n", filename);
        return -1;
    }
    infile.read(buf, w * h);
    infile.close();
    return 0;
}

int main() 
{
    PFU_START("Total");

    size_t picW = 640, picH = 360;
    RoiRect ri = { 270, 160, 53, 33 };

    Mosse tracker;

    char* frame = new char[picW * picH];
    if (loadFrame("input2\\tmp.001.yuv", frame, picW, picH)) {
        return -1;
    }

    tracker.init(frame, picW, picH, ri);
    //tracker.dump2txt();

    char* frame2 = new char[picW * picH];
    for (size_t i = 2; i <= 250; i++) {
        char filename[256] = {};
        sprintf_s(filename, "input2\\tmp.%03d.yuv", i);
        if (loadFrame(filename, frame2, picW, picH)) {
            return -1;
        }

        RoiRect ro = {};
        tracker.update(frame2, picW, picH, ro);

        tracker.dumpResult();

        //printf("OutROI: %d, %d, %d, %d\n", ro.x, ro.y, ro.w, ro.h);

        //tracker.dump2txt();
        //tracker.dump2bin();
    }

    PFU_STOP("Total");

    delete[] frame, frame2;
    return 0;
}
