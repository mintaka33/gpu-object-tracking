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
    Rect rect = { 387, 198, 30, 62 };

    Mosse tracker;

    char* frame = new char[picW * picH];
    loadFrame("tmp1.yuv", frame, picW, picH);

    tracker.init(frame, picW, picH, rect);

    char* frame2 = new char[picW * picH];
    loadFrame("tmp2.yuv", frame2, picW, picH);

    //tracker.update(frame2, picW, picH);

    tracker.dump2txt();
    tracker.dump2bin();

    PFU_STOP("Total");

    delete[] frame, frame2;
    return 0;
}
