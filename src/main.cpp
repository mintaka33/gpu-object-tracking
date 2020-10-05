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

    Mosse tracker;

    size_t picW = 640, picH = 360;
    Rect rect = { 387, 198, 30, 62 };
    char* frame = new char[picW * picH];
    loadFrame("tmp1.yuv", frame, picW, picH);

    tracker.init(frame, picW, picH, rect);

    tracker.dump();

    PFU_STOP("Total");

    return 0;
}
