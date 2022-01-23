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

    char *frame = nullptr;
    size_t picW = 1920, picH = 1080;
    RoiRect ri = { 6, 599, 517, 421 };

    Mosse tracker;

#ifdef USE_OPENCV
    Mat rawMat, inputMat, grayMat;
    cv::VideoCapture cap("test.264");
    cap >> rawMat;
    if (rawMat.empty()) {
        printf("ERROR: input video stream is empty, exiting\n");
        return -1;
    }
    cv::resize(rawMat, inputMat, cv::Size(picW, picH));
    cvtColor(inputMat, grayMat, COLOR_BGR2GRAY);
    frame = (char*)grayMat.data;
#else
    frame = new char[picW * picH];
    if (loadFrame("input2\\tmp.001.yuv", frame, picW, picH)) {
        printf("ERROR: cannot load input yuv files, exiting\n")
        return -1;
    }
#endif

    tracker.init(frame, picW, picH, ri);
    tracker.dump2txt();

    for (size_t i = 2; i <= 2; i++) {
#ifdef USE_OPENCV
        cap >> rawMat;
        if (rawMat.empty()) {
            printf("INFO: input video stream is empty, exiting\n");
            return -1;
    }
        cv::resize(rawMat, inputMat, cv::Size(picW, picH));
        cvtColor(inputMat, grayMat, COLOR_BGR2GRAY);
        frame = (char*)grayMat.data;
#else
        char filename[256] = {};
        sprintf_s(filename, "input2\\tmp.%03d.yuv", i);
        if (loadFrame(filename, frame, picW, picH)) {
            return -1;
        }
#endif
        RoiRect ro = {};
        tracker.update(frame, picW, picH, ro);

        tracker.dumpResult();

        //printf("OutROI: %d, %d, %d, %d\n", ro.x, ro.y, ro.w, ro.h);

        //tracker.dump2txt();
        //tracker.dump2bin();
    }

    PFU_STOP("Total");

#ifdef USE_OPENCV
#else
    delete[] frame;
#endif

    return 0;
}
