#include <stdio.h>

#include "perf.h"
#include "mosse.h"

//PerfUtil pfu;


int main() 
{
    PFU_START("test");

    int picW = 640, picH = 360;
    Rect rect = { 387, 198, 30, 62 };
    Mosse tracker(rect.w, rect.h);

    PFU_STOP("test");

    return 0;
}
