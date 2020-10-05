#include <stdio.h>

#include "perf.h"
#include "mosse.h"

int main() 
{
    PFU_START("Total");

    int picW = 640, picH = 360;
    Rect rect = { 387, 198, 30, 62 };
    Mosse tracker(rect.w, rect.h);
    tracker.dump();

    PFU_STOP("Total");

    return 0;
}
