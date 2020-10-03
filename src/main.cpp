#include <stdio.h>

#include "perf.h"
#include "mosse.h"

PerfUtil pfu;

int main() 
{
    int picW = 640, picH = 360;
    Rect rect = { 387, 198, 30, 62 };
    Mosse tracker(rect.w, rect.h);

    return 0;
}
