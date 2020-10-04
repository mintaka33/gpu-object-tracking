#pragma once

#include <stdint.h>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>

class PerfUtil
{
public:
    struct Tick
    {
        double freq;
        int64_t start;
        int64_t stop;
        double time;
    };
    struct PerfInfo
    {
        uint32_t count;
        double avg;
        double max;
        double min;
    };

public:
    PerfUtil();
    ~PerfUtil();

    static PerfUtil& getInstance() 
    {
        static PerfUtil instance;
        return instance;
    }

    void startTick(std::string tag);
    void stopTick(std::string tag);
    void savePerfData();

private:
    void printPerfSummary();
    void printPerfDetails();
    void printHeader(std::ofstream& fout);
    void printBody(std::ofstream& fout);
    void printFooter(std::ofstream& fout);
    std::string formatPerfData(std::string tag, std::vector<Tick>& record);
    void getPerfInfo(std::vector<Tick>& record, PerfInfo* info);
    std::string getDashString(uint32_t num);

private:
    std::map<std::string, std::vector<Tick>*> records;
};

#define PFU_START(tag) PerfUtil::getInstance().startTick(tag)
#define PFU_STOP(tag)  PerfUtil::getInstance().stopTick(tag)
