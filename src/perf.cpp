
#ifdef WIN32
#include <windows.h>
#else

#endif
#include "perf.h"

PerfUtil::PerfUtil()
{
}

PerfUtil::~PerfUtil()
{
    savePerfData();

    for (const auto data : records)
    {
        if (data.second)
        {
            delete data.second;
        }
    }
    records.clear();
}

void PerfUtil::startTick(std::string tag)
{
    Tick newTick = {};

#ifdef WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    newTick.start = li.QuadPart;
    QueryPerformanceFrequency(&li);
    newTick.freq = (double)li.QuadPart / 1000;
#else
    struct timespec ts = {};
    clock_gettime(CLOCK_REALTIME, &ts);
    newTick.start = int(ts.tv_sec * 1000000) + int(ts.tv_nsec / 1000);
#endif

    std::vector<Tick> *perf = nullptr;
    std::map<std::string, std::vector<Tick>*>::iterator it;
    it = records.find(tag);
    if (it == records.end())
    {
        perf = new std::vector<Tick>;
        perf->push_back(newTick);
        records[tag] = perf;
    }
    else
    {
        it->second->push_back(newTick);
    }
}

void PerfUtil::stopTick(std::string tag)
{
    std::map<std::string, std::vector<Tick>*>::iterator it;
    it = records.find(tag);
    if (it == records.end())
    {
        return;
    }

#ifdef WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    it->second->back().stop = li.QuadPart;
    it->second->back().time = 
        double(it->second->back().stop - it->second->back().start) / it->second->back().freq;
#else
    struct timespec ts = {};
    clock_gettime(CLOCK_REALTIME, &ts);
    it->second->back().stop = int(ts.tv_sec * 1000000) + int(ts.tv_nsec / 1000);
    it->second->back().time = double(it->second->back().stop - it->second->back().start) / 1000.0;
#endif
}

void PerfUtil::savePerfData()
{
    printPerfSummary();
    printPerfDetails();
}

void PerfUtil::printPerfSummary()
{
    std::ofstream fout;
    fout.open("perf_summary.txt");

    printHeader(fout);
    printBody(fout);
    printFooter(fout);

    fout.close();
}

void PerfUtil::printPerfDetails()
{
    std::ofstream fout;
    fout.open("perf_details.txt");

    for (auto data : records)
    {
        fout << getDashString((uint32_t)data.first.length());
        fout << data.first << std::endl;
        fout << getDashString((uint32_t)data.first.length());
        for (auto t : *data.second)
        {
            fout << t.time << std::endl;
        }
        fout << std::endl;
    }

    fout.close();
}

void PerfUtil::printHeader(std::ofstream& fout)
{
    fout << "Summary: " << std::endl;
    fout << getDashString(80);
    std::stringstream ss;
    ss.width(16);
    ss << "Tag Name";
    ss.width(16);
    ss << "Hit Count";
    ss.width(16);
    ss << "Average (ms)";
    ss.width(16);
    ss << "Minimum (ms)";
    ss.width(16);
    ss << "Maximum (ms)" << std::endl;
    fout << ss.str();
    fout << getDashString(80);
}

void PerfUtil::printBody(std::ofstream& fout)
{
    for (const auto& data : records)
    {
        fout << formatPerfData(data.first, *data.second);
    }
}

std::string PerfUtil::formatPerfData(std::string tag, std::vector<Tick>& record)
{
    std::stringstream ss;
    PerfInfo info = {};
    getPerfInfo(record, &info);

    ss.width(16);
    ss << tag;

    ss.precision(2);
    ss.setf(std::ios::fixed, std::ios::floatfield);

    ss.width(16);
    ss << info.count;
    ss.width(16);
    ss << info.avg;
    ss.width(16);
    ss << info.min;
    ss.width(16);
    ss << info.max << std::endl;

    return ss.str();
}

void PerfUtil::getPerfInfo(std::vector<Tick>& record, PerfInfo* info)
{
    if (record.size() <= 0)
        return;

    info->count = (uint32_t)record.size();
    double sum = 0, max = 0, min = 10000000.0;
    for (auto t : record)
    {
        sum += t.time;
        max = (max < t.time) ? t.time : max;
        min = (min > t.time) ? t.time : min;
    }
    info->avg = sum / info->count;
    info->max = max;
    info->min = min;
}

void PerfUtil::printFooter(std::ofstream& fout)
{
    fout << getDashString(80);
}

std::string PerfUtil::getDashString(uint32_t num)
{
    std::stringstream ss;
    ss.width(num);
    ss.fill('-');
    ss << std::left << "" << std::endl;
    return ss.str();
}