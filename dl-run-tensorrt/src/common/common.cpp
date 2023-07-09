#include "common.h"
#include <cstdio>
#include <cstdarg>
#include <filesystem>

namespace common {
    void Info(const char* value, ...)
    {
        va_list args;
        va_start(args, value);
        vfprintf(stdout, value, args);
        va_end(args);
    }

    bool fileExists(const std::string& filePath) {
        return std::filesystem::exists(filePath);
    }

};

