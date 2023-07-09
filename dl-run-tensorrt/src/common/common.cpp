#include "common.h"
#include <cstdio>
#include <cstdarg>


void Info(const char* value, ...)
{
    va_list args;
    va_start(args, value);
    vfprintf(stdout, value, args);
    va_end(args);
}


