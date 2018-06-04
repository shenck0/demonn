#pragma once
#include <stdio.h>
#include "demonn_config.h"

#ifdef _MSC_VER
    #define export_symbol __declspec(dllexport)
#else
    #define export_symbol __attribute((visibility("default")))
#endif

// Function naming
#define op(function, algorithm, direction, platform, implement) \
    function ## _ ## algorithm ## _ ## direction ## _ ## platform ## _ ## implement

// Helper macros
#define check(expr) \
    do{ \
        if(!(expr)){ \
            printf("check failed @file:%s line:%d expr:%s\n", __FILE__, __LINE__, #expr); \
            throw -1; \
        } \
    } while(0)

#ifdef NDEBUG
    #define checkd(expr) \
        ((void)0)
#else
    #define checkd(expr) \
            check(expr)
#endif

#define free_and_clear(ptr) \
    do{ \
        if (ptr) { \
            free(ptr); \
            ptr = 0; \
        } \
    } while(0)