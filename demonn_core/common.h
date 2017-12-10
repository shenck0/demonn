#pragma once
#include <stdio.h>

#ifdef _MSC_VER
    #define EXPORT_SYMBOL __declspec(dllexport)
#else
    #define EXPORT_SYMBOL __attribute((visibility("default")))
#endif

// Helper macros
#define check(expr) \
    do{ \
        if(!(expr)){ \
            printf("exception at %s L%d\n", __FILE__, __LINE__); \
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