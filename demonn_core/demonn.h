#pragma once

// core
#include "common.h"
#include "basic.h"
#include "activation.h"
#include "loss.h"
#include "conv.h"
#include "training.h"
#include "initializer.h"

// util
#include "tensor.hpp"

/*
    naming rules:
        
        void Function_Algorithm_Direction_Platform_Implement(
            ...parameters...
        );
        
        Function: conv2d/fully_connected/etc
        Algorithm: algorithm description
        Direction: forward/backward
        Platform: cpu/gpu/etc
        Implement: implement description


*/