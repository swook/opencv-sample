#pragma once

#include <functional>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

typedef uint8_t  uchar;
typedef uint32_t uint;
typedef uint64_t ulong;

extern bool GRAPHICAL; // Flag to enable/disable showImage

/**
 * Shows an image using imshow if GRAPHICAL is true
 * Disabled by option --headless/-hl
 */
void showImage(std::string title, const cv::Mat& img);

/**
 * Runs the provided function at most max_steps times and returns median cycles
 */
uint benchmark(std::function<void()> func, uint max_steps);

