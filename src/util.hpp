#pragma once

#include <functional>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

typedef unsigned int   uint;
typedef unsigned short ushort;
typedef unsigned char  uchar;

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

