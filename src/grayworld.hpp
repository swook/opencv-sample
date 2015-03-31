#pragma once

#include "opencv2/core/core.hpp"


/**
 * Performs gray world equalisation.
 * Returns automatic white balance-d image.
 *
 * NOTE: Assumed that input is an CV_8U BGR image
 */
cv::Mat AWB(const cv::Mat& in, float s1 = 1.5, float s2 = 1.5);



/**
 * Performs AWB using SSE intrinsics
 */
cv::Mat AWB_SSE(const cv::Mat& in, float s1 = 1.5, float s2 = 1.5);

