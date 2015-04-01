#pragma once

#include "opencv2/core/core.hpp"


/**
 * Performs gray world equalisation.
 * Returns automatic white balance-d image.
 *
 * NOTE: Assumed that input is an CV_8U BGR image
 */
void AWB(const cv::Mat& in, cv::Mat& out);



/**
 * Performs AWB using SSE intrinsics
 */
void AWB_SSE(const cv::Mat& in, cv::Mat& out);

