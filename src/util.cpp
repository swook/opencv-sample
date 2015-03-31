#include <algorithm>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "util.hpp"
#include "tsc_x86.h"

bool GRAPHICAL = true;

void showImage(std::string title, const cv::Mat& img)
{
	if (GRAPHICAL)
	{
		std::cout << "Displaying \"" << title << "\"." << std::endl;
		cv::imshow(title, img);
	}
}

uint benchmark(std::function<void()> func, uint max_steps)
{
	myInt64 start, end;
	std::vector<uint> list(max_steps);

	for (uint i = 0; i < max_steps; i++)
	{
		start = start_tsc();
		func();
		end = stop_tsc(start);
		list[i] = end;
	}
	std::sort(list.begin(), list.end());
	return list[max_steps / 2];
}

