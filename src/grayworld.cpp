#include <climits>

#include <x86intrin.h>

#include "boost/program_options.hpp"
#include "opencv2/core/core.hpp"

#include "grayworld.hpp"
#include "util.hpp"

cv::Mat AWB(const cv::Mat& in, float s1, float s2)
{
	uint H = in.rows,
	     W = in.cols,
	     N = H * W;

	// Create output file
	cv::Mat out = cv::Mat(in.size(), in.type());

	// Histogram with 2^8 bins
	std::vector<uchar> hist(UCHAR_MAX);

	// For each channel...
	for (uint c = 0; c < 3; c++)
	{
		// Reset histogram
		for (uint i = 0; i < UCHAR_MAX; i++)
			hist[i] = 0;

		// Construct histogram
		uchar v;
		for (uint j = 0; j < H; j++)
			for (uint i = 0; i < W; i++)
			{
				v = in.at<cv::Vec3b>(j, i)[c];
				hist[v]++;
			}

		// Construct cumulative histogram
		for (uint i = 1; i < UCHAR_MAX; i++)
			hist[i] += hist[i - 1];

		// Calculate V_min and V_max
		uchar vmin     = 0,
		      vmax     = UCHAR_MAX - 1,
		      vmin_lim = floor(s1 * (float)(N / 100.f)),
		      vmax_lim = ceil((float)N * (1.f - s2 / 100.f));
		while (hist[vmin + 1] < vmin_lim) vmin++;
		while (hist[vmax - 1] > vmax_lim) vmax--;
		if    (vmax < UCHAR_MAX - 1)      vmax++;

		// Saturate pixels
		for (uint j = 0; j < H; j++)
			for (uint i = 0; i < W; i++)
			{
				v = in.at<cv::Vec3b>(j, i)[c];
				if      (v < vmin) v = vmin;
				else if (v > vmax) v = vmax;
				v = (v - vmin) * (UCHAR_MAX / (vmax - vmin));
				out.at<cv::Vec3b>(j, i)[c] = v;
			}
	}

	return out;
}


cv::Mat AWB_SSE(const cv::Mat& in, float s1, float s2)
{
	uint H = in.rows,
	     W = in.cols,
	     N = H * W;

	// Create output file
	cv::Mat out = cv::Mat(in.size(), in.type());

	// Histogram with 2^8 bins
	std::vector<uchar> hist(UCHAR_MAX);

	// Look up table for final rescaling
	std::vector<uchar> lut(UCHAR_MAX);

	// Image of channel c
	std::vector<uchar> subin(N);

	// For each channel...
	for (uint c = 0; c < 3; c++)
	{
		uint i;

		// Reset histogram
		__m128i zeros = _mm_set1_epi8(0x00);
		for (i = 0; i < UCHAR_MAX; i += 16)
			_mm_store_si128((__m128i*) &hist[i], zeros);

		// Create copy as subin
		// and construct histogram
		const cv::Vec3b* _in = in.ptr<cv::Vec3b>(0);
		uchar v;
		for (i = 0; i < N; i++)
		{
			v = _in[i][c];
			subin[i] = v;
			hist[v]++;
		}

		// Construct cumulative histogram
		for (i = 1; i < UCHAR_MAX; i++)
			hist[i] += hist[i - 1];

		// Calculate V_min and V_max
		uchar vmin     = 0,
		      vmax     = UCHAR_MAX - 1,
		      vmin_lim = floor(s1 * (float)(N / 100.f)),
		      vmax_lim = ceil((float)N * (1.f - s2 / 100.f));
		while (hist[vmin + 1] < vmin_lim) vmin++;
		while (hist[vmax - 1] > vmax_lim) vmax--;
		if    (vmax < UCHAR_MAX - 1)      vmax++;

		// Correct pixels range
		uchar vsuf = UCHAR_MAX / (vmax - vmin);
		__m128i ones   = _mm_set1_epi8(0xFF),
			hchar  = _mm_set1_epi8(0x80),
			_vmin  = _mm_set1_epi8(vmin - 128),
		        _vmax  = _mm_set1_epi8(vmax - 128),
		        _vsuf  = _mm_set1_epi8(vsuf - 128),
			_uvmin = _mm_set1_epi8(vmin),
		        _uvmax = _mm_set1_epi8(vmax),
		        _uvsuf = _mm_set1_epi8(vsuf);

		// Set v < vmin to vmin
		for (i = 0; i < N; i += 16)
		{
			__m128i dat = _mm_load_si128((__m128i*) &subin[i]);
			__m128i cmp = _mm_cmplt_epi8(
					_mm_sub_epi8(dat, hchar),
					_vmin);                    // cmp
			__m128i tru = _mm_and_si128(cmp, ones);    // true
			__m128i fal = _mm_andnot_si128(cmp, ones); // false

			dat = _mm_or_si128(
				_mm_and_si128(_uvmin, tru),
				_mm_and_si128(dat,   fal)
			);
			_mm_store_si128((__m128i*) &subin[i], dat);
		}
		for (; i < N; i++)
			if (subin[i] < vmin) subin[i] = vmin;

		// Set v > vmax to vmax
		for (i = 0; i < N; i += 16)
		{
			__m128i dat = _mm_load_si128((__m128i*) &subin[i]);
			__m128i cmp = _mm_cmpgt_epi8(
					_mm_sub_epi8(dat, hchar),
					_vmax);                    // cmp
			__m128i tru = _mm_and_si128(cmp, ones);    // true
			__m128i fal = _mm_andnot_si128(cmp, ones); // false

			dat = _mm_or_si128(
				_mm_and_si128(_uvmax, tru),
				_mm_and_si128(dat,   fal)
			);
			_mm_store_si128((__m128i*) &subin[i], dat);
		}
		for (; i < N; i++)
			if (subin[i] > vmax) subin[i] = vmax;

		// Saturate
		// Construct lookup table
		for (i = 0; i < UCHAR_MAX; i++)
			lut[i] = (i - vmin) * vsuf;

		// Write to output
		cv::Vec3b* _out = out.ptr<cv::Vec3b>(0);
		for (i = 0; i < N; i++)
			_out[i][c] = lut[subin[i]];
	}

	return out;
}

