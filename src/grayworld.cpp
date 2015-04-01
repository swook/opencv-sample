#include <climits>

#include <x86intrin.h>

#include "boost/program_options.hpp"
#include "opencv2/core/core.hpp"

#include "grayworld.hpp"
#include "util.hpp"

using namespace cv;

void AWB(const Mat& in, Mat& out)
{
	uint H = in.rows,
	     W = in.cols,
	     N = H * W;

	// Create output matrix
	out = Mat(in.size(), in.type());

	// Calculate sum of values
	ulong sum1 = 0, sum2 = 0, sum3 = 0;
	Vec3b pixel;
	for (uint j = 0; j < H; j++)
		for (uint i = 0; i < W; i++)
		{
			pixel = in.at<Vec3b>(j, i);
			sum1 += pixel[0];
			sum2 += pixel[1];
			sum3 += pixel[2];
		}

	// Find inverse of averages
	float inv1 = (float)N / (float)sum1,
	      inv2 = (float)N / (float)sum2,
	      inv3 = (float)N / (float)sum3;

	// Find maximum
	float inv_max = max(inv1, max(inv2, inv3));

	// Scale by maximum
	inv1 /= inv_max;
	inv2 /= inv_max;
	inv3 /= inv_max;

	// Scale input pixel values
	for (uint j = 0; j < H; j++)
		for (uint i = 0; i < W; i++)
		{
			pixel = in.at<Vec3b>(j, i);
			pixel[0] = pixel[0] * inv1;
			pixel[1] = pixel[1] * inv2;
			pixel[2] = pixel[2] * inv3;
			out.at<Vec3b>(j, i) = pixel;
		}
}


void AWB_SSE(const Mat& in, Mat& out)
{
	uint H  = in.rows,
	     W  = in.cols,
	     N  = H * W,
	     N3 = N * 3,
	     i;

	// Create output matrix
	float output[N] __attribute__((__aligned__(16)));
	out = Mat(in.size(), in.type(), output);

	// Get direct pointers for quick access
	const uchar* _in  = in.ptr<uchar>(0);
	      uchar* _out = out.ptr<uchar>(0);

	/**
	 * Calculate sum of pixel values per channel
	 */
	ulong sum1, sum2, sum3; // 64 bits wide
	ulong _sums[6] = {0, 0, 0, 0, 0, 0};

	__m128i val,
	        sums1 = _mm_set1_epi64x(0),
	        sums2 = _mm_set1_epi64x(0),
	        sums3 = _mm_set1_epi64x(0);

	for (i = 0; i < N3; i += 12)
	{
		val = _mm_set_epi64x(
				_in[i + 1] + _in[i + 4],
				_in[i] + _in[i + 3]);
		sums1 = _mm_add_epi64(sums1, val);

		val = _mm_set_epi64x(
				_in[i + 6] + _in[i + 9],
				_in[i + 2] + _in[i + 5]);
		sums2 = _mm_add_epi64(sums2, val);

		val = _mm_set_epi64x(
				_in[i + 8] + _in[i + 11],
				_in[i + 7] + _in[i + 10]);
		sums3 = _mm_add_epi64(sums3, val);
	}

	_mm_store_si128((__m128i*) &_sums[0], sums1);
	_mm_store_si128((__m128i*) &_sums[2], sums2);
	_mm_store_si128((__m128i*) &_sums[4], sums3);

	sum1 = _sums[0] + _sums[3];
	sum2 = _sums[1] + _sums[4];
	sum3 = _sums[2] + _sums[5];

	// Cleanup
	for (; i < N3; i += 3)
	{
		sum1 += _in[i];
		sum2 += _in[i + 1];
		sum3 += _in[i + 2];
	}

	// Find inverse of averages
	float inv1 = (float)N / (float)sum1,
	      inv2 = (float)N / (float)sum2,
	      inv3 = (float)N / (float)sum3;

	// Find maximum
	float inv_max = max(inv1, max(inv2, inv3));

	// Scale by maximum
	inv1 /= inv_max;
	inv2 /= inv_max;
	inv3 /= inv_max;

	// Scale input pixel values
	__m128 fv1, fv2, fv3, fv4,
	       scal1 = _mm_set_ps(inv1, inv3, inv2, inv1),
	       scal2 = _mm_set_ps(inv2, inv1, inv3, inv2),
	       scal3 = _mm_set_ps(inv3, inv2, inv1, inv3),
	       scal4 = _mm_set_ps(0.f, inv3, inv2, inv1);
	__m128i zeros = _mm_setzero_si128(),
		iv1, iv2, iv3, iv4, iv5, iv6,
		inv, outv;
	for (i = 0; i < N3; i += 15)
	{
		// Load 16 uchars
		inv = _mm_loadu_si128((__m128i*) &_in[i]);

		// Split into two vectors of 8 ushorts
		iv1 = _mm_unpacklo_epi8(inv, zeros);
		iv2 = _mm_unpackhi_epi8(inv, zeros);

		// Split into four vectors of 4 uints
		iv3 = _mm_unpacklo_epi16(iv1, zeros);
		iv4 = _mm_unpackhi_epi16(iv1, zeros);
		iv5 = _mm_unpacklo_epi16(iv2, zeros);
		iv6 = _mm_unpackhi_epi16(iv2, zeros);

		// Convert into four vectors of 4 floats
		fv1 = _mm_cvtepi32_ps(iv3);
		fv2 = _mm_cvtepi32_ps(iv4);
		fv3 = _mm_cvtepi32_ps(iv5);
		fv4 = _mm_cvtepi32_ps(iv6);

		// Multiply by scaling factors
		fv1 = _mm_mul_ps(fv1, scal1);
		fv2 = _mm_mul_ps(fv2, scal2);
		fv3 = _mm_mul_ps(fv3, scal3);
		fv4 = _mm_mul_ps(fv4, scal4);

		// Convert back into four vectors of 4 uints
		iv1 = _mm_cvtps_epi32(fv1);
		iv2 = _mm_cvtps_epi32(fv2);
		iv3 = _mm_cvtps_epi32(fv3);
		iv4 = _mm_cvtps_epi32(fv4);

		// Pack into two vectors of 8 ushorts
		iv1 = _mm_packus_epi32(iv1, iv2);
		iv2 = _mm_packus_epi32(iv3, iv4);

		// Pack into vector of 16 uchars
		outv = _mm_packus_epi16(iv1, iv2);

		// Store
		_mm_storeu_si128((__m128i*) &_out[i], outv);
	}
	// Cleanup
	for(; i < N3; i += 3)
	{
		_out[i]     = _in[i]     * inv1;
		_out[i + 1] = _in[i + 1] * inv2;
		_out[i + 2] = _in[i + 2] * inv3;
	}
}

