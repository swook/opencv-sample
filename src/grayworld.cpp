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
	float inv1 = sum1 == 0 ? 0.f : (float)N / (float)sum1,
	      inv2 = sum2 == 0 ? 0.f : (float)N / (float)sum2,
	      inv3 = sum3 == 0 ? 0.f : (float)N / (float)sum3;

	// Find maximum
	float inv_max = max(inv1, max(inv2, inv3));

	// Scale by maximum
	if (inv_max > 0)
	{
		inv1 /= inv_max;
		inv2 /= inv_max;
		inv3 /= inv_max;
	}

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
	__m128i zeros = _mm_setzero_si128(),
		inv,
		iv1, iv2, iv3, iv4, iv5, iv6,
		sv1 = _mm_setzero_si128(),
		sv2 = _mm_setzero_si128(),
		sv3 = _mm_setzero_si128(),
		sv4 = _mm_setzero_si128();

	for (i = 0; i < N3; i += 15)
	{
		// Load 16 x 8bit uchars
		inv = _mm_loadu_si128((__m128i*) &_in[i]);

		// Split into two vectors of 8 ushorts
		iv1 = _mm_unpacklo_epi8(inv, zeros);
		iv2 = _mm_unpackhi_epi8(inv, zeros);

		// Split into four vectors of 4 uints
		iv3 = _mm_unpacklo_epi16(iv1, zeros);
		iv4 = _mm_unpackhi_epi16(iv1, zeros);
		iv5 = _mm_unpacklo_epi16(iv2, zeros);
		iv6 = _mm_unpackhi_epi16(iv2, zeros);


		// Add to accumulators
		sv1 = _mm_add_epi32(sv1, iv3);
		sv2 = _mm_add_epi32(sv2, iv4);
		sv3 = _mm_add_epi32(sv3, iv5);
		sv4 = _mm_add_epi32(sv4, iv6);
	}

	// Store accumulated values into memory
	uint sums[16];
	_mm_store_si128((__m128i*) &sums[0],  sv1);
	_mm_store_si128((__m128i*) &sums[4],  sv2);
	_mm_store_si128((__m128i*) &sums[8],  sv3);
	_mm_store_si128((__m128i*) &sums[12], sv4);

	// Perform final reduction
	ulong sum1 = sums[0] + sums[3] + sums[6] + sums[9],
	      sum2 = sums[1] + sums[4] + sums[7] + sums[10],
	      sum3 = sums[2] + sums[5] + sums[8] + sums[11];

	// Cleanup
	for (; i < N3; i += 3)
	{
		sum1 += _in[i];
		sum2 += _in[i + 1];
		sum3 += _in[i + 2];
	}

	// Find inverse of averages
	double dinv1 = sum1 == 0 ? 0.f : (float)N / (float)sum1,
	       dinv2 = sum2 == 0 ? 0.f : (float)N / (float)sum2,
	       dinv3 = sum3 == 0 ? 0.f : (float)N / (float)sum3;

	// Find maximum
	double inv_max = max(dinv1, max(dinv2, dinv3));

	// Scale by maximum
	if (inv_max > 0)
	{
		dinv1 /= inv_max;
		dinv2 /= inv_max;
		dinv3 /= inv_max;
	}

	// Convert to floats
	float inv1 = (float) dinv1,
	      inv2 = (float) dinv2,
	      inv3 = (float) dinv3;

	// Scale input pixel values
	__m128 fv1, fv2, fv3, fv4,
	       scal1 = _mm_set_ps(inv1, inv3, inv2, inv1),
	       scal2 = _mm_set_ps(inv2, inv1, inv3, inv2),
	       scal3 = _mm_set_ps(inv3, inv2, inv1, inv3),
	       scal4 = _mm_set_ps(0.f, inv3, inv2, inv1);
	__m128i outv;
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

