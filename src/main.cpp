#include <climits>
#include <fstream>

#include "boost/program_options.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "grayworld.hpp"
#include "util.hpp"

namespace po = boost::program_options;

bool BENCH = false;

int main(int argc, char** argv)
{
	/**
	 * Argument parsing
	 */
	po::options_description desc("Available options");
	desc.add_options()
	    ("help", "Show this message")
	    ("input-file,i", po::value<std::string>(), "Input file path")
	    ("output-file,o", po::value<std::string>(), "Output file path (default: output.png)")
	    ("headless", po::bool_switch()->default_value(false), "Run without graphical output")
	    ("benchmark", po::bool_switch()->default_value(false), "Benchmark various implementations")
	;

	po::positional_options_description p;
	p.add("input-file", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	po::notify(vm);

	if (vm.size() == 0 || vm.count("help") || !vm.count("input-file")) {
		std::cout << "Usage: " << argv[0]
			<< " [options] input-file" << std::endl
			<< desc;
		return 1;
	}

	if (vm["headless"].as<bool>()) GRAPHICAL = false;

	BENCH = vm["benchmark"].as<bool>();
	if (BENCH) GRAPHICAL = false;


	/**
	 * Read image file
	 */
	std::string const f = vm["input-file"].as<std::string>();
	cv::Mat const img = cv::imread(f, CV_LOAD_IMAGE_COLOR);
	if (!img.data) {
		throw std::runtime_error("Invalid input file: " + f);
		return -1;
	}
	showImage("Input Image", img);


	/**
	 * Validate AWB_SSE
	 */
	cv::Mat    out     = AWB(img);
	cv::Mat    out_sse = AWB_SSE(img);
	cv::Scalar diff    = cv::sum(out != out_sse);
	if (diff[0] + diff[1] + diff[2] > 0)
	{
		throw std::runtime_error("SSE implementation is wrong!");
		return -1;
	}

	/**
	 * Perform AWB
	 */

	if (BENCH)
	{
		uint max_steps = 5e2, nosse, sse;

		nosse = benchmark([img](){AWB(img);}, max_steps);
		printf("Naive version took %d cycles\n", nosse);

		sse = benchmark([img](){AWB_SSE(img);}, max_steps);
		printf("SSE version took %d cycles\n", sse);

		printf("> %.1fx speedup!\n", (float)nosse / (float)sse);
	} else {
		showImage("Output Image", out);
	}

	if (GRAPHICAL)
	{
		std::cout << "Press any key to quit..." << std::endl;
		cv::waitKey();
	}


	/**
	 * Write output file if necessary
	 */

	if (vm.count("output-file"))
	{
		auto opath = vm["output-file"].as<std::string>();
		std::cout << "Writing output to \"" << opath << "\"." << std::endl;
		imwrite(opath, out);
	}

	return 0;

}
