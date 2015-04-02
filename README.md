# Sample code for application to OpenCV, GSoC 2015

## Requirements

- OpenCV 2.4+
- Boost.Program\_options
- SSE4 capable processor


## Build

To build, run the following for Linux/UNIX:

    mkdir build
    cd build/

    cmake ..
    make

    ./grayworld ../lena.png --bench


## Usage

Use `--bench` to run a benchmark to compare the speed of the SSE-based
implementation with the naive implementation.

Use `--headless` to disable graphical output.

The default option is to simply show an auto-white-balanced image using
the gray world algorithm.

See `--help` for all available options.


## Speedup

The SSE version currently shows a speedup of up to 5.7x on an Intel Ivy Bridge
mobile processor compared to the naive implementation.

