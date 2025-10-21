#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cstdint>
#include <exception>

using namespace Halide;
using namespace cv;
using namespace std;

int main() {
    // Open the default camera.
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera.\n";
        return -1;
    }

    // Grab one frame to determine dimensions and channels
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        cerr << "Error: empty first frame.\n";
        return -1;
    }

    // Ensure BGR 3-channel layout
    if (frame.channels() == 4)
        cvtColor(frame, frame, COLOR_BGRA2BGR);
    else if (frame.channels() == 1)
        cvtColor(frame, frame, COLOR_GRAY2BGR);
    if (!frame.isContinuous())
        frame = frame.clone();

    const int width  = frame.cols;
    const int height = frame.rows;
    const int ch     = frame.channels();

    // Build the pipeline once (outside the capture loop)
    ImageParam input(UInt(8), 3, "input");
    input.dim(0).set_stride(ch);  // interleaved: x stride = channels
    input.dim(2).set_stride(1);
    input.dim(2).set_bounds(0, 3);

    // Clamp borders
    Func inputClamped = BoundaryConditions::repeat_edge(input);

    // Grayscale conversion (Rec.601 weights)
    Var x("x"), y("y");
    Func gray("gray");
    gray(x, y) = cast<uint8_t>(0.114f * inputClamped(x, y, 0) +
                               0.587f * inputClamped(x, y, 1) +
                               0.299f * inputClamped(x, y, 2));

    // 3×3 binomial blur 
    Func blur("blur");
    const uint16_t k[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
    Expr sum = cast<uint16_t>(0);
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            sum += cast<uint16_t>(gray(x + i - 1, y + j - 1)) * k[j][i];
    blur(x, y) = cast<uint8_t>(sum / 16);

    // Threshold (binary)
    Func output("output");
    Expr T = cast<uint8_t>(128);
    output(x, y) = select(blur(x, y) > T, cast<uint8_t>(255), cast<uint8_t>(0));    

    // Allocate output buffer once
    Buffer<uint8_t> outBuf(width, height);

    // JIT compile once outside the loop
    Pipeline pipe(output);
    pipe.compile_jit();

    namedWindow("Processing Workflow", WINDOW_NORMAL);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        if (frame.channels() == 4)
            cvtColor(frame, frame, COLOR_BGRA2BGR);
        else if (frame.channels() == 1)
            cvtColor(frame, frame, COLOR_GRAY2BGR);
        if (!frame.isContinuous())
            frame = frame.clone();

        // Use Halide::Buffer::make_interleaved directly
        Buffer<uint8_t> inputBuf =
            Buffer<uint8_t>::make_interleaved(frame.data, frame.cols, frame.rows, frame.channels());

        input.set(inputBuf);

        try {
            pipe.realize(outBuf);
        } catch (const Halide::RuntimeError& e) {
            cerr << "Halide runtime error: " << e.what() << "\n";
            break;
        } catch (const std::exception& e) {
            cerr << "std::exception: " << e.what() << "\n";
            break;
        }

        // Display
        Mat view(height, width, CV_8UC1, outBuf.data());
        imshow("Processing Workflow", view);
        if (waitKey(1) >= 0) break;
    }

    destroyAllWindows();
    return 0;
}