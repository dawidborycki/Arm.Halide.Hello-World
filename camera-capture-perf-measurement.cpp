#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cstdint>
#include <exception>
#include <chrono>
#include <iomanip>

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
    if (frame.channels() == 4)      cvtColor(frame, frame, COLOR_BGRA2BGR);
    else if (frame.channels() == 1) cvtColor(frame, frame, COLOR_GRAY2BGR);
    if (!frame.isContinuous())      frame = frame.clone();

    const int width  = frame.cols;
    const int height = frame.rows;
    const int ch     = frame.channels();   // should be 3 now

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

    // After defining output(x,y):
    blur.compute_root().parallel(y);   // parallelize across scanlines
    output.compute_root();             // cheap pixel-wise stage at root (optional)

    // Baseline schedule: materialize gray; fuse blur+threshold into output
    gray.compute_root();

    // After defining: input, gray, blur, thresholded
    // Halide::Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

    // // Tile & parallelize the consumer; vectorize inner x on planar output.
    // output
    //     .tile(x, y, xo, yo, xi, yi, 128, 64)
    //     .vectorize(xi, 16)
    //     .parallel(yo);

    // // Compute blur inside each tile and vectorize its inner x.
    // blur
    //     .compute_at(output, xo)
    //     .vectorize(x, 16);

    // // Cache RGB→gray per tile (reads interleaved input → keep unvectorized).
    // gray
    //     .compute_at(output, xo)
    //     .store_at(output, xo);
        
    // Tiling (partitioning only)
    Halide::Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

    output
        .tile(x, y, xo, yo, xi, yi, 128, 64)  // try 128x64; tune per CPU
        .vectorize(xi, 16)                    // safe: planar, unit-stride along x
        .parallel(yo);                        // run tiles across cores

    blur
        .compute_at(output, xo)               // keep work tile-local
        .vectorize(x, 16);                    // vectorize planar blur
        
    // Allocate output buffer once & JIT once
    Buffer<uint8_t> outBuf(width, height);
    Pipeline pipe(output);
    pipe.compile_jit();

    namedWindow("Processing Workflow", WINDOW_NORMAL);

    bool warmed_up = false;
    for (;;) {
        cap >> frame;
        if (frame.empty()) break;
        if (frame.channels() == 4)      cvtColor(frame, frame, COLOR_BGRA2BGR);
        else if (frame.channels() == 1) cvtColor(frame, frame, COLOR_GRAY2BGR);
        if (!frame.isContinuous())      frame = frame.clone();

        // Use Halide::Buffer::make_interleaved directly
        Buffer<uint8_t> inputBuf =
            Buffer<uint8_t>::make_interleaved(frame.data, frame.cols, frame.rows, frame.channels());
        input.set(inputBuf);

        // Performance timing strictly around realize()
        auto t0 = chrono::high_resolution_clock::now();
        pipe.realize(outBuf);
        auto t1 = chrono::high_resolution_clock::now();

        double ms = chrono::duration<double, milli>(t1 - t0).count();
        double fps = ms > 0.0 ? 1000.0 / ms : 0.0;
        double mpixps = ms > 0.0 ? (double(width) * double(height)) / (ms * 1000.0) : 0.0;

        cout << fixed << setprecision(2)
             << (warmed_up ? "" : "[warm-up] ")
             << "realize: " << ms << " ms  |  "
             << fps << " FPS  |  "
             << mpixps << " MPix/s\r" << flush;
        warmed_up = true;

        // Display
        Mat view(height, width, CV_8UC1, outBuf.data());
        imshow("Processing Workflow", view);
        if (waitKey(1) >= 0) break;
    }

    cout << "\n";
    destroyAllWindows();
    return 0;
}