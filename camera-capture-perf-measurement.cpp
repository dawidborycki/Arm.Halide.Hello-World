#include "Halide.h"
#include "HalideRuntime.h"          
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cstdint>
#include <exception>
#include <chrono>                    
#include <iomanip>                   

using namespace cv;
using namespace std;

// Clamp coordinate within [0, maxCoord - 1].
static inline Halide::Expr clampCoord(Halide::Expr coord, int maxCoord) {
    return Halide::clamp(coord, 0, maxCoord - 1);
}

int main() {
    // Open the default camera.
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera." << endl;
        return -1;
    }

    bool warmed_up = false;  // skip/report first-frame JIT separately

    while (true) {
        // Capture frame.
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Received empty frame." << endl;
            break;
        }
        if (!frame.isContinuous()) {
            frame = frame.clone();
        }

        int width    = frame.cols;
        int height   = frame.rows;
        int channels = frame.channels();   // typically 3 (BGR) or 4 (BGRA)

        // Wrap the interleaved BGR[BGR...] frame for Halide
        auto in_rt = Halide::Runtime::Buffer<uint8_t>::make_interleaved(
            frame.data, width, height, channels);
        Halide::Buffer<> inputBuffer(*in_rt.raw_buffer()); // front-end Buffer view

        // Define ImageParam for color input (x, y, c).
        Halide::ImageParam input(Halide::UInt(8), 3, "input");
        input.set(inputBuffer);

        const int C = frame.channels();          // 3 (BGR) or 4 (BGRA)
        input.dim(0).set_stride(C);              // x stride = channels (interleaved)
        input.dim(2).set_stride(1);              // c stride = 1 (adjacent bytes)
        input.dim(2).set_bounds(0, C);           // c in [0, C)

        // Define variables representing image coordinates.
        Halide::Var x("x"), y("y");

        // Grayscale in Halide (BGR order; ignore alpha if present)
        Halide::Func gray("gray");
        Halide::Expr r16 = Halide::cast<int16_t>(input(x, y, 2));
        Halide::Expr g16 = Halide::cast<int16_t>(input(x, y, 1));
        Halide::Expr b16 = Halide::cast<int16_t>(input(x, y, 0));

        // Integer approx: Y ≈ (77*R + 150*G + 29*B) >> 8
        gray(x, y) = Halide::cast<uint8_t>((77 * r16 + 150 * g16 + 29 * b16) >> 8);

        // Kernel layout: [1 2 1; 2 4 2; 1 2 1], sum = 16.
        int kernel_vals[3][3] = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };
        Halide::Buffer<int> kernelBuf(&kernel_vals[0][0], 3, 3);

        Halide::RDom r(0, 3, 0, 3);
        Halide::Func blur("blur");

        Halide::Expr val =
            Halide::cast<int16_t>( gray(clampCoord(x + r.x - 1, width),
                                        clampCoord(y + r.y - 1, height)) ) *
            Halide::cast<int16_t>( kernelBuf(r.x, r.y) );

        blur(x, y) = Halide::cast<uint8_t>(Halide::sum(val) / 16);

        // Thresholding stage
        Halide::Func thresholded("thresholded");
        thresholded(x, y) = Halide::cast<uint8_t>(
            Halide::select(blur(x, y) > 128, 255, 0)
        );

        // Schedule 
        //blur.compute_root().parallel(y);         // parallelize reduction across rows
        //thresholded.compute_root();              // cheap pixel-wise stage at root

        // Tiling 
        Halide::Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

        thresholded
            .tile(x, y, xo, yo, xi, yi, 128, 64)
            .vectorize(xi, 16)    
            .parallel(yo);

        blur
            .compute_at(thresholded, xo)
            .vectorize(x, 16);
            
        // // Cache RGB→gray per tile (reads interleaved input → keep unvectorized).
        // gray
        //     .compute_at(thresholded, xo)
        //     .store_at(thresholded, xo);

        // Performance timing around realize() only
        Halide::Buffer<uint8_t> outputBuffer;
        auto t0 = std::chrono::high_resolution_clock::now();

        try {
            outputBuffer = thresholded.realize({ width, height });
        } catch (const std::exception &e) {
            cerr << "Halide pipeline error: " << e.what() << endl;
            break;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // First frame includes JIT; mark it so you know why it's slower
        double fps = (ms > 0.0) ? 1000.0 / ms : 0.0;
        double mpixps = (ms > 0.0) ? (double(width) * double(height)) / (ms * 1000.0) : 0.0;

        std::cout << std::fixed << std::setprecision(2)
                  << (warmed_up ? "" : "[warm-up] ")
                  << "Halide realize: " << ms << " ms  |  "
                  << fps << " FPS  |  "
                  << mpixps << " MPix/s" << endl;

        warmed_up = true;

        // Wrap output in OpenCV Mat and display.
        Mat blurredThresholded(height, width, CV_8UC1, outputBuffer.data());
        imshow("Processed Image", blurredThresholded);

        // Wait for 30 ms (~33 FPS). Exit if any key is pressed.
        if (waitKey(30) >= 0) {
            break;
        }
    }

    std::cout << std::endl;
    cap.release();
    destroyAllWindows();
    return 0;
}