#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cstdint>
#include <exception>

using namespace cv;
using namespace std;

static inline Halide::Expr clampCoord(Halide::Expr coord, int maxCoord) {
    return Halide::clamp(coord, 0, maxCoord - 1);
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera." << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Received empty frame." << endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        if (!gray.isContinuous()) {
            gray = gray.clone();
        }

        int width  = gray.cols;
        int height = gray.rows;

        Halide::Buffer<uint8_t> inputBuffer(gray.data, width, height);
        Halide::ImageParam input(Halide::UInt(8), 2, "input");
        input.set(inputBuffer);

        int kernel_vals[3][3] = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };
        Halide::Buffer<int> kernelBuf(&kernel_vals[0][0], 3, 3);

        Halide::Var x("x"), y("y"), xy("xy");
        Halide::RDom r(0, 3, 0, 3);

        Halide::Func blur("blur");
        Halide::Expr val = Halide::cast<int32_t>(
            input(clampCoord(x + r.x - 1, width),
                  clampCoord(y + r.y - 1, height))
        ) * kernelBuf(r.x, r.y);
        blur(x, y) = Halide::cast<uint8_t>(Halide::sum(val) / 16);

        Halide::Func thresholded("thresholded");
        thresholded(x, y) = Halide::cast<uint8_t>(Halide::select(blur(x, y) > 128, 255, 0));

        // Fuse
        thresholded.fuse(x, y, xy);
        blur.compute_at(thresholded, xy);

        Halide::Buffer<uint8_t> outputBuffer;
        try {
            outputBuffer = thresholded.realize({width, height}); // 2D output as usual
        } catch (const Halide::CompileError &e) {
            cerr << "Halide compile error: " << e.what() << endl;
            break;
        } catch (const std::exception &e) {
            cerr << "Halide pipeline error: " << e.what() << endl;
            break;
        }

        Mat blurredThresholded(height, width, CV_8UC1, outputBuffer.data());
        imshow("Processed Image (Fused)", blurredThresholded);

        if (waitKey(30) >= 0) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}