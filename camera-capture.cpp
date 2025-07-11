#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>      // For std::string
#include <cstdint>     // For uint8_t, etc.
#include <exception>   // For std::exception

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

    while (true) {
        // Capture frame.
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Received empty frame." << endl;
            break;
        }

        // Convert to grayscale.
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        if (!gray.isContinuous()) {
            gray = gray.clone();
        }

        int width  = gray.cols;
        int height = gray.rows;

        // Wrap grayscale image into Halide buffer.
        Halide::Buffer<uint8_t> inputBuffer(gray.data, width, height);

        // Define ImageParam (symbolic representation of input image).
        Halide::ImageParam input(Halide::UInt(8), 2, "input");
        input.set(inputBuffer);

        // Variables representing spatial coordinates.
        Halide::Var x("x"), y("y"), x_outer, y_outer, x_inner, y_inner;

        int kernel_vals[3][3] = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };
        Halide::Buffer<int> kernelBuf(&kernel_vals[0][0], 3, 3);

        Halide::RDom r(0, 3, 0, 3);
        Halide::Func blur("blur");
        Halide::Expr val = Halide::cast<int32_t>(
            input(clampCoord(x + r.x - 1, width),
                clampCoord(y + r.y - 1, height))
        ) * kernelBuf(r.x, r.y);

        blur(x, y) = Halide::cast<uint8_t>(Halide::sum(val) / 16);        

        // Thresholding stage.
        Halide::Func thresholded("thresholded");
        thresholded(x, y) = Halide::cast<uint8_t>(
            Halide::select(blur(x, y) > 128, 255, 0)
        );

        // Parallelize
        //thresholded.parallel(y);

        // Apply tiling to divide computation into 64×64 tiles
        thresholded.tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
           .parallel(y_outer);

        // Compute blur within each tile explicitly to enhance cache efficiency
        //blur.compute_at(thresholded, x_outer);        

        // Realize pipeline.
        Halide::Buffer<uint8_t> outputBuffer;
        try {            
            outputBuffer = thresholded.realize({ width, height });
        } catch (const std::exception &e) {
            cerr << "Halide pipeline error: " << e.what() << endl;
            break;
        }

        // Wrap output in OpenCV Mat and display.
        Mat blurredThresholded(height, width, CV_8UC1, outputBuffer.data());
        imshow("Processed Image", blurredThresholded);

        // Wait for 30 ms (~33 FPS). Exit if any key is pressed.
        if (waitKey(30) >= 0) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}