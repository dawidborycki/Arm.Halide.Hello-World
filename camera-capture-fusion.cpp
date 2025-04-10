#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <exception>

using namespace cv;
using namespace std;

// This function clamps the coordinate (coord) within [0, maxCoord - 1].
static inline Halide::Expr clampCoord(Halide::Expr coord, int maxCoord) {
    return Halide::clamp(coord, 0, maxCoord - 1);
}

int main() {
    // Open the default camera with OpenCV.
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera." << endl;
        return -1;
    }

    while (true) {
        // Capture a frame from the camera.
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Received empty frame." << endl;
            break;
        }

        // Convert the frame to grayscale.
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Ensure the grayscale image is continuous in memory.
        if (!gray.isContinuous()) {
            gray = gray.clone();
        }

        int width  = gray.cols;
        int height = gray.rows;

        // Create a simple 2D Halide buffer from the grayscale Mat.
        Halide::Buffer<uint8_t> inputBuffer(gray.data, width, height);

        // Create a Halide ImageParam for a 2D UInt(8) image.
        Halide::ImageParam input(Halide::UInt(8), 2, "input");
        input.set(inputBuffer);

        // Define variables for x (width) and y (height).
        Halide::Var x("x"), y("y");

        // Define a function that applies a 3x3 Gaussian blur.
        Halide::Func blur("blur");
        Halide::RDom r(0, 3, 0, 3);

        // Kernel layout: [1 2 1; 2 4 2; 1 2 1], sum = 16.
        Halide::Expr weight = Halide::select(
            (r.x == 1 && r.y == 1), 4,
            (r.x == 1 || r.y == 1), 2,
            1
        );

        Halide::Expr offsetX = x + (r.x - 1);
        Halide::Expr offsetY = y + (r.y - 1);

        // Manually clamp offsets to avoid out-of-bounds.
        Halide::Expr clampedX = clampCoord(offsetX, width);
        Halide::Expr clampedY = clampCoord(offsetY, height);

        // Accumulate weighted sum in 32-bit int before normalization.
        Halide::Expr val = Halide::cast<int>(input(clampedX, clampedY)) * weight;

        blur(x, y) = Halide::cast<uint8_t>(Halide::sum(val) / 16);

        // Add a thresholding stage on top of the blurred result.
        // If blur(x,y) > 128 => 255, else 0
        Halide::Func thresholded("thresholded");
        thresholded(x, y) = Halide::cast<uint8_t>(
            Halide::select(blur(x, y) > 128, 255, 0)
        );

        // Apply fusion scheduling
        blur.compute_at(thresholded, x);
        thresholded.parallel(y);        

        // Realize the thresholded function. Wrap in try-catch for error reporting.
        Halide::Buffer<uint8_t> outputBuffer;
        try {
            outputBuffer = thresholded.realize({ width, height });
        } catch (const std::exception &e) {
            cerr << "Halide pipeline error: " << e.what() << endl;
            break;
        }

        // Wrap the Halide output in an OpenCV Mat and display.
        Mat blurredThresholded(height, width, CV_8UC1, outputBuffer.data());

        imshow("Processed image", blurredThresholded);

        // Exit the loop if a key is pressed.
        if (waitKey(30) >= 0) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}