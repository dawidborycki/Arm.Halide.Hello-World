#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>       
#include <cstdint>      

using namespace Halide;
using namespace cv;

int main() {
    // Static path for the input image.
    std::string imagePath = "img.png";

    // Load the input image using OpenCV (BGR by default).
    Mat input = imread(imagePath, IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error: Unable to load image from " << imagePath << std::endl;
        return -1;
    }
        
    // Convert from BGR to RGB (Halide expects RGB order).
    cvtColor(input, input, COLOR_BGR2RGB);

    // Wrap the OpenCV Mat data in a Halide::Buffer.
    // Dimensions: (width, height, channels)
    Buffer<uint8_t> inputBuffer(input.data, input.cols, input.rows, input.channels());

    // Create an ImageParam for symbolic indexing.
    ImageParam inputImage(UInt(8), 3);
    inputImage.set(inputBuffer);

    // Define Halide pipeline variables:
    // x, y - spatial coordinates (width, height)
    // c    - channel coordinate (R, G, B)
    Var x("x"), y("y"), c("c");
    Func invert("invert");
    invert(x, y, c) = 255 - inputImage(x, y, c);

    // Schedule the pipeline so that the channel dimension is the innermost loop,
    // ensuring that the output is interleaved.
    invert.reorder(c, x, y);

    // Realize the output buffer with the same dimensions as the input.
    Buffer<uint8_t> outputBuffer = invert.realize({input.cols, input.rows, input.channels()});

    // Wrap the Halide output buffer directly into an OpenCV Mat header.
    // CV_8UC3 indicates an 8-bit unsigned integer image (CV_8U) with 3 color channels (C3), typically representing RGB or BGR images.
    // This does not copy data; it creates a header that refers to the same memory.        
    Mat output(input.rows, input.cols, CV_8UC3, outputBuffer.data());

    // Convert RGB back to BGR for proper display in OpenCV.
    cvtColor(output, output, COLOR_RGB2BGR);

    // Display the input and processed image.
    imshow("Original Image", input);
    imshow("Inverted Image", output);

    // Wait indefinitely until a key is pressed.
    waitKey(0); // Wait for a key press before closing the window.

    return 0;
}