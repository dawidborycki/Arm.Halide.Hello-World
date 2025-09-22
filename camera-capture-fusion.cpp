#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstdint>
#include <exception>

using namespace Halide;
using namespace cv;
using namespace std;

enum class Schedule : int {
    Simple = 0,              // materialize gray + blur
    FuseBlurAndThreshold = 1,// materialize gray; fuse blur+threshold
    FuseAll = 2,             // fuse everything (default)
    Tile = 3,                // tile output; materialize gray per tile; blur fused
};

static const char* schedule_name(Schedule s) {
    switch (s) {
        case Schedule::Simple:               return "Simple";
        case Schedule::FuseBlurAndThreshold: return "FuseBlurAndThreshold";
        case Schedule::FuseAll:              return "FuseAll";
        case Schedule::Tile:                 return "Tile";
        default:                              return "Unknown";
    }
}

// Build the BGR->Gray -> 3x3 binomial blur -> threshold pipeline.
// We clamp the *ImageParam* at the borders (Func clamp of ImageParam works in Halide 19).
Pipeline make_pipeline(ImageParam& input, Schedule schedule) {
    Var x("x"), y("y");

    // Assume 3-channel BGR interleaved frames (we convert if needed).
    input.dim(0).set_stride(3);      // x-stride = channels
    input.dim(2).set_stride(1);      // c-stride = 1
    input.dim(2).set_bounds(0, 3);   // three channels

    Func inputClamped = BoundaryConditions::repeat_edge(input);

    // Gray (Rec.601)
    Func gray("gray");
    gray(x, y) = cast<uint8_t>(0.114f * inputClamped(x, y, 0)
                             + 0.587f * inputClamped(x, y, 1)
                             + 0.299f * inputClamped(x, y, 2));

    // 3x3 binomial blur (sum/16)
    Func blur("blur");
    const uint16_t k[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
    Expr blurSum = cast<uint16_t>(0);
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            blurSum = blurSum + cast<uint16_t>(gray(x + i - 1, y + j - 1)) * k[j][i];
    blur(x, y) = cast<uint8_t>(blurSum / 16);

    // Threshold (binary)
    Func thresholded("thresholded");
    Expr T = cast<uint8_t>(128);
    thresholded(x, y) = select(blur(x, y) > T, cast<uint8_t>(255), cast<uint8_t>(0));

    // Final output
    Func output("output");
    output(x, y) = thresholded(x, y);
    output.compute_root(); // we always realize 'output'

    // Scheduling to demonstrate OPERATOR FUSION vs MATERIALIZATION
    // Default in Halide = fusion/inlining (no schedule on producers).
    Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

    switch (schedule) {
        case Schedule::Simple:
            // Materialize gray and blur (two loop nests); thresholded fuses into output
            gray.compute_root();
            blur.compute_root();
            break;

        case Schedule::FuseBlurAndThreshold:
            // Materialize gray; blur and thresholded remain fused into output
            gray.compute_root();
            break;

        case Schedule::FuseAll:
            // No schedule on producers: gray, blur, thresholded all fuse into output
            break;

        case Schedule::Tile:
            // Tile the output; compute gray per tile; blur stays fused within tile
            output.tile(x, y, xo, yo, xi, yi, 64, 64);
            gray.compute_at(output, xo);
            break;
    }

    // (Optional) Print loop nest once to “x-ray” the schedule
    std::cout << "\n---- Loop structure (" << schedule_name(schedule) << ") ----\n";
    output.print_loop_nest();
    std::cout << "-----------------------------------------------\n";

    return Pipeline(output);
}

int main(int argc, char** argv) {
    // Optional CLI: start with a given schedule number 0..3
    Schedule current = Schedule::FuseAll;
    if (argc >= 2) {
        int s = std::atoi(argv[1]);
        if (s >= 0 && s <= 3) current = static_cast<Schedule>(s);
    }
    std::cout << "Starting with schedule: " << schedule_name(current)
              << " (press 0..3 to switch; q/Esc to quit)\n";

    // Open camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera.\n";
        return 1;
    }
    cap.set(CAP_PROP_CONVERT_RGB, true); // ask OpenCV for BGR frames

    // Grab one frame to get size/channels
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Error: empty first frame.\n";
        return 1;
    }
    if (frame.channels() == 4) {
        cvtColor(frame, frame, COLOR_BGRA2BGR);
    } else if (frame.channels() == 1) {
        cvtColor(frame, frame, COLOR_GRAY2BGR);
    }
    if (!frame.isContinuous()) frame = frame.clone();

    const int width  = frame.cols;
    const int height = frame.rows;

    // Halide inputs/outputs
    ImageParam input(UInt(8), 3, "input");
    Buffer<uint8_t, 2> outBuf(width, height, "out");

    // Build pipeline for the starting schedule
    Pipeline pipe = make_pipeline(input, current);

    bool warmed_up = false;
    namedWindow("Fusion Demo (live)", WINDOW_NORMAL);

    for (;;) {
        cap >> frame;
        if (frame.empty()) break;
        if (frame.channels() == 4) {
            cvtColor(frame, frame, COLOR_BGRA2BGR);
        } else if (frame.channels() == 1) {
            cvtColor(frame, frame, COLOR_GRAY2BGR);
        }
        if (!frame.isContinuous()) frame = frame.clone();

        // Wrap interleaved frame
        auto in_rt = Runtime::Buffer<uint8_t>::make_interleaved(
            frame.data, frame.cols, frame.rows, /*channels*/3);
        Buffer<> in_fe(*in_rt.raw_buffer());
        input.set(in_fe);

        // Time the Halide realize() only
        auto t0 = std::chrono::high_resolution_clock::now();
        try {
            pipe.realize(outBuf);
        } catch (const Halide::RuntimeError& e) {
            std::cerr << "Halide runtime error: " << e.what() << "\n";
            break;
        } catch (const std::exception& e) {
            std::cerr << "std::exception: " << e.what() << "\n";
            break;
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double fps = ms > 0.0 ? 1000.0 / ms : 0.0;
        double mpixps = ms > 0.0 ? (double(width) * double(height)) / (ms * 1000.0) : 0.0;

        std::cout << std::fixed << std::setprecision(2)
                  << (warmed_up ? "" : "[warm-up] ")
                  << schedule_name(current) << " | "
                  << ms << " ms  |  "
                  << fps << " FPS  |  "
                  << mpixps << " MPix/s\r" << std::flush;
        warmed_up = true;

        // Show result
        Mat view(height, width, CV_8UC1, outBuf.data());
        imshow("Fusion Demo (live)", view);
        int key = waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;

        // Hotkeys 0..3 to switch schedules live
        if (key >= '0' && key <= '3') {
            Schedule next = static_cast<Schedule>(key - '0');
            if (next != current) {
                std::cout << "\nSwitching to schedule: " << schedule_name(next) << "\n";
                current = next;
                try {
                    pipe = make_pipeline(input, current); // rebuild JIT with new schedule
                } catch (const Halide::CompileError& e) {
                    std::cerr << "Halide compile error: " << e.what() << "\n";
                    break;
                }
                warmed_up = false; // next frame includes JIT, label as warm-up
            }
        }
    }

    std::cout << "\n";
    destroyAllWindows();
    return 0;
}