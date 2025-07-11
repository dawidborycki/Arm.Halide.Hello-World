#include "Halide.h"
#include <iostream>
using namespace Halide;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <output_basename> \n";
        return 1;
    }

    std::string output_basename = argv[1];

    // Configure Halide Target for Android
    Halide::Target target;
    target.os = Halide::Target::OS::Android; 
    target.arch = Halide::Target::Arch::ARM;
    target.bits = 64;
    target.set_feature(Target::NoRuntime, false);

    // --- Define the pipeline ---
    // Define variables
    Var x("x"), y("y");

    // Define input parameter
    ImageParam input(UInt(8), 2, "input");

    // Create a clamped function that limits the access to within the image bounds
    Func clamped = Halide::BoundaryConditions::repeat_edge(input);

    // Now use the clamped function in processing
    RDom r(0, 3, 0, 3);
    Func blur("blur");

    // Initialize blur accumulation
    blur(x, y) = cast<uint16_t>(0);
    blur(x, y) += cast<uint16_t>(clamped(x + r.x - 1, y + r.y - 1));

    // Then continue with pipeline
    Func blur_div("blur_div");
    blur_div(x, y) = cast<uint8_t>(blur(x, y) / 9);

    // Thresholding
    Func thresholded("thresholded");
    Expr t = cast<uint8_t>(128);
    thresholded(x, y) = select(blur_div(x, y) > t, cast<uint8_t>(255), cast<uint8_t>(0));

    // Simple scheduling 
    blur_div.compute_root();
    thresholded.compute_root();

    // --- AOT compile to a file ---
    thresholded.compile_to_static_library(
        output_basename,      // base filename
        { input },            // list of inputs
        "blur_threshold",     // name of the generated function
        target
    );

    return 0;
}