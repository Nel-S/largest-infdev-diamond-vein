#ifndef __SETTINGS_CUH
#define __SETTINGS_CUH

/* To find what options are valid for the following settings, see the following file:*/
#include "Allowed Values for Settings.cuh"
#include "Test Data/1.8.9.cuh"

// The data to run the program on.
// TODO: Bundle material, version, layout, and coordinates into a single struct
constexpr Material MATERIAL = Material::Dirt;
constexpr Version VERSION = Version::v1_8_9;
#define TRUE_VEIN TRUE_VEIN_1_8_9__3
#define VEIN_COORDINATE VEIN_COORDINATE_1_8_9__3

/* The number of "parts" this program should be broken up into, for usage by PART_TO_START_FROM.*/
constexpr uint64_t NUMBER_OF_PARTS = 1;
// The part to run. The only valid values for this are 1, 2, ..., NUMBER_OF_PARTS.
constexpr uint64_t PART_TO_START_FROM = 1;

// The number of devices (GPUs) to run this program on.
constexpr int32_t NUMBER_OF_DEVICES = 1;
// The number of worker threads per device.
constexpr uint64_t WORKERS_PER_DEVICE = 1ULL << 31;
// The number of workers per device block.
constexpr uint64_t WORKERS_PER_BLOCK = 256;

/* To speed up the program, states are run through a series of filters that gradually decrease the list of candidates.
   This is the maximum number of results to store per each filter.*/
constexpr uint64_t MAX_RESULTS_PER_FILTER = 200000000;
// How frequently to print status updates.
constexpr std::chrono::seconds STATUS_FREQUENCY = std::chrono::seconds(30);
// const char *FILEPATH = "structure_seeds.txt";

// ~~~~~~~~

// The range of viable angle nextFloat values.
// TODO: Find a way to automatically calculate
// constexpr InclusiveRange<float> ANGLE_BOUNDS(0.2905f, 0.7095f, true);
__device__ constexpr InclusiveRange<float> ANGLE_BOUNDS(0.f, 1.f);

#endif