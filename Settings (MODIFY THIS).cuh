#ifndef __SETTINGS_CUH
#define __SETTINGS_CUH

/* To find what options are valid for the following settings, see the following file:*/
#include "Allowed Values for Settings.cuh"
#include "Test Data/Beta 1.6.cuh"
#include "Test Data/1.7.9.cuh"
#include "Test Data/1.8.9.cuh"
#include <chrono>

// The data to run the program on.
#define INPUT_DATA TEST_DATA_1_8_9__3
#define INPUT_DATA_LAYOUT TEST_DATA_1_8_9__3_LAYOUT

/* The number of "parts" this program should be broken up into, for usage by PART_TO_START_FROM.*/
constexpr uint64_t NUMBER_OF_PARTS = 1;
// The part to run. The only valid values for this are 1, 2, ..., NUMBER_OF_PARTS.
constexpr uint64_t PART_TO_START_FROM = 1;

// The number of devices (GPUs) to run this program on.
constexpr int32_t NUMBER_OF_DEVICES = 1;
// The number of worker threads per device.
constexpr uint64_t WORKERS_PER_DEVICE = 1ULL << 32;
// The number of workers per device block.
constexpr uint64_t WORKERS_PER_BLOCK = 256;

/* To speed up the program, states are run through a series of filters that gradually decrease the list of candidates.
   This is the maximum number of results to store per each filter.*/
constexpr uint64_t MAX_RESULTS_PER_FILTER = 200000000;
/* If true,  will treat blocks outside the input layout as all stone, speeding the program up but returning fewer results.
   If false, will treat blocks outside the input layout as unknown.*/
constexpr bool TREAT_COORDINATES_OUTSIDE_INPUT_AS_STONE = true;
/* If true, will beginning checking ...
   The program will be more likely to return results sooner, but it may also run slower.*/
constexpr bool START_IN_MIDDLE_OF_RANGE = true;
/* How frequently to print status updates.
   Disabled if set to zero.*/
constexpr std::chrono::seconds STATUS_FREQUENCY = std::chrono::seconds(30);
// const char *FILEPATH = "structure_seeds.txt";

// ~~~~~~~~
/* These are temporary settings that may possibly be automated in the future.*/

// The range of viable angle nextFloat values.
// TODO: Find a way to automatically calculate
// __device__ constexpr Pair<InclusiveRange<float>> ANGLE_BOUNDS = {{0.f, 0.2905f}, {0.7095f, 1.f}}; // TEST_DATA_1_8_9__1
__device__ constexpr Pair<InclusiveRange<float>> ANGLE_BOUNDS = {{0.115f, 0.296f}, {0.704f, 0.885f}}; // TEST_DATA_1_8_9__3
// __device__ constexpr Pair<InclusiveRange<float>> ANGLE_BOUNDS = {{0.f, 1.f}, {0.f, 1.f}};

// The maximum number of state advancements to check when recovering lowest-48-bits-of-worldseeds from 1.12- veins in Combination.cu.
constexpr uint64_t PRE_1_12_TEMP_MAX_POPULATION_CALLS = 1000;

#endif