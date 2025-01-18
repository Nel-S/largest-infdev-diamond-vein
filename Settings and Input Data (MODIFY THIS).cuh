#ifndef __SETTINGS_AND_INPUT_DATA_CUH
#define __SETTINGS_AND_INPUT_DATA_CUH

// To find what options are valid for the following settings, see the following files:
// General
#include "src/General/Allowed Values for General Settings.cuh"
// Veins
#include "src/Veins/Allowed Values for Input Data.cuh"

#include <chrono>
#include <memory>
#include <vector>

constexpr uint64_t MINIMUM_VOLUME = 0;

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
constexpr uint64_t MAX_RESULTS_PER_FILTER = 120000000;
/* How frequently to print status updates.
   Disabled if set to zero.*/
constexpr std::chrono::seconds STATUS_FREQUENCY = std::chrono::seconds(30);

/* If true, will beginning checking ...
   The program will be more likely to return results sooner, but it may also run slower.*/
constexpr bool START_IN_MIDDLE_OF_RANGE = true;

#endif