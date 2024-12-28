#ifndef __SETTINGS_AND_INPUT_DATA_CUH
#define __SETTINGS_AND_INPUT_DATA_CUH

// To find what options are valid for the following settings, see the following files:
// General
#include "src/General/Allowed Values for General Settings.cuh"
// End Pillars
#include "src/End Pillars/Allowed Values for Input Data.cuh"
// Veins
#include "src/Veins/Allowed Values for Input Data.cuh"

#include <chrono>
#include <memory>
#include <vector>

/* ====================================================================
                             GENERAL SETTINGS
   ==================================================================== */

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
/* How frequently to print status updates.
   Disabled if set to zero.*/
constexpr std::chrono::seconds STATUS_FREQUENCY = std::chrono::seconds(30);
// const char *FILEPATH = "structure_seeds.txt";


std::vector<InputData*> INPUT_DATA;

/* =====================================================================
                               VEIN SETTINGS
   ===================================================================== */

/* To find what options are valid for the following settings, see the following file:*/

// The data to run the program on.
#define VEIN_INPUT_DATA VEINS_TEST_DATA_1_8_9__1
#define VEIN_INPUT_DATA_LAYOUT VEINS_TEST_DATA_1_8_9__1_LAYOUT

/* If true, will beginning checking ...
   The program will be more likely to return results sooner, but it may also run slower.*/
constexpr bool START_IN_MIDDLE_OF_RANGE = false;

// ~~~~~~~~
/* These are temporary settings that may possibly be automated in the future.*/

// The maximum number of state advancements to check when recovering lowest-48-bits-of-worldseeds from 1.12- veins in Combination.cu.
constexpr uint64_t PRE_1_12_TEMP_MAX_POPULATION_CALLS = 1000;


/* =====================================================================
                            END PILLAR SETTINGS
   ===================================================================== */

// The data to run the program on.
#define END_PILLAR_INPUT_DATA END_PILLARS_TEST_DATA_2

#endif