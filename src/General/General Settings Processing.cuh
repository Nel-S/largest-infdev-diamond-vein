#ifndef __GENERAL__SETTINGS_PROCESSING_CUH
#define __GENERAL__SETTINGS_PROCESSING_CUH

#include "..\..\Settings and Input Data (MODIFY THIS).cuh"
#include "Pseudorandom Number Generators.cuh"

constexpr int64_t BLOCKS_TO_FAR_LANDS = 12550821;
constexpr int64_t CHUNKS_PER_AXIS = (BLOCKS_TO_FAR_LANDS >> 4) - ((-BLOCKS_TO_FAR_LANDS) >> 4) + 1 - 1;
constexpr uint64_t TOTAL_CHUNKS = CHUNKS_PER_AXIS*CHUNKS_PER_AXIS;
constexpr uint64_t TOTAL_ITERATIONS = constexprCeil(static_cast<double>(TOTAL_CHUNKS)/static_cast<double>(WORKERS_PER_DEVICE));
constexpr uint64_t ITERATION_PARTS_OFFSET = constexprFloor(static_cast<double>(TOTAL_ITERATIONS) * static_cast<double>(constexprMin(constexprMax(PART_TO_START_FROM, UINT64_C(1)), NUMBER_OF_PARTS) - 1) / static_cast<double>(NUMBER_OF_PARTS));
constexpr uint64_t GLOBAL_ITERATIONS_NEEDED = TOTAL_ITERATIONS - ITERATION_PARTS_OFFSET;

constexpr uint64_t ACTUAL_STORAGE_CAPACITY = constexprMin(MAX_RESULTS_PER_FILTER, WORKERS_PER_DEVICE);
__device__ Pair<int32_t> STORAGE_ARRAY[ACTUAL_STORAGE_CAPACITY];
__managed__ size_t storageArraySize = 0;
__device__ Pair<int32_t> STORAGE_ARRAY_2[ACTUAL_STORAGE_CAPACITY];
__managed__ size_t storageArray2Size = 0;


#endif