#ifndef __GENERATE_AND_VALIDATE_DATA_CUH
#define __GENERATE_AND_VALIDATE_DATA_CUH

#include "..\Settings (MODIFY THIS).cuh"
#include "Veins Logic.cuh"

// The length of the input vein along each axis.
__device__ constexpr struct Coordinate INPUT_LENGTHS = {
	static_cast<int32_t>(sizeof(**TRUE_VEIN)/sizeof(***TRUE_VEIN)),
	static_cast<int32_t>(sizeof(  TRUE_VEIN)/sizeof(  *TRUE_VEIN)),
	static_cast<int32_t>(sizeof( *TRUE_VEIN)/sizeof( **TRUE_VEIN))
};

// TODO: Will ultimately be replaced with VERSION <= v1.12.2
constexpr bool USE_POPULATION_OFFSET = true;

constexpr Pair<Coordinate> WORLD_BOUNDS = getWorldBounds(VERSION);
// Ensure the vein falls within the boundaries of the world
// static_assert(WORLD_BOUNDS.first.x <= VEIN_COORDINATE.x && VEIN_COORDINATE.x + INPUT_LENGTHS.x - 1 <= WORLD_BOUNDS.second.x, "Error: Data stretches beyond the world's boundaries in the x-direction.");
// static_assert(WORLD_BOUNDS.first.y <= VEIN_COORDINATE.y && VEIN_COORDINATE.y + INPUT_LENGTHS.y - 1 <= WORLD_BOUNDS.second.y, "Error: Data stretches beyond the world's boundaries in the y-direction.");
// static_assert(WORLD_BOUNDS.first.z <= VEIN_COORDINATE.z && VEIN_COORDINATE.z + INPUT_LENGTHS.z - 1 <= WORLD_BOUNDS.second.z, "Error: Data stretches beyond the world's boundaries in the z-direction.");

constexpr int32_t DIRT_SIZE = getVeinSize(MATERIAL, VERSION);
constexpr Pair<Coordinate> MAX_DIRT_DISPLACEMENT = getMaxVeinDisplacement(MATERIAL, VERSION);

// The bounding box the generation point can lay within, based on the dimensions of the vein alone.
__device__ constexpr Pair<Coordinate> VEIN_GENERATION_POINT_BOUNDING_BOX = {
	{
		VEIN_COORDINATE.x + (INPUT_LENGTHS.x - 1) - MAX_DIRT_DISPLACEMENT.second.x,
		// The generation point can't lie below the minimum y-coordinate
		constexprMax(VEIN_COORDINATE.y + (INPUT_LENGTHS.y - 1) - MAX_DIRT_DISPLACEMENT.second.y, WORLD_BOUNDS.first.y),
		VEIN_COORDINATE.z + (INPUT_LENGTHS.z - 1) - MAX_DIRT_DISPLACEMENT.second.z
	}, {
		VEIN_COORDINATE.x - MAX_DIRT_DISPLACEMENT.first.x,
		// The generation point can't lie above the maximum y-coordinate
		constexprMin(VEIN_COORDINATE.y - MAX_DIRT_DISPLACEMENT.first.y, WORLD_BOUNDS.second.y),
		VEIN_COORDINATE.z - MAX_DIRT_DISPLACEMENT.first.z
	}
};
static_assert(VEIN_GENERATION_POINT_BOUNDING_BOX.first.x <= VEIN_GENERATION_POINT_BOUNDING_BOX.second.x, "Error: Data results in an impossible generation point bounding box in the x-direction.");
static_assert(VEIN_GENERATION_POINT_BOUNDING_BOX.first.y <= VEIN_GENERATION_POINT_BOUNDING_BOX.second.y, "Error: Data results in an impossible generation point bounding box in the y-direction.");
static_assert(VEIN_GENERATION_POINT_BOUNDING_BOX.first.z <= VEIN_GENERATION_POINT_BOUNDING_BOX.second.z, "Error: Data results in an impossible generation point bounding box in the z-direction.");

// The number of possible chunks that could have originally generated the vein.
constexpr Pair<int32_t> NUMBER_OF_POSSIBLE_ORIGIN_CHUNKS = {
	((VEIN_GENERATION_POINT_BOUNDING_BOX.second.x - 8*USE_POPULATION_OFFSET) >> 4) - ((VEIN_GENERATION_POINT_BOUNDING_BOX.first.x - 8*USE_POPULATION_OFFSET) >> 4) + 1,
	((VEIN_GENERATION_POINT_BOUNDING_BOX.second.z - 8*USE_POPULATION_OFFSET) >> 4) - ((VEIN_GENERATION_POINT_BOUNDING_BOX.first.z - 8*USE_POPULATION_OFFSET) >> 4) + 1
};
constexpr int32_t TOTAL_NUMBER_OF_POSSIBLE_CHUNKS = NUMBER_OF_POSSIBLE_ORIGIN_CHUNKS.first * NUMBER_OF_POSSIBLE_ORIGIN_CHUNKS.second;
static_assert(1 <= NUMBER_OF_POSSIBLE_ORIGIN_CHUNKS.first, "Error: Data results in a non-positive number of possible origin chunks in the x-direction.");
static_assert(1 <= NUMBER_OF_POSSIBLE_ORIGIN_CHUNKS.second, "Error: Data results in a non-positive number of possible origin chunks in the z-direction.");
static_assert(1 <= TOTAL_NUMBER_OF_POSSIBLE_CHUNKS, "Error: Data results in an impossible number of possible origin chunks.");

struct ChunksToExamine {
	// The (block) coordinates of each possible origin chunk.
	Pair<int32_t> coordinates[TOTAL_NUMBER_OF_POSSIBLE_CHUNKS];
	// The range of possible nextInt values for the x-, y-, and z-directions if the vein had originated in that chunk.
	Pair<Coordinate> generationPointNextIntRanges[TOTAL_NUMBER_OF_POSSIBLE_CHUNKS];
	
	constexpr ChunksToExamine() : coordinates(), generationPointNextIntRanges() {
		for (int32_t i = 0; i < TOTAL_NUMBER_OF_POSSIBLE_CHUNKS; ++i) {
			coordinates[i] = {
				16*(((VEIN_GENERATION_POINT_BOUNDING_BOX.first.x - 8*USE_POPULATION_OFFSET) >> 4) + (i % NUMBER_OF_POSSIBLE_ORIGIN_CHUNKS.first)),
				16*(((VEIN_GENERATION_POINT_BOUNDING_BOX.first.z - 8*USE_POPULATION_OFFSET) >> 4) + (i / NUMBER_OF_POSSIBLE_ORIGIN_CHUNKS.first))
			};
			generationPointNextIntRanges[i] = {
				{
					constexprMax((VEIN_GENERATION_POINT_BOUNDING_BOX.first.x - 8*USE_POPULATION_OFFSET) - coordinates[i].first, 0),
					// y-bound was already capped when calculating the original bounding box
					VEIN_GENERATION_POINT_BOUNDING_BOX.first.y,
					constexprMax((VEIN_GENERATION_POINT_BOUNDING_BOX.first.z - 8*USE_POPULATION_OFFSET) - coordinates[i].second, 0)
				},
				{
					constexprMin((VEIN_GENERATION_POINT_BOUNDING_BOX.second.x - 8*USE_POPULATION_OFFSET) - coordinates[i].first, 15),
					// y-bound was already capped when calculating the original bounding box
					VEIN_GENERATION_POINT_BOUNDING_BOX.second.y,
					constexprMin((VEIN_GENERATION_POINT_BOUNDING_BOX.second.z - 8*USE_POPULATION_OFFSET) - coordinates[i].second, 15)
				}
			};
		}
	}
};

__device__ constexpr ChunksToExamine CHUNKS_TO_EXAMINE;
// constexpr uint64_t GLOBAL_ITERATIONS_NEEDED = constexprCeil(static_cast<double>(static_cast<uint64_t>(VEIN_GENERATION_POINT_BOUNDING_BOX.second.y - VEIN_GENERATION_POINT_BOUNDING_BOX.first.y + 1) << 40)/(static_cast<double>(WORKERS_PER_DEVICE)*static_cast<double>(NUMBER_OF_PARTS)));
// constexpr uint64_t ITERATION_PARTS_OFFSET = constexprFloor(static_cast<double>(GLOBAL_ITERATIONS_NEEDED) * static_cast<double>(constexprMin(constexprMax(PART_TO_START_FROM, UINT64_C(1)), NUMBER_OF_PARTS) - 1) / static_cast<double>(NUMBER_OF_PARTS));
constexpr uint64_t TOTAL_ITERATIONS = constexprCeil(static_cast<double>(static_cast<uint64_t>(VEIN_GENERATION_POINT_BOUNDING_BOX.second.y - VEIN_GENERATION_POINT_BOUNDING_BOX.first.y + 1) << 40)/static_cast<double>(WORKERS_PER_DEVICE));
constexpr uint64_t ITERATION_PARTS_OFFSET = constexprFloor(static_cast<double>(TOTAL_ITERATIONS) * static_cast<double>(constexprMin(constexprMax(PART_TO_START_FROM, UINT64_C(1)), NUMBER_OF_PARTS) - 1) / static_cast<double>(NUMBER_OF_PARTS));
constexpr uint64_t GLOBAL_ITERATIONS_NEEDED = TOTAL_ITERATIONS - ITERATION_PARTS_OFFSET;

constexpr uint64_t ACTUAL_STORAGE_CAPACITY = constexprMin(MAX_RESULTS_PER_FILTER, WORKERS_PER_DEVICE);
__device__ uint64_t STORAGE_ARRAY[ACTUAL_STORAGE_CAPACITY];
__managed__ size_t storageArraySize = 0;

#endif