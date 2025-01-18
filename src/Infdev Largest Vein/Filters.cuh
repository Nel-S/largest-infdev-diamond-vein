#ifndef __VEINS__FILTERS_CUH
#define __VEINS__FILTERS_CUH

#include "..\General\General Settings Processing.cuh"
#include "src\Veins\Underlying Logic.cuh"
#include <mutex>

uint64_t globalCurrentIteration = 0;
std::mutex mutex;

// Tests if a chunk, its East neighbor, its South neighbor, and its Southeast neighbor can all generate diamond veins.
__global__ void veinFilter1(uint64_t iterationRangeStart) {
	uint64_t index = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + static_cast<uint64_t>(threadIdx.x) + iterationRangeStart;
	Pair<int32_t> chunkCoordinates = {
		static_cast<int32_t>(static_cast<int64_t>(index / CHUNKS_PER_AXIS) + ((-BLOCKS_TO_FAR_LANDS) >> 4)),
		static_cast<int32_t>(static_cast<int64_t>(index % CHUNKS_PER_AXIS) + ((-BLOCKS_TO_FAR_LANDS) >> 4))
	};
	// if (abs(chunkCoordinates.first << 4) >= 1000 || abs(chunkCoordinates.second << 4) >= 1000) return;
	Random random;
	// 1/8th chance
	if (!veinCanGenerateDiamonds(chunkCoordinates, random)) return;

	// 1/8^2 = 1/64th chance
	// if (!veinCanGenerateDiamonds({chunkCoordinates.first + 1, chunkCoordinates.second}, random)) return;

	// 1/8^3 = 1/512th chance
	// if (!veinCanGenerateDiamonds({chunkCoordinates.first, chunkCoordinates.second + 1}, random)) return;

	if (!veinCanGenerateDiamonds({chunkCoordinates.first + 1, chunkCoordinates.second}, random) && !veinCanGenerateDiamonds({chunkCoordinates.first, chunkCoordinates.second + 1}, random)) return;

	// 1/8^4 = 1/4096th chance
	// if (!veinCanGenerateDiamonds({chunkCoordinates.first + 1, chunkCoordinates.second + 1}, random)) return;

	size_t arrayIndex = atomicAdd(&storageArraySize, 1);
	if (storageArraySize >= MAX_RESULTS_PER_FILTER) return;
	STORAGE_ARRAY[arrayIndex] = chunkCoordinates;
}

// Tests if a chunk, its East neighbor, its South neighbor, and its Southeast neighbor's diamond veins could all overlap.
__global__ void veinFilter2() {
	size_t arrayIndex = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
	if (storageArraySize <= arrayIndex) return;
	Pair<int32_t> chunkCoordinates = STORAGE_ARRAY[arrayIndex];

	Random random;
	// Called solely to seed and advance the random instance
	veinCanGenerateDiamonds(chunkCoordinates, random);
	Coordinate generationPoint = getGenerationPoint(chunkCoordinates, random);
	Pair<Coordinate> boundingBox = getMaxVeinBlockDisplacement(VeinMaterial::Diamond, Version::Infdev_20100617_1, generationPoint);
	// If bounding box is too far Northwest to possibly intersect the East or South chunks, skip
	if ((generationPoint.x % 16) + boundingBox.second.x < 16 + MAX_DIAMOND_BOUNDING_BOX_DISPLACEMENT.first.x - 1
	 && (generationPoint.z % 16) + boundingBox.second.z < 16 + MAX_DIAMOND_BOUNDING_BOX_DISPLACEMENT.first.z - 1) return;

	// printf("%" PRId32 "\t%" PRId32 "\t%" PRId32 "\n", generationPoint.x, generationPoint.y, generationPoint.z);

	Pair<int32_t> chunkCoordinatesEast = {chunkCoordinates.first + 1, chunkCoordinates.second};
	// Called solely to seed and advance the random instance
	veinCanGenerateDiamonds(chunkCoordinatesEast, random);
	Coordinate generationPointEast = getGenerationPoint(chunkCoordinatesEast, random);
	Pair<Coordinate> boundingBoxEast = getMaxVeinBlockDisplacement(VeinMaterial::Diamond, Version::Infdev_20100617_1, generationPointEast);

	Pair<int32_t> chunkCoordinatesSouth = {chunkCoordinates.first, chunkCoordinates.second + 1};
	// Called solely to seed and advance the random instance
	veinCanGenerateDiamonds(chunkCoordinatesSouth, random);
	Coordinate generationPointSouth = getGenerationPoint(chunkCoordinatesSouth, random);
	Pair<Coordinate> boundingBoxSouth = getMaxVeinBlockDisplacement(VeinMaterial::Diamond, Version::Infdev_20100617_1, generationPointSouth);

	if ((generationPointEast.x  % 16) - boundingBoxEast.first.x  > -1 + MAX_DIAMOND_BOUNDING_BOX_DISPLACEMENT.second.x + 1
	 && (generationPointSouth.z % 16) - boundingBoxSouth.first.z > -1 + MAX_DIAMOND_BOUNDING_BOX_DISPLACEMENT.second.z + 1) return;

	// If bounding box is too far Northeast to possibly intersect the West or South chunks, skip
	// printf("(%" PRId32 "\t%" PRId32 "\t%" PRId32 ")\t= (%" PRId32 "\t%" PRId32  ")\n", generationPointEast.x, generationPointEast.y, generationPointEast.z, generationPointEast.x % 16, generationPointEast.z % 16);
	// if ((generationPointEast.x % 16) - boundingBoxEast.first.x  > -1 + MAX_DIAMOND_BOUNDING_BOX_DISPLACEMENT.second.x + 1
	//  && (generationPointEast.z % 16) + boundingBoxEast.second.z < 16 + MAX_DIAMOND_BOUNDING_BOX_DISPLACEMENT.first.z  - 1) return;

	// printf("%" PRId32 "\t%" PRId32 "\t%" PRId32 "\n", generationPointEast.x, generationPointEast.y, generationPointEast.z);

	// // If bounding box is too far Southwest to possibly intersect the East or North chunks, skip
	// if ((generationPointSouth.x % 16) + boundingBoxSouth.second.x < 16 + MAX_DIAMOND_BOUNDING_BOX_DISPLACEMENT.first.x  - 1
	//  && (generationPointSouth.z % 16) - boundingBoxSouth.first.z  > -1 + MAX_DIAMOND_BOUNDING_BOX_DISPLACEMENT.second.z + 1) return;

	// Pair<int32_t> chunkCoordinatesSoutheast = {chunkCoordinates.first + 1, chunkCoordinates.second + 1};
	// // Called solely to seed and advance the random instance
	// veinCanGenerateDiamonds(chunkCoordinatesSoutheast, random);
	// Coordinate generationPointSoutheast = getGenerationPoint(chunkCoordinatesSoutheast, random);
	// Pair<Coordinate> boundingBoxSoutheast = getMaxVeinBlockDisplacement(VeinMaterial::Diamond, Version::Infdev_20100617_1, generationPointSoutheast);
	// // If bounding box is too far Southeast to possibly intersect the West or North chunks, skip
	// if ((generationPointSoutheast.x % 16) - boundingBoxSoutheast.first.x > -1 + MAX_DIAMOND_BOUNDING_BOX_DISPLACEMENT.second.x + 1
	//  && (generationPointSoutheast.z % 16) - boundingBoxSoutheast.first.z > -1 + MAX_DIAMOND_BOUNDING_BOX_DISPLACEMENT.second.z + 1) return;

	// TODO: Get combined volume of bounding boxes
	uint64_t volume = 0;
	if (volume < MINIMUM_VOLUME) return;

	// arrayIndex = atomicAdd(&storageArray2Size, 1);
	// if (storageArray2Size >= MAX_RESULTS_PER_FILTER) return;
	// STORAGE_ARRAY_2[arrayIndex] = chunkCoordinates;
	printf("%" PRId32 "\t%" PRId32 "\t%" PRId32 "\t%" PRIu64 "\n", generationPoint.x, generationPoint.y, generationPoint.z, volume);
}

// __global__ void veinFilter3() {
// 	size_t arrayIndex = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
// 	if (storageArraySize <= arrayIndex) return;
// 	Pair<int32_t> chunkCoordinates = STORAGE_ARRAY[arrayIndex];
// 	Random random;
// 	std::unordered_set<Coordinate, Coordinate::Hash> blocks;
// 	// Called solely to seed and advance the random instance
// 	veinCanGenerateDiamonds(chunkCoordinates, random);
// 	Coordinate generationPoint = getGenerationPoint(chunkCoordinates, random);
// 	emulateVein(generationPoint, random, blocks);

// 	Pair<int32_t> chunkCoordinatesEast = {chunkCoordinates.first + 1, chunkCoordinates.second};
// 	std::unordered_set<Coordinate, Coordinate::Hash> blocksEast;
// 	// Called solely to seed and advance the random instance
// 	veinCanGenerateDiamonds(chunkCoordinatesEast, random);
// 	Coordinate generationPointEast = getGenerationPoint(chunkCoordinatesEast, random);
// 	emulateVein(generationPointEast, random, blocksEast);

// 	Pair<int32_t> chunkCoordinatesSouth = {chunkCoordinates.first, chunkCoordinates.second + 1};
// 	std::unordered_set<Coordinate, Coordinate::Hash> blocksSouth;
// 	// Called solely to seed and advance the random instance
// 	veinCanGenerateDiamonds(chunkCoordinatesSouth, random);
// 	Coordinate generationPointSouth = getGenerationPoint(chunkCoordinatesSouth, random);
// 	emulateVein(generationPointSouth, random, blocksSouth);

// 	Pair<int32_t> chunkCoordinatesSoutheast = {chunkCoordinates.first + 1, chunkCoordinates.second + 1};
// 	std::unordered_set<Coordinate, Coordinate::Hash> blocksSoutheast;
// 	// Called solely to seed and advance the random instance
// 	veinCanGenerateDiamonds(chunkCoordinatesSoutheast, random);
// 	Coordinate generationPointSoutheast = getGenerationPoint(chunkCoordinatesSoutheast, random);
// 	emulateVein(generationPointSoutheast, random, blocksSoutheast);

// 	// TODO: Get combined contiguous volumne of veins
// 	uint64_t volume = 0;
// 	if (volume < MINIMUM_VOLUME) return;

// 	removeOtherVeinBlocks(chunkCoordinates, blocks);
// 	removeOtherVeinBlocks(chunkCoordinatesEast, blocksEast);
// 	removeOtherVeinBlocks(chunkCoordinatesSouth, blocksSouth);
// 	removeOtherVeinBlocks(chunkCoordinatesSoutheast, blocksSoutheast);
// 	// TODO: Get combined contiguous volume of veins
// 	volume = 0;
// 	if (volume < MINIMUM_VOLUME) return;

	// printf("%" PRId32 "\t%" PRId32 "\t%" PRId32 "\t%" PRIu64 "\n", generationPoint.x, generationPoint.y, generationPoint.z, volume);
// }


void initialFilter() {
	while (true) {
		// Manual atomicAdd since that isn't callable in host code
		mutex.lock();
		uint64_t currentIteration = globalCurrentIteration;
		// The if condition is technically unnecessary, but it keeps the ETA from becoming negative
		if (globalCurrentIteration < GLOBAL_ITERATIONS_NEEDED) ++globalCurrentIteration;
		mutex.unlock();
		if (GLOBAL_ITERATIONS_NEEDED <= currentIteration) break;
		uint64_t actualIterationToTest = START_IN_MIDDLE_OF_RANGE ? GLOBAL_ITERATIONS_NEEDED/2 + (2*(currentIteration & 1) - 1)*((currentIteration + 1)/2) : currentIteration;

		// Reset storage array 1, and call filter 1
		storageArraySize = 0;
		veinFilter1<<<constexprCeil(static_cast<double>(WORKERS_PER_DEVICE)/static_cast<double>(WORKERS_PER_BLOCK)), WORKERS_PER_BLOCK>>>((actualIterationToTest + ITERATION_PARTS_OFFSET)*WORKERS_PER_DEVICE);
		TRY_CUDA(cudaDeviceSynchronize());
		// If no results were returned, skip filter2
		if (!storageArraySize) continue;

		// If *too many* results were returned, warn and truncate
		if (storageArraySize > ACTUAL_STORAGE_CAPACITY) {
			fprintf(stderr, "WARNING: Iteration %" PRIu64 " returned %" PRIu64 " more results than the storage array can hold. Discarding the extras. (In future, increase MAX_RESULTS_PER_FILTER or decrease WORKERS_PER_DEVICE.)\n", currentIteration, storageArraySize - ACTUAL_STORAGE_CAPACITY);
			storageArraySize = ACTUAL_STORAGE_CAPACITY;
		}
		// Reset storage array 2, and call filter 2
		storageArray2Size = 0;
		veinFilter2<<<constexprCeil(static_cast<double>(WORKERS_PER_DEVICE)/static_cast<double>(WORKERS_PER_BLOCK)), WORKERS_PER_BLOCK>>>();
		TRY_CUDA(cudaDeviceSynchronize());
		// If no results were returned, skip filter3
		// if (!storageArray2Size) continue;

		// // If *too many* results were returned, warn and truncate
		// if (storageArray2Size > ACTUAL_STORAGE_CAPACITY) {
		// 	fprintf(stderr, "WARNING: Iteration %" PRIu64 " returned %" PRIu64 " more results than the storage array can hold. Discarding the extras. (In future, increase MAX_RESULTS_PER_FILTER or decrease WORKERS_PER_DEVICE.)\n", currentIteration, storageArray2Size - ACTUAL_STORAGE_CAPACITY);
		// 	storageArray2Size = ACTUAL_STORAGE_CAPACITY;
		// }
		// // Reset storage array 1, and call filter 3
		// storageArraySize = 0;
		// veinFilter3<<<constexprCeil(static_cast<double>(WORKERS_PER_DEVICE)/static_cast<double>(WORKERS_PER_BLOCK)), WORKERS_PER_BLOCK>>>();
		// TRY_CUDA(cudaDeviceSynchronize());
	}
}

#endif