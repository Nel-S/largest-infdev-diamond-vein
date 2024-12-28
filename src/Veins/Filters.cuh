#ifndef __VEINS__FILTERS_CUH
#define __VEINS__FILTERS_CUH

#include "Input Data Processing.cuh"

#ifdef VEIN_DATA_PROVIDED
// TODO: Replace this with lattice reduction?
__global__ void veinFilter1(const uint64_t iterationStateRangeStart, const size_t chunkIndex) {
	/* Initialize java.util.Random instance with the current state.
	If the state is such that an immediate nextInt(256) would return a value greater than the generation point bounding box's maximum y-value, abort.*/
	// TODO: Rework to support triangular distributions
	uint64_t state = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + static_cast<uint64_t>(threadIdx.x) + (static_cast<uint64_t>(VEIN_GENERATION_POINT_BOUNDING_BOX.first.y) << BITS_LEFT_AFTER_Y) + iterationStateRangeStart;
	int32_t generationPointYOffset = static_cast<int32_t>(state >> BITS_LEFT_AFTER_Y);
	if (generationPointYOffset > VEIN_GENERATION_POINT_BOUNDING_BOX.second.y) return;
	Random random = Random().setState(state);

	// Test if a nextInt(16) previous to it would return a value within the range of possible x-offsets. If not, abort.
	int32_t generationPointXOffset = random.nextInt<-1>(16);
	if (generationPointXOffset < CHUNKS_TO_EXAMINE.generationPointNextIntRanges[chunkIndex].first.x || CHUNKS_TO_EXAMINE.generationPointNextIntRanges[chunkIndex].second.x < generationPointXOffset) return;

	// Test if a nextInt(16) two calls ahead would return a value within the range of possible z-offsets. If not, abort.
	int32_t generationPointZOffset = random.nextInt<2>(16);
	if (generationPointZOffset < CHUNKS_TO_EXAMINE.generationPointNextIntRanges[chunkIndex].first.z || CHUNKS_TO_EXAMINE.generationPointNextIntRanges[chunkIndex].second.z < generationPointZOffset) return;

	// Test if a nextFloat() after that would return an angle within the range of possible angles. If not, abort.
	float angle = random.nextFloat();
	if (!ANGLE_BOUNDS.first.contains(angle) && !ANGLE_BOUNDS.second.contains(angle)) return;

	// int32_t y1 = random.nextInt(3);
	// int32_t y2 = random.nextInt(3);
	// if (...) return;

	// Save state to storage array
	size_t arrayIndex = atomicAdd(&storageArraySize, 1);
	if (ACTUAL_STORAGE_CAPACITY <= arrayIndex) return;
	STORAGE_ARRAY[arrayIndex] = state;
}

__global__ void veinFilter2(const size_t chunkIndex) {
	// Retrieve state from 
	size_t arrayIndex = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
	if (storageArraySize <= arrayIndex) return;
	uint64_t state = STORAGE_ARRAY[arrayIndex];

	int32_t generationPointYOffset = static_cast<int32_t>(state >> BITS_LEFT_AFTER_Y);
	Random random = Random().setState(state);
	int32_t generationPointXOffset = random.nextInt<-1>(16);
	int32_t generationPointZOffset = random.nextInt<2>(16);
	double angle = static_cast<double>(random.nextFloat());
	int32_t y1 = random.nextInt(3);
	int32_t y2 = random.nextInt(3);

	// Calculate the indices in the input layout that correspond to the generation point.
	// Coordinate offsetInInputLayout = {
	// 	(16*CHUNKS_TO_EXAMINE.coordinates[chunkIndex].first  + 8*USE_POPULATION_OFFSET + generationPointXOffset) - VEIN_INPUT_DATA.coordinate.x,
	// 	(VEIN_RANGE.lowerBound + generationPointYOffset) - VEIN_INPUT_DATA.coordinate.y,
	// 	(16*CHUNKS_TO_EXAMINE.coordinates[chunkIndex].second + 8*USE_POPULATION_OFFSET + generationPointZOffset) - VEIN_INPUT_DATA.coordinate.z
	// };
	Coordinate veinGenerationPoint = {
		16*CHUNKS_TO_EXAMINE.coordinates[chunkIndex].first  + 8*USE_POPULATION_OFFSET + generationPointXOffset,
		VEIN_RANGE.lowerBound + generationPointYOffset,
		16*CHUNKS_TO_EXAMINE.coordinates[chunkIndex].second + 8*USE_POPULATION_OFFSET + generationPointZOffset
	};
	// Initialize emulated vein
	VeinState currentVein[INPUT_DIMENSIONS.y][INPUT_DIMENSIONS.z][INPUT_DIMENSIONS.x];
	for (int32_t y = 0; y < INPUT_DIMENSIONS.y; ++y) {
		for (int32_t z = 0; z < INPUT_DIMENSIONS.z; ++z) {
			for (int32_t x = 0; x < INPUT_DIMENSIONS.x; ++x) currentVein[y][z][x] = VeinState::Background;
		}
	}

	// Emulates the vein generation algorithm.
	angle *= PI;
	double maxX = sin(angle)*static_cast<double>(VEIN_SIZE)/8.;
	double maxZ = cos(angle)*static_cast<double>(VEIN_SIZE)/8.;
	for (int32_t k = 0; k < VEIN_SIZE + (VEIN_INPUT_DATA.version <= Version::v1_7_10); ++k) {
		double interpoland = static_cast<double>(k)/static_cast<double>(VEIN_SIZE);
		// Linearly interpolates between -sin(f)*VEIN_SIZE/8. and sin(f)*VEIN_SIZE/8.; y1 and y2; and -cos(f)*VEIN_SIZE/8. and sin(f)*VEIN_SIZE/8..
		double xInterpolation = static_cast<double>(veinGenerationPoint.x) + maxX*(1. - 2.*interpoland);
		double yInterpolation = static_cast<double>(veinGenerationPoint.y) + static_cast<double>(y1) + static_cast<double>(y2 - y1) * interpoland + (VEIN_INPUT_DATA.version <= Version::Beta_1_7_3 ? 2 : -2);
		double zInterpolation = static_cast<double>(veinGenerationPoint.z) + maxZ*(1. - 2.*interpoland);
		double maxRadius = (sin(PI*interpoland) + 1.)*static_cast<double>(VEIN_SIZE)/32.*random.nextDouble() + 0.5;
		double maxRadiusSquared = maxRadius*maxRadius;

		int32_t xStart = static_cast<int32_t>(VEIN_INPUT_DATA.version <= Version::Beta_1_5_02 ? xInterpolation - maxRadius : floor(xInterpolation - maxRadius));
		int32_t xEnd   = static_cast<int32_t>(VEIN_INPUT_DATA.version <= Version::Beta_1_5_02 ? xInterpolation + maxRadius : floor(xInterpolation + maxRadius));
		int32_t yStart = static_cast<int32_t>(VEIN_INPUT_DATA.version <= Version::Beta_1_5_02 ? yInterpolation - maxRadius : floor(yInterpolation - maxRadius));
		int32_t yEnd   = static_cast<int32_t>(VEIN_INPUT_DATA.version <= Version::Beta_1_5_02 ? yInterpolation + maxRadius : floor(yInterpolation + maxRadius));
		int32_t zStart = static_cast<int32_t>(VEIN_INPUT_DATA.version <= Version::Beta_1_5_02 ? zInterpolation - maxRadius : floor(zInterpolation - maxRadius));
		int32_t zEnd   = static_cast<int32_t>(VEIN_INPUT_DATA.version <= Version::Beta_1_5_02 ? zInterpolation + maxRadius : floor(zInterpolation + maxRadius));

		for (int32_t x = xStart; x <= xEnd; ++x) {
			double vectorX = static_cast<double>(x) + 0.5 - xInterpolation;
			double vectorXsquared = vectorX*vectorX;
			if (vectorXsquared >= maxRadiusSquared) continue;
			for (int32_t y = yStart; y <= yEnd; ++y) {
				double vectorY = static_cast<double>(y) + 0.5 - yInterpolation;
				double vectorYsquared = vectorY*vectorY;
				if (vectorXsquared + vectorYsquared >= maxRadiusSquared) continue;
				for (int32_t z = zStart; z <= zEnd; ++z) {
					double vectorZ = static_cast<double>(z) + 0.5 - zInterpolation;
					double vectorZsquared = vectorZ*vectorZ;
					if (vectorXsquared + vectorYsquared + vectorZsquared >= maxRadiusSquared) continue;

					// Calculate equivalent coordinate within layout
					int32_t xIndex = x - VEIN_INPUT_DATA.coordinate.x;
					int32_t yIndex = y - VEIN_INPUT_DATA.coordinate.y;
					int32_t zIndex = z - VEIN_INPUT_DATA.coordinate.z;
					// If that coordinate would fall outside the layout's bounds:
					if (xIndex < 0 || INPUT_DIMENSIONS.x <= xIndex || yIndex < 0 || INPUT_DIMENSIONS.y <= yIndex || zIndex < 0 || INPUT_DIMENSIONS.z <= zIndex) {
						// Quit if acting as stone, otherwise ignore (if treating as unknown)
						if (VEIN_INPUT_DATA.defaultStateOutsideLayout == VeinState::Background) return;
					} else {
						// Otherwise if it does fall within the layout's bounds, but the input data differs, quit
						if (inputLayoutCopy.copy[yIndex][zIndex][xIndex] == VeinState::Background) return;
						// Otherwise add to current vein
						currentVein[yIndex][zIndex][xIndex] = VeinState::Vein;
					}
				}
			}
		}
	}

	// Make sure current vein and input vein are identical, and abort if not
	for (int32_t y = 0; y < INPUT_DIMENSIONS.y; ++y) {
		for (int32_t z = 0; z < INPUT_DIMENSIONS.z; ++z) {
			for (int32_t x = 0; x < INPUT_DIMENSIONS.x; ++x) {
				if (inputLayoutCopy.copy[y][z][x] != VeinState::Unknown && inputLayoutCopy.copy[y][z][x] != currentVein[y][z][x]) return;
			}
		}
	}

	// Step two calls back so we're at the call that originally created the vein
	random = Random().setState(state);
	random.skip<-2>();
	// Then print result
	// TODO: Convert to storing results in an array, so they can be printed to a file if desired
	printf("\t%" PRIu64 "\t(%" PRId32 ", %" PRId32 ")\n", random.state, CHUNKS_TO_EXAMINE.coordinates[chunkIndex].first, CHUNKS_TO_EXAMINE.coordinates[chunkIndex].second);
}

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

		// For each potential origin chunk:
		for (size_t currentChunkToTest = 0; currentChunkToTest < static_cast<size_t>(TOTAL_NUMBER_OF_POSSIBLE_CHUNKS); ++currentChunkToTest) {
			// Reset storage array, and call filter 1
			storageArraySize = 0;
			veinFilter1<<<constexprCeil(static_cast<double>(WORKERS_PER_DEVICE)/static_cast<double>(WORKERS_PER_BLOCK)), WORKERS_PER_BLOCK>>>((actualIterationToTest + ITERATION_PARTS_OFFSET)*WORKERS_PER_DEVICE, currentChunkToTest);
			TRY_CUDA(cudaDeviceSynchronize());
			// If no results were returned, skip filter2
			if (!storageArraySize) continue;

			// If *too many* results were returned, warn and truncate
			if (storageArraySize > ACTUAL_STORAGE_CAPACITY) {
				fprintf(stderr, "WARNING: Iteration %" PRIu64 " on chunk %zd returned %" PRIu64 " more results than the storage array can hold. Discarding the extras. (In future, increase MAX_RESULTS_PER_FILTER or decrease WORKERS_PER_DEVICE.)\n", currentIteration, currentChunkToTest, storageArraySize - ACTUAL_STORAGE_CAPACITY);
				storageArraySize = ACTUAL_STORAGE_CAPACITY;
			}
			// Call filter 2
			veinFilter2<<<constexprCeil(static_cast<double>(WORKERS_PER_DEVICE)/static_cast<double>(WORKERS_PER_BLOCK)), WORKERS_PER_BLOCK>>>(currentChunkToTest);
			TRY_CUDA(cudaDeviceSynchronize());
		}
	}
}



struct EstimatedResults {
	uint64_t chunk[TOTAL_NUMBER_OF_POSSIBLE_CHUNKS];
	__device__ constexpr EstimatedResults() : chunk() {
		for (int32_t i = 0; i < TOTAL_NUMBER_OF_POSSIBLE_CHUNKS; ++i) {
			this->chunk[i] = static_cast<uint64_t>(static_cast<double>(twoToThePowerOf(48)) *
				(CHUNKS_TO_EXAMINE.generationPointNextIntRanges[i].second.x - CHUNKS_TO_EXAMINE.generationPointNextIntRanges[i].first.x + 1)/16. *
				(CHUNKS_TO_EXAMINE.generationPointNextIntRanges[i].second.y - CHUNKS_TO_EXAMINE.generationPointNextIntRanges[i].first.y + 1)/static_cast<double>(Y_BOUNDS.getRange()) *
				(CHUNKS_TO_EXAMINE.generationPointNextIntRanges[i].second.z - CHUNKS_TO_EXAMINE.generationPointNextIntRanges[i].first.z + 1)/16. *
				static_cast<double>(ANGLE_BOUNDS.first.getRange()) *
				static_cast<double>(ANGLE_BOUNDS.second.getRange())
			);
		}
	}
};
__device__ constexpr EstimatedResults estimatedResults;
// static_assert(estimatedResults.chunk[0]/static_cast<double>(GLOBAL_ITERATIONS_NEEDED) <= ACTUAL_STORAGE_CAPACITY, "");

#endif

#endif