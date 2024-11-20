#ifndef __FILTERS_CUH
#define __FILTERS_CUH

#include "Settings and Input Data Processing.cuh"


// TODO: Replace this with lattice reduction?
__global__ void filter1(const uint64_t iterationStateRangeStart, const size_t chunkIndex) {
	/* Initialize java.util.Random instance with the current state.
	   If the state is such that an immediate nextInt(256) would return a value greater than the generation point bounding box's maximum y-value, abort.*/
	uint64_t state = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + static_cast<uint64_t>(threadIdx.x) + (static_cast<uint64_t>(VEIN_GENERATION_POINT_BOUNDING_BOX.first.y) << 40) + iterationStateRangeStart;
	int32_t generationPointY = static_cast<int32_t>(state >> 40);
	if (generationPointY > VEIN_GENERATION_POINT_BOUNDING_BOX.second.y) return;
	Random random = Random().setState(state);

	// Test if a nextInt(16) previous to it would return a value within the range of possible x-offsets. If not, abort.
	int32_t generationPointXOffset = random.nextInt<-1>(16);
	if (generationPointXOffset < CHUNKS_TO_EXAMINE.generationPointNextIntRanges[chunkIndex].first.x || CHUNKS_TO_EXAMINE.generationPointNextIntRanges[chunkIndex].second.x < generationPointXOffset) return;

	// Test if a nextInt(16) two calls ahead would return a value within the range of possible z-offsets. If not, abort.
	int32_t generationPointZOffset = random.nextInt<2>(16);
	if (generationPointZOffset < CHUNKS_TO_EXAMINE.generationPointNextIntRanges[chunkIndex].first.z || CHUNKS_TO_EXAMINE.generationPointNextIntRanges[chunkIndex].second.z < generationPointZOffset) return;

	// Test if a nextFloat() after that would return an angle within the range of possible angles. If not, abort.
	float angle = random.nextFloat();
	if (!ANGLE_BOUNDS.contains(angle)) return;

	// int32_t y1 = random.nextInt(3);
	// int32_t y2 = random.nextInt(3);
	// if (...) return;

	// Save state to storage array
	size_t arrayIndex = atomicAdd(&storageArraySize, 1);
	if (ACTUAL_STORAGE_CAPACITY <= arrayIndex) return;
	STORAGE_ARRAY[arrayIndex] = state;
}

__global__ void filter2(const size_t chunkIndex) {
	// Retrieve state from 
	size_t arrayIndex = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
	if (storageArraySize <= arrayIndex) return;
	uint64_t state = STORAGE_ARRAY[arrayIndex];

	int32_t generationPointY = static_cast<int32_t>(state >> 40);
	Random random = Random().setState(state);
	int32_t generationPointXOffset = random.nextInt<-1>(16);
	int32_t generationPointZOffset = random.nextInt<2>(16);
	float angle = random.nextFloat();
	int32_t y1 = random.nextInt(3);
	int32_t y2 = random.nextInt(3);

	// Calculate the indices in the vein array that correspond to the generation point.
	struct Coordinate generationPointInVein = {
		(CHUNKS_TO_EXAMINE.coordinates[chunkIndex].first  + 8*USE_POPULATION_OFFSET + generationPointXOffset) - VEIN_COORDINATE.x,
		generationPointY - VEIN_COORDINATE.y,
		(CHUNKS_TO_EXAMINE.coordinates[chunkIndex].second + 8*USE_POPULATION_OFFSET + generationPointZOffset) - VEIN_COORDINATE.z
	};
	// Initialize emulated vein
	VeinStates currentVein[INPUT_LENGTHS.y][INPUT_LENGTHS.z][INPUT_LENGTHS.x];
	for (int32_t y = 0; y < INPUT_LENGTHS.y; ++y) {
		for (int32_t z = 0; z < INPUT_LENGTHS.z; ++z) {
			for (int32_t x = 0; x < INPUT_LENGTHS.x; ++x) currentVein[y][z][x] = VeinStates::Stone;
		}
	}

	// Emulates the vein generation algorithm.
	// In 1.7.9, this is <= DIRT_SIZE
	for (int32_t k = 0; k < DIRT_SIZE; ++k) {
		double interpoland = static_cast<double>(k)/static_cast<double>(DIRT_SIZE);
		// Linearly interpolates between -sin(f)*DIRT_SIZE/8. and sin(f)*DIRT_SIZE/8.; y1 and y2; and -cos(f)*DIRT_SIZE/8. and sin(f)*DIRT_SIZE/8..
		double xInterpolation = sin(angle*PI) * static_cast<double>(DIRT_SIZE)/8. - (sin(angle*PI) * static_cast<double>(DIRT_SIZE)/4.) * interpoland;
		double yInterpolation = static_cast<double>(y1) + static_cast<double>(y2 - y1) * interpoland;
		double zInterpolation = cos(angle*PI) * static_cast<double>(DIRT_SIZE)/8. - (cos(angle*PI) * static_cast<double>(DIRT_SIZE)/4.) * interpoland;
		double maxRadiusSqrt = (sin(interpoland*PI) + 1.) * static_cast<double>(DIRT_SIZE)/32. * random.nextDouble() + 0.5;
		double maxRadius = maxRadiusSqrt*maxRadiusSqrt;

		for (int32_t xOffset = static_cast<int32_t>(floor(xInterpolation - maxRadiusSqrt)); xOffset <= static_cast<int32_t>(floor(xInterpolation + maxRadiusSqrt)); ++xOffset) {
			double vectorX = static_cast<double>(xOffset) + 0.5 - xInterpolation;
			double vectorXsquared = vectorX*vectorX;
			if (vectorXsquared >= maxRadius) continue;
			for (int32_t yOffset = static_cast<int32_t>(floor(yInterpolation - maxRadiusSqrt)); yOffset <= static_cast<int32_t>(floor(yInterpolation + maxRadiusSqrt)); ++yOffset) {
				double vectorY = static_cast<double>(yOffset) + 0.5 - yInterpolation;
				double vectorYsquared = vectorY*vectorY;
				if (vectorXsquared + vectorYsquared >= maxRadius) continue;
 				for (int32_t zOffset = static_cast<int32_t>(floor(zInterpolation - maxRadiusSqrt)); zOffset <= static_cast<int32_t>(floor(zInterpolation + maxRadiusSqrt)); ++zOffset) {
					double vectorZ = static_cast<double>(zOffset) + 0.5 - zInterpolation;
					double vectorZsquared = vectorZ*vectorZ;
					if (vectorXsquared + vectorYsquared + vectorZsquared >= maxRadius) continue;

					// TODO: Calculate
					int32_t x = generationPointInVein.x + xOffset;
					int32_t y = generationPointInVein.y + yOffset;
					int32_t z = generationPointInVein.z + zOffset;
					if (x < 0 || INPUT_LENGTHS.x <= x || y < 0 || INPUT_LENGTHS.y <= y || z < 0 || INPUT_LENGTHS.z <= z || TRUE_VEIN[y][z][x] == VeinStates::Stone) return;
					currentVein[y][z][x] = VeinStates::Vein;
				}
			}
		}
	}

	for (int32_t y = 0; y < INPUT_LENGTHS.y; ++y) {
		for (int32_t z = 0; z < INPUT_LENGTHS.z; ++z) {
			for (int32_t x = 0; x < INPUT_LENGTHS.x; ++x) {
				if (TRUE_VEIN[y][z][x] != VeinStates::Unknown && TRUE_VEIN[y][z][x] != currentVein[y][z][x]) return;
			}
		}
	}

	// Step two calls back so we're at the call that originally created the vein
	random = Random().setState(state);
	random.skip<-2>();
	printf("\t%" PRIu64 "\t(%" PRId32 ", %" PRId32 ")\n", random.state, CHUNKS_TO_EXAMINE.coordinates[chunkIndex].first, CHUNKS_TO_EXAMINE.coordinates[chunkIndex].second);
}

#endif