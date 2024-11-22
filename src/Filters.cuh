#ifndef __FILTERS_CUH
#define __FILTERS_CUH

#include "Settings and Input Data Processing.cuh"


// TODO: Replace this with lattice reduction?
__global__ void filter1(const uint64_t iterationStateRangeStart, const size_t chunkIndex) {
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

	int32_t generationPointYOffset = static_cast<int32_t>(state >> BITS_LEFT_AFTER_Y);;
	Random random = Random().setState(state);
	int32_t generationPointXOffset = random.nextInt<-1>(16);
	int32_t generationPointZOffset = random.nextInt<2>(16);
	double angle = static_cast<double>(random.nextFloat());
	int32_t y1 = random.nextInt(3);
	int32_t y2 = random.nextInt(3);

	// Calculate the indices in the vein array that correspond to the generation point.
	struct Coordinate offsetInLayout = {
		(CHUNKS_TO_EXAMINE.coordinates[chunkIndex].first  + 8*USE_POPULATION_OFFSET + generationPointXOffset) - NARROWED_INPUT_COORDINATE.x,
		(VEIN_RANGE.lowerBound + generationPointYOffset) - NARROWED_INPUT_COORDINATE.y,
		(CHUNKS_TO_EXAMINE.coordinates[chunkIndex].second + 8*USE_POPULATION_OFFSET + generationPointZOffset) - NARROWED_INPUT_COORDINATE.z
	};
	// Coordinate veinGenerationPoint = {
	// 	CHUNKS_TO_EXAMINE.coordinates[chunkIndex].first  + 8*USE_POPULATION_OFFSET + generationPointXOffset,
	// 	VEIN_RANGE.lowerBound + generationPointYOffset,
	// 	CHUNKS_TO_EXAMINE.coordinates[chunkIndex].second + 8*USE_POPULATION_OFFSET + generationPointZOffset
	// };
	// Initialize emulated vein
	VeinStates currentVein[NARROWED_INPUT_DIMENSIONS.y][NARROWED_INPUT_DIMENSIONS.z][NARROWED_INPUT_DIMENSIONS.x];
	for (int32_t y = 0; y < NARROWED_INPUT_DIMENSIONS.y; ++y) {
		for (int32_t z = 0; z < NARROWED_INPUT_DIMENSIONS.z; ++z) {
			for (int32_t x = 0; x < NARROWED_INPUT_DIMENSIONS.x; ++x) currentVein[y][z][x] = VeinStates::Stone;
		}
	}

	// Emulates the vein generation algorithm.
	angle *= PI;
	double maxX = sin(angle)*static_cast<double>(VEIN_SIZE)/8.;
	double maxZ = cos(angle)*static_cast<double>(VEIN_SIZE)/8.;
	for (int32_t k = 0; k < VEIN_SIZE + (INPUT_DATA.version == ExperimentalVersion::v1_7_9); ++k) {
		double interpoland = static_cast<double>(k)/static_cast<double>(VEIN_SIZE);
		// Linearly interpolates between -sin(f)*VEIN_SIZE/8. and sin(f)*VEIN_SIZE/8.; y1 and y2; and -cos(f)*VEIN_SIZE/8. and sin(f)*VEIN_SIZE/8..
		double xInterpolation = maxX*(1. - 2.*interpoland);
		double yInterpolation = static_cast<double>(y1) + static_cast<double>(y2 - y1) * interpoland - 2;
		double zInterpolation = maxZ*(1. - 2.*interpoland);
		double maxRadiusSqrt = (sin(PI*interpoland) + 1.)*static_cast<double>(VEIN_SIZE)/32.*random.nextDouble() + 0.5;
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

					// Calculate equivalent coordinate within layout
					int32_t x = xOffset + offsetInLayout.x;
					int32_t y = yOffset + offsetInLayout.y;
					int32_t z = zOffset + offsetInLayout.z;
					// If that coordinate would fall outside the layout's bounds:
					if (x < 0 || NARROWED_INPUT_DIMENSIONS.x <= x || y < 0 || NARROWED_INPUT_DIMENSIONS.y <= y || z < 0 || NARROWED_INPUT_DIMENSIONS.z <= z) {
						// Abort if acting as stone, otherwise ignore (if treating as unknown)
						if (TREAT_COORDINATES_OUTSIDE_INPUT_AS_STONE) return;
					} else {
						// Otherwise if it does fall within the grid, but the input data differs, abort
						if (INPUT_DATA_LAYOUT[y][z][x] == VeinStates::Stone) return;
						// Otherwise add to current vein
						currentVein[y][z][x] = VeinStates::Vein;
					}
				}
			}
		}
	}

	// Make sure current vein and input vein are identical, and abort if not
	for (int32_t y = 0; y < NARROWED_INPUT_DIMENSIONS.y; ++y) {
		for (int32_t z = 0; z < NARROWED_INPUT_DIMENSIONS.z; ++z) {
			for (int32_t x = 0; x < NARROWED_INPUT_DIMENSIONS.x; ++x) {
				if (INPUT_DATA_LAYOUT[y][z][x] != VeinStates::Unknown && INPUT_DATA_LAYOUT[y][z][x] != currentVein[y][z][x]) return;
			}
		}
	}

	// Step two calls back so we're at the call that originally created the vein
	random = Random().setState(state);
	random.skip<-2>();
	// Then print result
	printf("\t%" PRIu64 "\t(%" PRId32 ", %" PRId32 ")\n", random.state, CHUNKS_TO_EXAMINE.coordinates[chunkIndex].first, CHUNKS_TO_EXAMINE.coordinates[chunkIndex].second);
}

#endif