#ifndef __FILTERS__END_PILLARS_CUH
#define __FILTERS__END_PILLARS_CUH

#include "Input Data Processing.cuh"

#ifdef END_PILLAR_DATA_PROVIDED

__global__ bool endPillarFilter(std::set<int32_t> &pillarSeeds) {
	pillarSeeds.clear();
	bool foundAny = false;
	for (int32_t pillarSeed = 0; pillarSeed < (1 << 16); ++pillarSeed) {
		int32_t arrangement[NUMBER_OF_PILLARS] = {
			Y_76__Diameter_5, Y_79__Diameter_5__Caged, Y_82__Diameter_5__Caged, Y_85__Diameter_7,  Y_88__Diameter_7,
			Y_91__Diameter_7, Y_94__Diameter_9,        Y_97__Diameter_9,        Y_100__Diameter_9, Y_103__Diameter_11
		};
		Random random(pillarSeed);
		for (uint32_t i = NUMBER_OF_PILLARS - 1; i; --i) constexprSwap(arrangement[i], arrangement[random.nextInt(i + 1)]);

		bool confirmedPillarSeed = false;
		for (int32_t orientation = (orientationIsKnown ? END_PILLAR_INPUT_DATA.layoutOrientation : -1); !confirmedPillarSeed && orientation <= (orientationIsKnown ? END_PILLAR_INPUT_DATA.layoutOrientation : 1); orientation += 2) {
			for (uint32_t offset = (eastIndexIsKnown ? END_PILLAR_INPUT_DATA.eastmostPillarIndex : 0); !confirmedPillarSeed && offset <= (eastIndexIsKnown ? END_PILLAR_INPUT_DATA.eastmostPillarIndex : NUMBER_OF_PILLARS - 1); ++offset) {
				for (uint32_t i = 0; i < NUMBER_OF_PILLARS; ++i) {
					if (!(END_PILLAR_INPUT_DATA.knownPillars[i] & arrangement[(i + orientation*offset) % NUMBER_OF_PILLARS])) goto nextOffset;
				}
				confirmedPillarSeed = true;
				nextOffset: continue;
			}
		}
		if (!confirmedPillarSeed) continue;
		pillarSeeds.insert(pillarSeed);
		foundAny = true;
	}
	return foundAny;
}
#endif

void initialFilter(uint64_t *outputArray, size_t *outputArraySize, const size_t outputArrayCapacity) {
	
}

uint64_t estimatedIterationsForInitialFilter() {

}

void furtherFilter(uint64_t *inputAndOutputArray, size_t *inputAndOutputArraySize) {
	if (!inputAndOutputArray) throw std::invalid_argument("Nonexistent input/output array provided.");
	if (!inputAndOutputArraySize) throw std::invalid_argument("Nonexistent input/output array size provided.");
	if (!*inputAndOutputArraySize) throw std::invalid_argument("Empty input array provided.");

}

uint64_t estimatedIterationsForFurtherFilter() {

}

#endif