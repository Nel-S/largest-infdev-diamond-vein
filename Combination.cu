#include "src/General/Population Chunk Reversal.cuh"
#include <algorithm>
#include <iterator>
#include <set>

struct InputData2 {
	uint64_t internalState;
	Pair<int32_t> chunkCoordinate;
	Version version;
};

// The data of the first vein.
constexpr InputData2 VEIN_1_POSSIBLE_DATA[] = {
	{137298410352641, {3, -5}, Version::v1_8_9}
};
// The data of the second vein.
constexpr InputData2 VEIN_2_POSSIBLE_DATA[] = {
	{177995862726538, {4, -5}, Version::v1_8_9},
	{196714877191822, {4, -5}, Version::v1_8_9}
};

int main() {
	results.clear();
	for (size_t veinState = 0; veinState < sizeof(VEIN_1_POSSIBLE_DATA)/sizeof(*VEIN_1_POSSIBLE_DATA); ++veinState) {
		InclusiveRange<uint64_t> populationCallsRange = getPopulationCallsRange(VEIN_1_POSSIBLE_DATA[veinState].version);
		Random random = Random().setState(VEIN_1_POSSIBLE_DATA[veinState].internalState).skip(-static_cast<int64_t>(populationCallsRange.lowerBound));
		for (uint64_t i = populationCallsRange.lowerBound; i <= populationCallsRange.upperBound; ++i, random.skip<-1>()) {
			reversePopulationSeed(random.state ^ LCG::MULTIPLIER, VEIN_1_POSSIBLE_DATA[veinState].chunkCoordinate.first, VEIN_1_POSSIBLE_DATA[veinState].chunkCoordinate.second, VEIN_1_POSSIBLE_DATA[veinState].version);
		}
	}
	std::set<uint64_t> vein1PossibleStructureSeeds(results);
	printf("Vein 1: %zd possibilities\n", vein1PossibleStructureSeeds.size());

	results.clear();
	for (size_t veinState = 1; veinState < 2; ++veinState) {
		InclusiveRange<uint64_t> populationCallsRange = getPopulationCallsRange(VEIN_2_POSSIBLE_DATA[veinState].version);
		Random random = Random().setState(VEIN_2_POSSIBLE_DATA[veinState].internalState).skip(-static_cast<int64_t>(populationCallsRange.lowerBound));
		for (uint64_t i = populationCallsRange.lowerBound; i <= populationCallsRange.upperBound; ++i, random.skip<-1>()) {
			reversePopulationSeed(random.state ^ LCG::MULTIPLIER, VEIN_2_POSSIBLE_DATA[veinState].chunkCoordinate.first, VEIN_2_POSSIBLE_DATA[veinState].chunkCoordinate.second, VEIN_2_POSSIBLE_DATA[veinState].version);
		}
	}
	printf("Vein 2: %zd possibilities\n", vein1PossibleStructureSeeds.size());
	
	std::set<uint64_t> commonStructureSeeds;
	std::set_intersection(vein1PossibleStructureSeeds.begin(), vein1PossibleStructureSeeds.end(), results.begin(), results.end(), std::inserter(commonStructureSeeds, commonStructureSeeds.begin()));
	printf("Common: %zd possibilities\n", commonStructureSeeds.size());
	for (const uint64_t &structureSeed : commonStructureSeeds) printf("%" PRIu64 "\n", structureSeed);
	
	return 0;
}


// constexpr uint64_t STRUCTURE_SEED = 255166352657805;

// int main() {
// 	for (uint64_t i = 0; i < 65536; ++i) {
// 		uint64_t worldseed = (i << 48) | STRUCTURE_SEED;
// 		if (Random().isFromNextLong(worldseed)) printf("%" PRId64 "\n", worldseed);
// 	}
// 	return 0;
// }