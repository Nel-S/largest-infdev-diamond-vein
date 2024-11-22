#include "src/Veins Logic.cuh"
#include <unordered_set>

constexpr Material MATERIAL = Material::Dirt;
// constexpr Material MATERIAL = static_cast<Material>(ExperimentalMaterial::Redstone);
constexpr Version VERSION = Version::v1_8_9;
// constexpr Version VERSION = static_cast<Version>(ExperimentalVersion::v1_7_9);
constexpr Pair<int32_t> ORIGIN_CHUNK = {3, 5};
// constexpr uint64_t INITIAL_INTERNAL_STATE = 64696796158506;
// constexpr uint64_t INITIAL_INTERNAL_STATE = 260269899193147;
constexpr uint64_t INITIAL_INTERNAL_STATE = 137298410352641;

// ~~~

// From Stack Overflow
struct CoordinateHashFunction {
	size_t operator()(const Coordinate& coordinate) const {
		return static_cast<size_t>(coordinate.x) * 17 + (static_cast<size_t>(coordinate.y) ^ static_cast<size_t>(coordinate.z));
	}
};

constexpr Pair<Coordinate> WORLD_BOUNDS = getWorldBounds(VERSION);
constexpr int32_t VEIN_SIZE = getVeinSize(MATERIAL, VERSION);
constexpr InclusiveRange<int32_t> VEIN_RANGE = getVeinRange(MATERIAL, VERSION);
constexpr Pair<Coordinate> MAX_VEIN_DISPLACEMENT = getMaxVeinBlockDisplacement(MATERIAL, VERSION);
constexpr Coordinate MAX_VEIN_DIMENSIONS = getMaxVeinDimensions(MATERIAL, VERSION);
constexpr bool VEIN_USES_TRIANGULAR_DISTRIBUTION = veinUsesTriangularDistribution(MATERIAL, VERSION);
// TODO: Will ultimately be replaced with VERSION <= v1.12.2
constexpr bool USE_POPULATION_OFFSET = true;

std::unordered_set<Coordinate, CoordinateHashFunction> emulateVeinRaw(const Pair<int32_t> &chunk, Random &random, VeinStates vein[MAX_VEIN_DIMENSIONS.y][MAX_VEIN_DIMENSIONS.z][MAX_VEIN_DIMENSIONS.x], Coordinate &veinCoordinate) {
	if (!vein) throw std::invalid_argument("Null pointer provided for vein.\n");
	for (int32_t y = 0; y < MAX_VEIN_DIMENSIONS.y; ++y) {
		for (int32_t z = 0; z < MAX_VEIN_DIMENSIONS.z; ++z) {
			for (int32_t x = 0; x < MAX_VEIN_DIMENSIONS.x; ++x) vein[y][z][x] = VeinStates::Stone;
		}
	}
	std::unordered_set<Coordinate, CoordinateHashFunction> coordsSet;

	Coordinate veinGenerationPoint = {
		chunk.first*16 + random.nextInt(16),
		VEIN_RANGE.lowerBound + (VEIN_USES_TRIANGULAR_DISTRIBUTION ? random.nextInt(VEIN_RANGE.upperBound) + random.nextInt(VEIN_RANGE.upperBound) - VEIN_RANGE.upperBound : random.nextInt(VEIN_RANGE.upperBound - VEIN_RANGE.lowerBound)),
		chunk.second*16 + random.nextInt(16)
	};
	veinCoordinate.x = veinGenerationPoint.x + MAX_VEIN_DISPLACEMENT.first.x;
	veinCoordinate.y = veinGenerationPoint.y + MAX_VEIN_DISPLACEMENT.first.y;
	veinCoordinate.z = veinGenerationPoint.z + MAX_VEIN_DISPLACEMENT.first.z;

	float angle = random.nextFloat() * static_cast<float>(PI);
	double maxX = static_cast<double>(static_cast<float>(veinGenerationPoint.x + 8) + sinf(angle)*static_cast<float>(VEIN_SIZE)/8.f);
	double minX = static_cast<double>(static_cast<float>(veinGenerationPoint.x + 8) - sinf(angle)*static_cast<float>(VEIN_SIZE)/8.f);
	double maxZ = static_cast<double>(static_cast<float>(veinGenerationPoint.z + 8) + cosf(angle)*static_cast<float>(VEIN_SIZE)/8.f);
	double minZ = static_cast<double>(static_cast<float>(veinGenerationPoint.z + 8) - cosf(angle)*static_cast<float>(VEIN_SIZE)/8.f);

	double y1 = static_cast<double>(veinGenerationPoint.y + random.nextInt(3) - 2);
	double y2 = static_cast<double>(veinGenerationPoint.y + random.nextInt(3) - 2);

	for (int32_t k = 0; k < VEIN_SIZE + (VERSION == static_cast<Version>(ExperimentalVersion::v1_7_9)); ++k) {
		float interpoland = static_cast<float>(k)/static_cast<float>(VEIN_SIZE);
		double xInterpolation = maxX + (minX - maxX) * static_cast<double>(interpoland);
		double yInterpolation = y1 + (y2 - y1) * static_cast<double>(interpoland);
		double zInterpolation = maxZ + (minZ - maxZ) * static_cast<double>(interpoland);
		double commonDiameterTerm = random.nextDouble() * static_cast<double>(VEIN_SIZE)/16.;
		double horizontalMaxDiameter = static_cast<double>(sinf(static_cast<float>(PI)*interpoland) + 1.f)*commonDiameterTerm + 1.;
		double verticalMaxDiameter = static_cast<double>(sinf(static_cast<float>(PI)*interpoland) + 1.f)*commonDiameterTerm + 1.;
		int32_t xStart = static_cast<int32_t>(floor(xInterpolation - horizontalMaxDiameter/2.));
		int32_t yStart = static_cast<int32_t>(floor(yInterpolation - verticalMaxDiameter/2.));
		int32_t zStart = static_cast<int32_t>(floor(zInterpolation - horizontalMaxDiameter/2.));
		int32_t xEnd = static_cast<int32_t>(floor(xInterpolation + horizontalMaxDiameter/2.));
		int32_t yEnd = static_cast<int32_t>(floor(yInterpolation + verticalMaxDiameter/2.));
		int32_t zEnd = static_cast<int32_t>(floor(zInterpolation + horizontalMaxDiameter/2.));

		for (int32_t x = xStart; x <= xEnd; ++x) {
			double vectorX = (static_cast<double>(x) + 0.5 - xInterpolation)/(horizontalMaxDiameter/2.);
			if (vectorX*vectorX >= 1.) continue;
			for (int32_t y = yStart; y <= yEnd; ++y) {
				double vectorY = (static_cast<double>(y) + 0.5 - yInterpolation)/(verticalMaxDiameter/2.);
				if (vectorX*vectorX + vectorY*vectorY >= 1.) continue;
 				for (int32_t z = zStart; z <= zEnd; ++z) {
					double vectorZ = (static_cast<double>(z) + 0.5 - zInterpolation)/(horizontalMaxDiameter/2.);
					if (vectorX*vectorX + vectorY*vectorY + vectorZ*vectorZ >= 1.) continue;

					if (y < WORLD_BOUNDS.first.y || WORLD_BOUNDS.second.y < y) continue;
					// printf("(%d, %d, %d)\n", x, y, z);
					coordsSet.insert(Coordinate(x, y, z));
					// TODO: Not correct yet (but coordsSet is)
					// size_t xIndex = static_cast<size_t>(x - veinCoordinate.x);
					// size_t yIndex = static_cast<size_t>(y - veinCoordinate.y);
					// size_t zIndex = static_cast<size_t>(z - veinCoordinate.z);
					// vein[yIndex][zIndex][xIndex] = VeinStates::Vein;
				}
			}
		}
	}
	return coordsSet;
}

std::unordered_set<Coordinate, CoordinateHashFunction> emulateVeinCleaned(const Pair<int32_t> &chunk, Random &random, VeinStates vein[MAX_VEIN_DIMENSIONS.y][MAX_VEIN_DIMENSIONS.z][MAX_VEIN_DIMENSIONS.x], Coordinate &veinCoordinate) {
	if (!vein) throw std::invalid_argument("Null pointer provided for vein.\n");
	for (int32_t y = 0; y < MAX_VEIN_DIMENSIONS.y; ++y) {
		for (int32_t z = 0; z < MAX_VEIN_DIMENSIONS.z; ++z) {
			for (int32_t x = 0; x < MAX_VEIN_DIMENSIONS.x; ++x) vein[y][z][x] = VeinStates::Stone;
		}
	}
	std::unordered_set<Coordinate, CoordinateHashFunction> coordsSet;

	int32_t call1 = random.nextInt(16);
	int32_t call2 = VEIN_USES_TRIANGULAR_DISTRIBUTION ? random.nextInt(VEIN_RANGE.upperBound) : random.nextInt(VEIN_RANGE.upperBound - VEIN_RANGE.lowerBound);
	int32_t call3 = VEIN_USES_TRIANGULAR_DISTRIBUTION ? random.nextInt(VEIN_RANGE.upperBound) : INT32_MIN;
	int32_t call4 = random.nextInt(16);
	printf("%d\t%d\t%d\t%d\n", call1, call2, call3, call4);
	// Coordinate veinGenerationPoint = {
	// 	chunk.first*16 + random.nextInt(16) + 8*USE_POPULATION_OFFSET,
	// 	VEIN_RANGE.lowerBound + (VEIN_USES_TRIANGULAR_DISTRIBUTION ? random.nextInt(VEIN_RANGE.upperBound) + random.nextInt(VEIN_RANGE.upperBound) - VEIN_RANGE.upperBound : random.nextInt(VEIN_RANGE.upperBound - VEIN_RANGE.lowerBound)) - 2,
	// 	chunk.second*16 + random.nextInt(16) + 8*USE_POPULATION_OFFSET
	// };
	Coordinate veinGenerationPoint = {
		chunk.first*16 + call1 + 8*USE_POPULATION_OFFSET,
		VEIN_RANGE.lowerBound + (VEIN_USES_TRIANGULAR_DISTRIBUTION ? call2 + call3 - VEIN_RANGE.upperBound : call2),
		chunk.second*16 + call4 + 8*USE_POPULATION_OFFSET
	};
	printf("%d\t%d\t%d\n", veinGenerationPoint.x, veinGenerationPoint.y, veinGenerationPoint.z);
	veinCoordinate.x = veinGenerationPoint.x + MAX_VEIN_DISPLACEMENT.first.x;
	veinCoordinate.y = veinGenerationPoint.y + MAX_VEIN_DISPLACEMENT.first.y - 2;
	veinCoordinate.z = veinGenerationPoint.z + MAX_VEIN_DISPLACEMENT.first.z;

	double angle = static_cast<double>(random.nextFloat());
	int32_t y1 = random.nextInt(3);
	int32_t y2 = random.nextInt(3);

	angle *= PI;
	double maxX = sin(angle)*static_cast<double>(VEIN_SIZE)/8.;
	double maxZ = cos(angle)*static_cast<double>(VEIN_SIZE)/8.;
	for (int32_t k = 0; k < VEIN_SIZE + (VERSION == static_cast<Version>(ExperimentalVersion::v1_7_9)); ++k) {
		double interpoland = static_cast<double>(k)/static_cast<double>(VEIN_SIZE);
		// Linearly interpolates between -sin(f)*VEIN_SIZE/8. and sin(f)*VEIN_SIZE/8.; y1 and y2; and -cos(f)*VEIN_SIZE/8. and sin(f)*VEIN_SIZE/8..
		double xInterpolation = maxX*(1. - 2.*interpoland);
		double yInterpolation = static_cast<double>(y1) + static_cast<double>(y2 - y1) * interpoland - 2;
		double zInterpolation = maxZ*(1. - 2.*interpoland);
		double maxRadiusSqrt = (sin(PI*interpoland) + 1.)*static_cast<double>(VEIN_SIZE)/32.*random.nextDouble() + 0.5;
		double maxRadius = maxRadiusSqrt*maxRadiusSqrt;

		for (int32_t xOffset = static_cast<int32_t>(floor(xInterpolation - maxRadiusSqrt)); xOffset <= static_cast<int32_t>(floor(xInterpolation + maxRadiusSqrt)); ++xOffset) {
			double vectorX = static_cast<double>(xOffset) + 0.5 - xInterpolation;
			double vectorXSquared = vectorX*vectorX;
			if (vectorXSquared >= maxRadius) continue;
			for (int32_t yOffset = static_cast<int32_t>(floor(yInterpolation - maxRadiusSqrt)); yOffset <= static_cast<int32_t>(floor(yInterpolation + maxRadiusSqrt)); ++yOffset) {
				double vectorY = static_cast<double>(yOffset) + 0.5 - yInterpolation;
				double vectorYSquared = vectorY*vectorY;
				if (vectorXSquared + vectorYSquared >= maxRadius) continue;
 				for (int32_t zOffset = static_cast<int32_t>(floor(zInterpolation - maxRadiusSqrt)); zOffset <= static_cast<int32_t>(floor(zInterpolation + maxRadiusSqrt)); ++zOffset) {
					double vectorZ = static_cast<double>(zOffset) + 0.5 - zInterpolation;
					double vectorZSquared = vectorZ*vectorZ;
					if (vectorXSquared + vectorYSquared + vectorZSquared >= maxRadius) continue;

					int32_t y = veinGenerationPoint.y + yOffset;
					if (y < WORLD_BOUNDS.first.y || WORLD_BOUNDS.second.y < y) continue;
					// printf("(%d, %d, %d)\n", veinGenerationPoint.x + xOffset, y, veinGenerationPoint.z + zOffset);
					coordsSet.insert(Coordinate(veinGenerationPoint.x + xOffset, y, veinGenerationPoint.z + zOffset));
					size_t xIndex = static_cast<size_t>(xOffset - MAX_VEIN_DISPLACEMENT.first.x);
					size_t yIndex = static_cast<size_t>(yOffset - MAX_VEIN_DISPLACEMENT.first.y);
					size_t zIndex = static_cast<size_t>(zOffset - MAX_VEIN_DISPLACEMENT.first.z);
					// printf("(%d, %d, %d) -> (%zd, %zd, %zd)\n", veinGenerationPoint.x + xOffset, y, veinGenerationPoint.z + zOffset, xIndex, yIndex, zIndex);
					// printf("(%d, %d, %d) -> (%zd, %zd, %zd)\n", xOffset, yOffset, zOffset, xIndex, yIndex, zIndex);
					vein[yIndex][zIndex][xIndex] = VeinStates::Vein;
				}
			}
		}
	}
	// printf("[%d, %d, %d]\n\n", veinCoordinate.x, veinCoordinate.y, veinCoordinate.z);
	// for (const Coordinate &coord : coordsSet) printf("(%d, %d, %d) -> (%d, %d, %d)\n", coord.x, coord.y, coord.z, coord.x - veinCoordinate.x, coord.y - veinCoordinate.y, coord.z - veinCoordinate.z);
	return coordsSet;
}


void printVein(const VeinStates vein[MAX_VEIN_DIMENSIONS.y][MAX_VEIN_DIMENSIONS.z][MAX_VEIN_DIMENSIONS.x], const Coordinate &veinCoordinate) {
	printf("(%" PRId32 ", %" PRId32 ", %" PRId32 ")\n", veinCoordinate.x, veinCoordinate.y, veinCoordinate.z);
	for (int32_t y = 0; y < MAX_VEIN_DIMENSIONS.y; ++y) {
		for (int32_t z = 0; z < MAX_VEIN_DIMENSIONS.z; ++z) {
			for (int32_t x = 0; x < MAX_VEIN_DIMENSIONS.x; ++x) {
				switch (vein[y][z][x]) {
					case VeinStates::Stone:
						printf("_");
						break;
					case VeinStates::Vein:
						printf("X");
						break;
					default: printf("?");
				}
			}
			printf("\n");
		}
		printf("\n\n");
	}
}

int main() {
	VeinStates vein[MAX_VEIN_DIMENSIONS.y][MAX_VEIN_DIMENSIONS.z][MAX_VEIN_DIMENSIONS.x];
	Random random = Random().setState(INITIAL_INTERNAL_STATE);
	Coordinate veinCoordinate;
	std::unordered_set<Coordinate, CoordinateHashFunction> exactSet = emulateVeinRaw(ORIGIN_CHUNK, random, vein, veinCoordinate);
	// printVein(vein, veinCoordinate);
	// printf("\n\n~~~~~~~\n\n");
	random = Random().setState(INITIAL_INTERNAL_STATE);
	std::unordered_set<Coordinate, CoordinateHashFunction> emulationSet = emulateVeinCleaned(ORIGIN_CHUNK, random, vein, veinCoordinate);
	printf("%d", exactSet == emulationSet);
	printVein(vein, veinCoordinate);
	return 0;
}