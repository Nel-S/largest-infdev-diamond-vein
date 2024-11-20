#include "src/Veins Logic.cuh"

constexpr Material MATERIAL = Material::Dirt;
constexpr Version VERSION = Version::v1_8_9;
constexpr Pair<int32_t> ORIGIN_CHUNK = {3, 5};
// constexpr uint64_t INITIAL_INTERNAL_STATE = 64696796158506;
constexpr uint64_t INITIAL_INTERNAL_STATE = 260269899193147;

// ~~~

constexpr int32_t DIRT_SIZE = getVeinSize(MATERIAL, VERSION);
constexpr Pair<Coordinate> MAX_DIRT_DISPLACEMENT = getMaxVeinDisplacement(MATERIAL, VERSION);
constexpr Coordinate MAX_VEIN_SIZE = getMaxVeinDimensions(MATERIAL, VERSION);
// TODO: Will ultimately be replaced with VERSION <= v1.12.2
constexpr bool USE_POPULATION_OFFSET = true;

void emulateVein(const Pair<int32_t> &chunk, Random &random, VeinStates vein[MAX_VEIN_SIZE.y][MAX_VEIN_SIZE.z][MAX_VEIN_SIZE.x], Coordinate &veinCoordinate) {
	if (!vein) throw std::invalid_argument("Null pointer provided for vein.\n");
	for (int32_t y = 0; y < MAX_VEIN_SIZE.y; ++y) {
		for (int32_t z = 0; z < MAX_VEIN_SIZE.z; ++z) {
			for (int32_t x = 0; x < MAX_VEIN_SIZE.x; ++x) vein[y][z][x] = VeinStates::Stone;
		}
	}

	veinCoordinate.x = chunk.first*16 + 8*USE_POPULATION_OFFSET + random.nextInt(16) + MAX_DIRT_DISPLACEMENT.first.x;
	printf("[%" PRIu64 "]\n", random.state);
	veinCoordinate.y = random.nextInt(256) + MAX_DIRT_DISPLACEMENT.first.y;
	veinCoordinate.z = chunk.second*16 + 8*USE_POPULATION_OFFSET + random.nextInt(16) + MAX_DIRT_DISPLACEMENT.first.z;

	float angle = random.nextFloat();
	int32_t y1 = random.nextInt(3);
	int32_t y2 = random.nextInt(3);

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

					size_t x = static_cast<size_t>(xOffset - MAX_DIRT_DISPLACEMENT.first.x);
					size_t y = static_cast<size_t>(yOffset - MAX_DIRT_DISPLACEMENT.first.y);
					size_t z = static_cast<size_t>(zOffset - MAX_DIRT_DISPLACEMENT.first.z);
					vein[y][z][x] = VeinStates::Vein;
				}
			}
		}
	}
}

int main() {
	VeinStates vein[MAX_VEIN_SIZE.y][MAX_VEIN_SIZE.z][MAX_VEIN_SIZE.x];
	Random random = Random().setState(INITIAL_INTERNAL_STATE);
	Coordinate veinCoordinate;
	emulateVein(ORIGIN_CHUNK, random, vein, veinCoordinate);
	printf("(%" PRId32 ", %" PRId32 ", %" PRId32 ")\n", veinCoordinate.x, veinCoordinate.y, veinCoordinate.z);
	for (int32_t y = 0; y < MAX_VEIN_SIZE.y; ++y) {
		for (int32_t z = 0; z < MAX_VEIN_SIZE.z; ++z) {
			for (int32_t x = 0; x < MAX_VEIN_SIZE.x; ++x) {
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
	return 0;
}