#ifndef __VEINS_CUH
#define __VEINS_CUH

#include "PRNG Logic.cuh"
#include "../Allowed Values for Settings.cuh"

constexpr [[nodiscard]] Pair<Coordinate> getWorldBounds(const Version version) {
	switch (version) {
		case Version::v1_7_9:
		case Version::v1_8_9:
			return {{-30000000, 0, -30000000}, {29999999, 255, 29999999}};
		default: throw std::invalid_argument("Unknown version provided.");
	}
}

// The maximum number of blocks away 1.8.9 dirt can be placed from the vein's generation point.
// First is farthest in the negative directions; second is farthest in the positive directions.
constexpr [[nodiscard]] int32_t getVeinSize(const Material material, const Version version) {
	switch (material) {
		case Material::Dirt: switch (version) {
			case Version::v1_7_9: return 32;
			case Version::v1_8_9: return 33;
			default: throw std::invalid_argument("Unknown version provided.");
		}
		default: throw std::invalid_argument("Unknown material provided.");
	}
}

constexpr [[nodiscard]] Pair<Coordinate> getMaxVeinDisplacement(const Material material, const Version version) {
	switch (material) {
		case Material::Dirt: switch (version) {
			case Version::v1_8_9: return {{-6, -3, -6}, {5, 4, 5}};
			default: throw std::invalid_argument("Unknown version provided.");
		}
		default: throw std::invalid_argument("Unknown material provided.");
	}
}

constexpr [[nodiscard]] Coordinate getMaxVeinDimensions(const Material material, const Version version) {
	Pair<Coordinate> maxVeinDisplacement = getMaxVeinDisplacement(material, version);
	return Coordinate(
		maxVeinDisplacement.second.x - maxVeinDisplacement.first.x + 1,
		maxVeinDisplacement.second.y - maxVeinDisplacement.first.y + 1,
		maxVeinDisplacement.second.z - maxVeinDisplacement.first.z + 1
	);
};

#endif