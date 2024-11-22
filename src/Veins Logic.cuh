#ifndef __VEINS_CUH
#define __VEINS_CUH

#include "PRNG Logic.cuh"
#include "../Allowed Values for Settings.cuh"

constexpr [[nodiscard]] Pair<Coordinate> getWorldBounds(const Version version) {
	switch (version) {
		case ExperimentalVersion::v1_7_9:
		case Version::v1_8_9:
			return {{-30000000, 0, -30000000}, {29999999, 255, 29999999}};
		default: throw std::invalid_argument("Invalid version provided.");
	}
}

constexpr [[nodiscard]] int32_t getVeinAttemptCount(const Material material, const Version version) {
	switch (material) {
		case Material::Dirt: switch (version) {
			case ExperimentalVersion::v1_7_9: return 20;
			case Version::v1_8_9: return 10;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Gravel: switch (version) {
			case ExperimentalVersion::v1_7_9: return 10;
			case Version::v1_8_9: return 8;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Granite: switch (version) {
			// 1.7.9 doesn't generate granite
			case Version::v1_8_9: return 10;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Diorite: switch (version) {
			// 1.7.9 doesn't generate diorite
			case Version::v1_8_9: return 10;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Andesite: switch (version) {
			// 1.7.9 doesn't generate andesite
			case Version::v1_8_9: return 10;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Coal: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return 20;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Iron: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return 20;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Gold: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return 2;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Redstone: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return 8;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Diamond: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return 1;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Lapis_Lazuli: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return 1;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		default: throw std::invalid_argument("Invalid material provided.");
	}
}

constexpr [[nodiscard]] int32_t getVeinSize(const Material material, const Version version) {
	switch (material) {
		case Material::Dirt: switch (version) {
			case ExperimentalVersion::v1_7_9: return 32;
			case Version::v1_8_9: return 33;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Gravel: switch (version) {
			case ExperimentalVersion::v1_7_9: return 32;
			case Version::v1_8_9: return 33;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Granite: switch (version) {
			// 1.7.9 doesn't generate granite
			case Version::v1_8_9: return 33;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Diorite: switch (version) {
			// 1.7.9 doesn't generate diorite
			case Version::v1_8_9: return 33;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Andesite: switch (version) {
			// 1.7.9 doesn't generate andesite
			case Version::v1_8_9: return 33;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Coal: switch (version) {
			case ExperimentalVersion::v1_7_9: return 16;
			case Version::v1_8_9: return 17;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Iron: switch (version) {
			case ExperimentalVersion::v1_7_9: return 8;
			case Version::v1_8_9: return 9;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Gold: switch (version) {
			case ExperimentalVersion::v1_7_9: return 8;
			case Version::v1_8_9: return 9;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Redstone: switch (version) {
			case ExperimentalVersion::v1_7_9: return 7;
			case Version::v1_8_9: return 8;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Diamond: switch (version) {
			case ExperimentalVersion::v1_7_9: return 7;
			case Version::v1_8_9: return 8;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Lapis_Lazuli: switch (version) {
			case ExperimentalVersion::v1_7_9: return 6;
			case Version::v1_8_9: return 7;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		default: throw std::invalid_argument("Invalid material provided.");
	}
}

// TODO: Differentiate triangular distributions from uniform distributions
constexpr [[nodiscard]] InclusiveRange<int32_t> getVeinRange(const Material material, const Version version) {
	switch (material) {
		case Material::Dirt: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return {0, 256};
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Gravel: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return {0, 256};
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Granite: switch (version) {
			// 1.7.9 doesn't generate granite
			case Version::v1_8_9: return {0, 80};
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Diorite: switch (version) {
			// 1.7.9 doesn't generate diorite
			case Version::v1_8_9: return {0, 80};
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Andesite: switch (version) {
			// 1.7.9 doesn't generate andesite
			case Version::v1_8_9: return {0, 80};
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Coal: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return {0, 128};
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Iron: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return {0, 64};
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Gold: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return {0, 32};
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Redstone: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return {0, 16};
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Diamond: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return {0, 16};
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Lapis_Lazuli: switch (version) {
			case Version::v1_8_9: return {16, 16};
			default: throw std::invalid_argument("Invalid version provided.");
		}
		default: throw std::invalid_argument("Invalid material provided.");
	}
}

// TODO: Actually figure out
constexpr [[nodiscard]] bool veinUsesTriangularDistribution(const Material material, const Version version) {
	switch (material) {
		case Material::Dirt: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return false;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Gravel: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return false;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Granite: switch (version) {
			// 1.7.9 doesn't generate granite
			case Version::v1_8_9: return false;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Diorite: switch (version) {
			// 1.7.9 doesn't generate diorite
			case Version::v1_8_9: return false;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Andesite: switch (version) {
			// 1.7.9 doesn't generate andesite
			case Version::v1_8_9: return false;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Coal: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return false;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Iron: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return false;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Gold: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return false;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Redstone: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return false;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Diamond: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return false;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		case ExperimentalMaterial::Lapis_Lazuli: switch (version) {
			case ExperimentalVersion::v1_7_9:
			case Version::v1_8_9:
				return true;
			default: throw std::invalid_argument("Invalid version provided.");
		}
		default: throw std::invalid_argument("Invalid material provided.");
	}
}


// The maximum number of blocks away the vein's blocks can be placed from its generation point.
// First is farthest in the negative directions; second is farthest in the positive directions.
constexpr [[nodiscard]] Pair<Coordinate> getMaxVeinBlockDisplacement(const Material material, const Version version) {
	double veinSize = static_cast<double>(getVeinSize(material, version));
	/* In 1.7.9, the maximum interpoland is 1.; in 1.8.9, it's 1. - 1./veinSize.*/
	int32_t horizontalMinDisplacement = static_cast<int32_t>(constexprFloor(-veinSize*(4 + 0.999999999999999 /* 1 - 1/2^53 */ *(constexprSin((1. - static_cast<double>(version <= ExperimentalVersion::v1_7_9)/veinSize)*PI) + 1))/32. - 0.25));
	int32_t horizontalMaxDisplacement = static_cast<int32_t>(constexprFloor(veinSize*4.999999999999999 /* 5 - 1/2^53 */ /32. + 0.5));
	double commonVerticalTerm = veinSize * 0.999999999999999 /* 1 - 1/2^53 */ * (constexprSin(constexprFloor((veinSize + static_cast<double>(version <= ExperimentalVersion::v1_7_9))/2.)*PI/veinSize) + 1)/32.;
	return {
		{
			/* Occurs when interpoland is maximized, nextDouble is maximized, and angle is as near 0.5 as possible
			   Will have to be slightly modified for 1.7.9*/
			horizontalMinDisplacement,
			// Occurs when interpoland is as close to half as possible, nextDouble is maximized, and r1 = r2 and are minimized
			static_cast<int32_t>(constexprFloor(-commonVerticalTerm - 2.5)),
			/* Occurs when interpoland is maximized, nextDouble is maximized, and angle is minimized
			   Will have to be slightly modified for 1.7.9*/
			horizontalMinDisplacement
		}, {
			// Occurs when interpoland is minimized, nextDouble is maximized, and angle is as near 0.5 as possible
			horizontalMaxDisplacement,
			// Occurs when interpoland is as close to half as possible, nextDouble is maximized, and r1 = r2 and are maximized
			static_cast<int32_t>(constexprFloor(commonVerticalTerm + 0.5)),
			// Occurs when interpoland is minimized, nextDouble is maximized, and angle is minimized
			horizontalMaxDisplacement
		}
	};
}

constexpr [[nodiscard]] Coordinate getMaxVeinDimensions(const Material material, const Version version) {
	Pair<Coordinate> maxVeinDisplacement = getMaxVeinBlockDisplacement(material, version);
	return Coordinate(
		maxVeinDisplacement.second.x - maxVeinDisplacement.first.x + 1,
		maxVeinDisplacement.second.y - maxVeinDisplacement.first.y + 1,
		maxVeinDisplacement.second.z - maxVeinDisplacement.first.z + 1
	);
};

#endif