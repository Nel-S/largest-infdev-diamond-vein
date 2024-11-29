#ifndef __VEINS_CUH
#define __VEINS_CUH

#include "PRNG Logic.cuh"
#include "../Allowed Values for Settings.cuh"

// Returns the coordinates corresponding to the edges of the world
constexpr [[nodiscard]] InclusiveRange<int32_t> getYBounds(const Version version) {
	if (version <= Version::Beta_1_8_through_v1_2) return {0, 127};
	return {0, 255};
	// TODO: Will need updating if 1.18+ is ever implemented (if it can even be implemented)
}


constexpr [[nodiscard]] int32_t getVeinAttemptCount(const Material material, const Version version) {
	switch (material) {
		case Material::Dirt:
			// Infdev 20100617-1-(?) didn't generate dirt
			// if (version <= ExperimentalVersion::Infdev_20100617_1) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_7_through_v1_7_10) return 20;
			if (version <= Version::v1_8_through_v1_12_2) return 10;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 10;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Gravel:
			// Infdev 20100617-1-(?) didn't generate gravel
			// if (version <= ExperimentalVersion::Infdev_20100617_1) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_7_through_v1_7_10) return 10;
			if (version <= Version::v1_8_through_v1_12_2) return 8;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 8;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case ExperimentalMaterial::Granite:
		case ExperimentalMaterial::Diorite:
		case ExperimentalMaterial::Andesite:
			// 1.7.10- doesn't generate granite, diorite, or andesite
			if (version <= Version::v1_7_through_v1_7_10) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_8_through_v1_12_2) return 10;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 10;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Coal:
			if (version <= Version::v1_8_through_v1_12_2) return 20;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 20;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Iron:
			// if (version <= ExperimentalVersion::Infdev_20100325) return 10;
			if (version <= Version::v1_8_through_v1_12_2) return 20;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 20;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Gold:
			// TODO: Checked if nextInt(2) == 0 prior to attempt. Use to filter out half of all states?
			if (version <= ExperimentalVersion::Infdev_20100625_1_through_Alpha_1_0_0) return 1;
			if (version <= Version::v1_8_through_v1_12_2) return 2;
			switch (version) {
				// TODO: 1.13+ Badlands has extra gold (ORE_GOLD_EXTRA)
				case ExperimentalVersion::v1_16_5:
					return 2;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Redstone:
			// Alpha 1.0.0- didn't generate redstone
			if (version <= ExperimentalVersion::Infdev_20100625_1_through_Alpha_1_0_0) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_8_through_v1_12_2) return 8;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 8;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Diamond:
			// TODO: Checked if nextInt(4) == 0 prior to attempt. Use to filter out 3/4ths of all states?
			if (version <= ExperimentalVersion::Infdev_20100625_1_through_Alpha_1_0_0) return 1;
			if (version <= Version::v1_8_through_v1_12_2) return 1;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 1;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case ExperimentalMaterial::Lapis_Lazuli:
			// Beta 1.1_02- doesn't generate lapis
			if (version <= ExperimentalVersion::Alpha_1_0_1_01_through_Beta_1_1_02) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_8_through_v1_12_2) return 1;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 1;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		default: throw std::invalid_argument("Invalid material provided.");
	}
}

constexpr [[nodiscard]] int32_t getVeinSize(const Material material, const Version version) {
	switch (material) {
		case Material::Dirt:
		case Material::Gravel:
			// Infdev 20100617-1-(?) didn't generate dirt or gravel
			// if (version <= ExperimentalVersion::Infdev_20100617_1) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_7_through_v1_7_10) return 32;
			if (version <= ExperimentalVersion::v1_13) return 33;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 33;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case ExperimentalMaterial::Granite:
		case ExperimentalMaterial::Diorite:
		case ExperimentalMaterial::Andesite:
			// 1.7.10- doesn't generate granite, diorite, or andesite
			if (version <= Version::v1_7_through_v1_7_10) throw std::invalid_argument("Invalid version provided.");
			if (version <= ExperimentalVersion::v1_13) return 33;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 33;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Coal:
			if (version <= Version::v1_7_through_v1_7_10) return 16;
			if (version <= ExperimentalVersion::v1_13) return 17;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 17;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Iron:
		case Material::Gold:
			// if (version <= ExperimentalVersion::Infdev_20100325) return 16;
			if (version <= Version::v1_7_through_v1_7_10) return 8;
			if (version <= ExperimentalVersion::v1_13) return 9;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 9;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Redstone:
			// Alpha 1.0.0- didn't generate redstone
			if (version <= ExperimentalVersion::Infdev_20100625_1_through_Alpha_1_0_0) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_7_through_v1_7_10) return 7;
			if (version <= ExperimentalVersion::v1_13) return 8;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 8;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Diamond:
			// if (version <= ExperimentalVersion::Infdev_20100325) return 16;
			if (version <= ExperimentalVersion::Infdev_20100625_1_through_Alpha_1_0_0) return 8;
			if (version <= Version::v1_7_through_v1_7_10) return 7;
			if (version <= ExperimentalVersion::v1_13) return 8;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 8;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case ExperimentalMaterial::Lapis_Lazuli:
			// Beta 1.1- doesn't generate lapis
			if (version <= ExperimentalVersion::Alpha_1_0_1_01_through_Beta_1_1_02) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_7_through_v1_7_10) return 6;
			if (version <= ExperimentalVersion::v1_13) return 7;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return 7;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		default: throw std::invalid_argument("Invalid material provided.");
	}
}

constexpr [[nodiscard]] InclusiveRange<int32_t> getVeinRange(const Material material, const Version version) {
	switch (material) {
		case Material::Dirt:
		case Material::Gravel:
			// Infdev 20100617-1-(?) didn't generate dirt or gravel
			// if (version <= ExperimentalVersion::Infdev_20100617_1) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_2_1_through_v1_6_4) return {0, 128};
			if (version <= Version::v1_8_through_v1_12_2) return {0, 256};
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return {0, 256};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case ExperimentalMaterial::Granite:
		case ExperimentalMaterial::Diorite:
		case ExperimentalMaterial::Andesite:
			// 1.7.10- doesn't generate granite, diorite, or andesite
			if (version <= Version::v1_7_through_v1_7_10) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_8_through_v1_12_2) return {0, 80};
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return {0, 80};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Coal:
			if (version <= Version::v1_8_through_v1_12_2) return {0, 128};
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return {0, 128};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Iron:
			if (version <= Version::v1_8_through_v1_12_2) return {0, 64};
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return {0, 64};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Gold:
			if (version <= Version::v1_8_through_v1_12_2) return {0, 32};
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return {0, 32};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Redstone:
			// Alpha 1.0.0- didn't generate redstone
			if (version <= ExperimentalVersion::Infdev_20100625_1_through_Alpha_1_0_0) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_8_through_v1_12_2) return {0, 16};
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return {0, 16};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Diamond:
			if (version <= Version::v1_8_through_v1_12_2) return {0, 16};
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return {0, 16};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case ExperimentalMaterial::Lapis_Lazuli:
			// Beta 1.1- didn't generate lapis
			if (version <= ExperimentalVersion::Alpha_1_0_1_01_through_Beta_1_1_02) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_8_through_v1_12_2) return {16, 16};
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return {16, 16};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		default: throw std::invalid_argument("Invalid material provided.");
	}
}

constexpr [[nodiscard]] bool veinUsesTriangularDistribution(const Material material, const Version version) {
	switch (material) {
		case Material::Dirt:
		case Material::Gravel:
			// Infdev 20100617-1-(?) didn't generate dirt or gravel
			// if (version <= ExperimentalVersion::Infdev_20100617_1) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_8_through_v1_12_2) return false;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return false;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case ExperimentalMaterial::Granite:
		case ExperimentalMaterial::Diorite:
		case ExperimentalMaterial::Andesite:
			// 1.7.10- didn't generate granite, diorite, or andesite
			if (version <= Version::v1_7_through_v1_7_10) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_8_through_v1_12_2) return false;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return false;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Coal:
		case Material::Iron:
		case Material::Gold:
		case Material::Diamond:
			if (version <= Version::v1_8_through_v1_12_2) return false;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return false;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case Material::Redstone:
			// Alpha 1.0.0- didn't generate redstone
			if (version <= ExperimentalVersion::Infdev_20100625_1_through_Alpha_1_0_0) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_8_through_v1_12_2) return false;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return false;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case ExperimentalMaterial::Lapis_Lazuli:
			// Beta 1.1- didn't generate lapis
			if (version <= ExperimentalVersion::Alpha_1_0_1_01_through_Beta_1_1_02) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_8_through_v1_12_2) return true;
			switch (version) {
				case ExperimentalVersion::v1_16_5:
					return true;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		default: throw std::invalid_argument("Invalid material provided.");
	}
}


// The maximum number of blocks away the vein's blocks can be placed from its generation point.
// First is farthest in the negative directions; second is farthest in the positive directions.
// TODO: Implement Beta 1.4-
constexpr [[nodiscard]] Pair<Coordinate> getMaxVeinBlockDisplacement(const Material material, const Version version) {
	double veinSize = static_cast<double>(getVeinSize(material, version));
	/* In 1.7.9, the maximum interpoland is 1.; in 1.8.9, it's 1. - 1./veinSize.*/
	int32_t horizontalMinDisplacement = static_cast<int32_t>(constexprFloor(-veinSize*(4 + 0.999999999999999 /* 1 - 1/2^53 */ *(constexprSin((1. - static_cast<double>(version <= Version::v1_7_10)/veinSize)*PI) + 1))/32. - 0.25));
	int32_t horizontalMaxDisplacement = static_cast<int32_t>(constexprFloor(veinSize*4.999999999999999 /* 5 - 1/2^53 */ /32. + 0.5));
	double commonVerticalTerm = veinSize * 0.999999999999999 /* 1 - 1/2^53 */ * (constexprSin(constexprFloor((veinSize + static_cast<double>(version <= Version::v1_7_10))/2.)*PI/veinSize) + 1)/32.;
	return {
		{
			/* Occurs when interpoland is maximized, nextDouble is maximized, and angle is as near 0.5 as possible
			   Will have to be slightly modified for 1.7.9*/
			horizontalMinDisplacement,
			// Occurs when interpoland is as close to half as possible, nextDouble is maximized, and r1 = r2 and are minimized
			static_cast<int32_t>(constexprFloor(-commonVerticalTerm - 0.5)) + (version <= Version::Beta_1_6_through_Beta_1_7_3 ? 2 : -2),
			/* Occurs when interpoland is maximized, nextDouble is maximized, and angle is minimized
			   Will have to be slightly modified for 1.7.9*/
			horizontalMinDisplacement
		}, {
			// Occurs when interpoland is minimized, nextDouble is maximized, and angle is as near 0.5 as possible
			horizontalMaxDisplacement,
			// Occurs when interpoland is as close to half as possible, nextDouble is maximized, and r1 = r2 and are maximized
			static_cast<int32_t>(constexprFloor(commonVerticalTerm + 2.5)) + (version <= Version::Beta_1_6_through_Beta_1_7_3 ? 2 : -2),
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

// TODO: Generalize
// constexpr [[nodiscard]] Pair<InclusiveRange<float>> getAngleBounds(const Material material, const Version version) {
// 	int32_t veinSize = getVeinSize(material, version);
// 	Coordinate maxVeinDimensions = getMaxVeinDimensions(material, version);

// 	InclusiveRange<float> lower(0.f, 1.f), upper(0.f, 1.f);
// 	switch (veinSize) {
// 		case 33: switch (version) {
// 			case Version::v1_7_through_v1_7_10:
// 				switch (maxVeinDimensions.x) {
// 					case 11:
// 					case 10:
// 						upper.lowerBound = 0.335f;
// 						lower.upperBound = 0.665f;
// 						break;
// 					case 9:
// 					case 8:
// 						upper.lowerBound = 0.335f;
// 						lower.upperBound = 0.665f;
// 						break;
// 					case 7:
// 					case 6:
// 						upper.lowerBound = 0.335f;
// 						lower.upperBound = 0.665f;
// 						break;
// 				}
// 				break;
// 			case Version::v1_8_through_v1_12_2: switch (maxVeinDimensions.x) {
// 				case 11:
// 					upper.lowerBound = 0.335f;
// 					lower.upperBound = 0.665f;
// 					break;
// 				case 10:
// 					upper.lowerBound = 0.335f;
// 					lower.upperBound = 0.665f;
// 					break;
// 			}
// 		}
// 	}
// 	return {lower, upper};
// }

#endif