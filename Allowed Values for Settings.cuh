#ifndef __ALLOWED_VALUES_FOR_SETTINGS_CUH
#define __ALLOWED_VALUES_FOR_SETTINGS_CUH

#include "src/Base Logic.cuh"
#include <array>

enum VeinStates {
	Stone,   _ = Stone,
	Vein,    V = Vein,
	Unknown, u = Unknown
};

enum Material {
	Dirt
};

enum Version {
	v1_8_9
};

struct InputData {
	Version version;
	Material material;
	Coordinate coordinate, layoutDimensions;
	// VeinStates ***layout;
	// std::array<std::array<std::array<VeinStates, layoutDimensions.x>, layoutDimensions.z>, layoutDimensions.y> array;
};

enum ExperimentalMaterial {
	Gravel = -1,
	// Not currently supported due to non-power-of-two range
	Granite = -2,
	// Not currently supported due to non-power-of-two range
	Diorite = -3,
	// Not currently supported due to non-power-of-two range
	Andesite = -4,
	Coal = -5,
	Iron = -6,
	Gold = -7,
	Redstone = -8,
	Diamond = -9,
	// Not currently supported due to triangular distribution
	Lapis_Lazuli = -10
};

enum ExperimentalVersion {
	v1_7_9 = -1
};

#endif