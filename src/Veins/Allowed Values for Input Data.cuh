#ifndef __VEINS__ALLOWED_VALUES_FOR_INPUT_DATA_CUH
#define __VEINS__ALLOWED_VALUES_FOR_INPUT_DATA_CUH

#include "../General/Allowed Values for General Settings.cuh"
#include "../General/Base Logic.cuh"

/* ===== VEINS ===== */

// The valid values that can be included in your input layout.
enum VeinState {
	Background, _ = VeinState::Background, Stone = VeinState::Background, Netherrack = VeinState::Background,
	Vein,       V = VeinState::Vein,
	Unknown,    u = VeinState::Unknown
};

// The supported vein materials.
enum VeinMaterial {
	Dirt,
	// TODO: Gravel has additional/different attributes in 1.16.5 basalt deltas
	Gravel,
	Coal,
	Iron,
	// TODO: Gold has additional/different attributes in 1.13+ (1.10+?) badlands, 1.16.5 Nethers, and 1.16.5 basalt deltas
	Gold,
	Redstone,
	Diamond,
	Infested_Stone, Stone_Monster_Egg = VeinMaterial::Infested_Stone
};

struct VeinsInputData : InputData {
	VeinMaterial material;
	// Biome biome;
	Coordinate coordinate;
	VeinState defaultStateOutsideLayout;

	// constexpr VeinsInputData() : version(), material(), biome(), coordinate() {}
	// constexpr VeinsInputData(const Version version, const VeinMaterial material, const Coordinate coordinate) : version(version), material(material), biome(), coordinate() {}
};


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/* WARNING: These are experimental features that don't work yet.*/

enum ExperimentalVeinMaterial {
	// Not currently supported due to non-power-of-two range
	Granite = -1,
	// Not currently supported due to non-power-of-two range
	Diorite = -2,
	// Not currently supported due to non-power-of-two range
	Andesite = -3,
	// Not currently supported due to triangular distribution
	Lapis_Lazuli = -4,
	// Not currently supported due to non-power-of-two range
	// TODO: Quartz has additional/different attributes in 1.16.5 basalt deltas
	Quartz = -5,
	// Not currently supported due to non-power-of-two range
	Magma = -6,
	Soul_Sand = -7,
	Blackstone = -10,
	Ancient_Debris = -11, // Has two possible vein configurations
};

// The various test data.
#include "Test Data/Beta 1.4.cuh"
#include "Test Data/Beta 1.6.cuh"
#include "Test Data/1.7.9.cuh"
#include "Test Data/1.8.9.cuh"

#endif