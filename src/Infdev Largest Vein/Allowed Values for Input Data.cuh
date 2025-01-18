#ifndef __VEINS__ALLOWED_VALUES_FOR_INPUT_DATA_CUH
#define __VEINS__ALLOWED_VALUES_FOR_INPUT_DATA_CUH

#include "../General/Allowed Values for General Settings.cuh"
#include "../General/Base Logic.cuh"

/* ===== VEINS ===== */

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
};

#endif