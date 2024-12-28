#include "../Allowed Values for Input Data.cuh"

// Internal state: ?? (in chunk (?, ?))
// Structure seed: 15179077681579 (with ?? advancements)
__device__ constexpr VeinsInputData VEINS_TEST_DATA_BETA_1_4__1 = {
	CrackerType::Veins,
	// The version that the vein was generated within.
	Version::Beta_1_4,
	// The material of the vein.
	VeinMaterial::Dirt,
	// The coordinate corresponding to the first top-most corner in the input layout below (the -x/-y/-z corner).
	{-336, 34, 86},
	// The default state outside of the input layout.
	VeinState::Stone,
};
// Note: Don't get your x-directions, y-directions, and z-directions mixed up.
constexpr VeinState VEINS_TEST_DATA_BETA_1_4__1_LAYOUT[][7][8] = {
	{ // -y
		// -x                +x
		{_, _, _, _, _, _, _, _}, // -z
		{_, _, _, _, _, _, V, _},
		{_, _, _, _, _, V, V, _},
		{_, _, _, _, V, _, _, _},
		{_, _, _, _, _, _, _, _},
		{_, _, _, _, _, _, _, _},
		{_, _, _, _, _, _, _, _}  // +z
	}, {
		// -x                +x
		{_, _, _, _, _, _, _, _}, // -z
		{_, _, _, _, _, V, V, V},
		{_, _, _, V, V, u, u, V},
		{_, _, V, V, u, V, V, V},
		{_, V, V, V, V, V, _, _},
		{V, V, V, V, V, _, _, _},
		{_, V, V, _, _, _, _, _}  // +z
	}, {
		// -x                +x
		{_, _, _, _, _, _, V, _}, // -z
		{_, _, _, _, _, V, u, V},
		{_, _, u, V, V, u, u, V},
		{_, V, u, u, u, u, V, V},
		{V, u, u, u, u, V, _, _},
		{V, u, u, u, V, _, _, _},
		{V, V, V, V, _, _, _, _}  // +z
	}, {
		// -x                +x
		{_, _, _, _, _, _, _, _}, // -z
		{_, _, _, _, _, V, V, _},
		{_, _, u, V, V, V, V, V},
		{_, V, V, u, V, V, V, _},
		{V, u, u, u, V, V, _, _},
		{V, u, u, V, V, _, _, _},
		{V, V, V, V, _, _, _, _}  // +z
	}, {
		// -x                +x
		{_, _, _, _, _, _, _, _}, // -z
		{_, _, _, _, _, _, _, _},
		{_, _, _, _, _, _, _, _},
		{_, _, _, u, _, _, _, _},
		{_, V, V, V, _, _, _, _},
		{_, V, V, _, u, _, _, _},
		{_, _, _, _, _, _, _, _}  // +z
	} // +y
};
