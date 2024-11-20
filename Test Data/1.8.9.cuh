#include "../src/Base Logic.cuh"

__device__ constexpr VeinStates TRUE_VEIN_1_8_9__1[][10][6] = {
	{ // -y
		// -x								+x
		{Stone, Stone, Stone, Stone, Stone, Stone}, // -z
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Vein,  Stone, Stone},
		{Stone, Stone, Stone, Vein,  Stone, Stone},
		{Stone, Stone, Stone, Vein,  Stone, Stone},
		{Stone, Stone, Stone, Vein,  Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone}  // +z
	}, {
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Vein,  Vein,  Stone, Stone},
		{Stone, Vein,  Vein,  Vein,  Stone, Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Stone, Vein,  Vein,  Stone}
	}, {
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Vein,  Vein,  Vein,  Stone, Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Vein },
		{Stone, Vein,  Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Stone, Vein,  Vein,  Stone}
	}, {
		{Stone, Vein,  Vein,  Stone, Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone}
	}, {
		{Stone, Vein,  Vein,  Stone, Stone, Stone},
		{Stone, Vein,  Vein,  Vein,  Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Stone, Vein,  Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone}
	}, {
		// -x								+x
		{Stone, Stone, Stone, Stone, Stone, Stone}, // -z
		{Stone, Stone, Vein,  Stone, Stone, Stone},
		{Stone, Vein,  Vein,  Stone, Stone, Stone},
		{Stone, Stone, Vein,  Vein,  Stone, Stone},
		{Stone, Stone, Vein,  Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone}  // +z
	} // +y
};
__device__ constexpr struct Coordinate VEIN_COORDINATE_1_8_9__1 = {54, 47, -76};


// Internal state: 8675309
__device__ constexpr VeinStates TRUE_VEIN_1_8_9__2[][10][6] = {
	{ // -y
		// -x								+x
		{Stone, Stone, Stone, Stone, Stone, Stone}, // -z
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Vein,  Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Vein,  Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone}, // +z
	}, {
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Vein,  Vein,  Stone, Stone, Stone},
		{Stone, Vein,  Vein,  Vein,  Stone, Stone},
		{Stone, Vein,  Vein,  Vein,  Stone, Stone},
		{Stone, Stone, Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Stone, Vein,  Vein,  Vein },
		{Stone, Stone, Stone, Stone, Vein,  Vein },
	}, {
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Vein,  Vein,  Stone, Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Stone, Vein,  Vein,  Vein },
		{Stone, Stone, Stone, Stone, Vein,  Vein },
	}, {
		{Vein,  Vein,  Stone, Stone, Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Stone, Stone, Vein,  Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
	}, {
		{Vein,  Vein,  Stone, Stone, Stone, Stone},
		{Vein,  Vein,  Vein,  Stone, Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Vein,  Vein,  Vein,  Stone, Stone},
		{Stone, Stone, Stone, Vein,  Vein,  Stone},
		{Stone, Stone, Stone, Vein,  Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
	}, {
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Vein,  Vein,  Stone, Stone, Stone},
		{Stone, Vein,  Vein,  Stone, Stone, Stone},
		{Stone, Stone, Vein,  Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone}
	} // +y
};
__device__ constexpr struct Coordinate VEIN_COORDINATE_1_8_9__2 = {53, 95, 83};



// Internal state: 260269899193147 (CUDA offset = 8675309)
__device__ constexpr VeinStates TRUE_VEIN_1_8_9__3[][6][10] = {
	{ // -y
		{Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Vein,  Vein,  Vein,  Vein,  Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Vein,  Vein,  Vein,  Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Vein,  Vein,  Stone, Vein,  Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone}
	}, {
		{Stone, Stone, Vein,  Vein,  Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Stone, Stone, Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Stone, Stone},
		{Stone, Stone, Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Stone, Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Stone, Stone, Stone, Stone, Vein,  Vein,  Vein,  Stone}
	}, {
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone, Stone, Stone, Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Stone, Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Stone, Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein },
		{Stone, Stone, Stone, Stone, Stone, Vein,  Vein,  Vein,  Vein,  Vein }
	}, {
		{Stone, Vein,  Vein,  Vein,  Vein,  Stone, Stone, Stone, Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Stone, Stone, Stone, Stone},
		{Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Stone, Stone},
		{Stone, Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Stone, Vein,  Vein,  Vein,  Vein,  Vein,  Vein,  Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone}
	}, {
		{Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Vein,  Vein,  Vein,  Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Vein,  Vein,  Vein,  Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Vein,  Vein,  Stone, Vein,  Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone},
		{Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone, Stone}
	}
};
__device__ constexpr struct Coordinate VEIN_COORDINATE_1_8_9__3 = {54, -1, 87};