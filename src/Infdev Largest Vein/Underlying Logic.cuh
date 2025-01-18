#ifndef __VEINS__UNDERLYING_LOGIC_CUH
#define __VEINS__UNDERLYING_LOGIC_CUH

#include "../General/Pseudorandom Number Generators.cuh"
#include "Allowed Values for Input Data.cuh"
#include <unordered_set>

constexpr [[nodiscard]] bool isSupported(const VeinMaterial material, const Version version) {
	return material == VeinMaterial::Diamond && version >= Version::Infdev_20100325 && version <= Version::Infdev_20100617_1;
}

// Returns the range of the world's y-axis
// TODO: Add Nether?
constexpr [[nodiscard]] InclusiveRange<int32_t> getWorldYRange(const Version version) {
	if (version <= Version::v1_2) return {0, 127};
	return {0, 255};
	// TODO: Will need updating if 1.18+ is ever implemented (if it can even be implemented)
}

__host__ __device__ constexpr [[nodiscard]] int32_t getVeinSize(const VeinMaterial material, const Version version) {
	switch (material) {
		case VeinMaterial::Dirt:
		case VeinMaterial::Gravel:
			// Infdev 20100617-1-(?) didn't generate dirt or gravel
			if (version <= Version::Infdev_20100617_1) RAISE_EXCEPTION_OR_RETURN_DEFAULT_VALUE(0, "Invalid version provided.");
			if (version <= Version::v1_7_10) return 32;
			if (version <= Version::v1_13) return 33;
			switch (version) {
				case Version::v1_16_5:
					return 33;
				default: RAISE_EXCEPTION_OR_RETURN_DEFAULT_VALUE(0, "Invalid version provided.");
			}
		case VeinMaterial::Coal:
			if (version <= Version::v1_7_10) return 16;
			if (version <= Version::v1_13) return 17;
			switch (version) {
				case Version::v1_16_5:
					return 17;
				default: RAISE_EXCEPTION_OR_RETURN_DEFAULT_VALUE(0, "Invalid version provided.");
			}
		case VeinMaterial::Iron:
		case VeinMaterial::Gold:
			if (version <= Version::Infdev_20100617_1) return 16;
			if (version <= Version::v1_7_10) return 8;
			if (version <= Version::v1_13) return 9;
			switch (version) {
				case Version::v1_16_5:
					return 9;
				default: RAISE_EXCEPTION_OR_RETURN_DEFAULT_VALUE(0, "Invalid version provided.");
			}
		case VeinMaterial::Redstone:
			// Alpha 1.0.0- didn't generate redstone
			if (version <= Version::Alpha_1_0_0) RAISE_EXCEPTION_OR_RETURN_DEFAULT_VALUE(0, "Invalid version provided.");
			if (version <= Version::v1_7_10) return 7;
			if (version <= Version::v1_13) return 8;
			switch (version) {
				case Version::v1_16_5:
					return 8;
				default: RAISE_EXCEPTION_OR_RETURN_DEFAULT_VALUE(0, "Invalid version provided.");
			}
		case VeinMaterial::Diamond:
			if (version <= Version::Infdev_20100617_1) return 16;
			if (version <= Version::Alpha_1_0_0) return 8;
			if (version <= Version::v1_7_10) return 7;
			if (version <= Version::v1_13) return 8;
			switch (version) {
				case Version::v1_16_5:
					return 8;
				default: RAISE_EXCEPTION_OR_RETURN_DEFAULT_VALUE(0, "Invalid version provided.");
			}
		default: RAISE_EXCEPTION_OR_RETURN_DEFAULT_VALUE(0, "Invalid material provided.");
	}
}

constexpr [[nodiscard]] InclusiveRange<int32_t> getVeinYRange(const VeinMaterial material, const Version version) {
	switch (material) {
		case VeinMaterial::Dirt:
		case VeinMaterial::Gravel:
			// Infdev 20100617-1-(?) didn't generate dirt or gravel
			if (version <= Version::Infdev_20100617_1) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_6_4) return {0, 128};
			if (version <= Version::v1_13) return {0, 256};
			switch (version) {
				case Version::v1_16_5:
					return {0, 256};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case VeinMaterial::Coal:
			if (version <= Version::v1_13) return {0, 128};
			switch (version) {
				case Version::v1_16_5:
					return {0, 128};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case VeinMaterial::Iron:
			if (version <= Version::v1_13) return {0, 64};
			switch (version) {
				case Version::v1_16_5:
					return {0, 64};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case VeinMaterial::Gold:
			if (version <= Version::v1_9_4) return {0, 32};
			if (version <= Version::v1_13) return {0, 32}; // PLUS {32, 48} in Badlands-related biomes
			switch (version) {
				case Version::v1_16_5:
					return {0, 32}; // PLUS {32, 48} in Badlands-related biomes
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case VeinMaterial::Redstone:
			// Alpha 1.0.0- didn't generate redstone
			if (version <= Version::Alpha_1_0_0) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_13) return {0, 16};
			switch (version) {
				case Version::v1_16_5:
					return {0, 16};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case VeinMaterial::Diamond:
			if (version <= Version::v1_13) return {0, 16};
			switch (version) {
				case Version::v1_16_5:
					return {0, 16};
				default: throw std::invalid_argument("Invalid version provided.");
			}
		default: throw std::invalid_argument("Invalid material provided.");
	}
}

constexpr [[nodiscard]] bool veinUsesTriangularDistribution(const VeinMaterial material, const Version version) {
	switch (material) {
		case VeinMaterial::Dirt:
		case VeinMaterial::Gravel:
			// Infdev 20100617-1-(?) didn't generate dirt or gravel
			// if (version <= Version::Infdev_20100617_1) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_13) return false;
			switch (version) {
				case Version::v1_16_5:
					return false;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case VeinMaterial::Coal:
		case VeinMaterial::Iron:
		case VeinMaterial::Gold:
		case VeinMaterial::Diamond:
			if (version <= Version::v1_13) return false;
			switch (version) {
				case Version::v1_16_5:
					return false;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		case VeinMaterial::Redstone:
			// Alpha 1.0.0- didn't generate redstone
			if (version <= Version::Alpha_1_0_0) throw std::invalid_argument("Invalid version provided.");
			if (version <= Version::v1_13) return false;
			switch (version) {
				case Version::v1_16_5:
					return false;
				default: throw std::invalid_argument("Invalid version provided.");
			}
		default: throw std::invalid_argument("Invalid material provided.");
	}
}

/* Represents 1 - 1/2^53.
   However, it is not exactly equal because multiple calculations would otherwise lose precision and return 1.,
    which would defeat the purpose.*/
constexpr double MAX_DOUBLE_IN_RANGE = 0.999999999999999;

// The maximum number of blocks away the vein's blocks can be placed from its generation point.
// First is farthest in the negative directions; second is farthest in the positive directions.
__host__ __device__ constexpr [[nodiscard]] Pair<Coordinate> getMaxVeinBlockDisplacement(const VeinMaterial material, const Version version, const Coordinate &generationPoint) {
	double veinSize = static_cast<double>(getVeinSize(material, version));
	/* In 1.7.9, the maximum interpoland is 1.; in 1.8.9, it's 1. - 1./veinSize.*/
	double commonHorizontalMinTerm = static_cast<double>(Version::v1_8 <= version)/4. - veinSize*(4 + MAX_DOUBLE_IN_RANGE*(constexprSin((1. - static_cast<double>(Version::v1_8 <= version)/veinSize)*PI) + 1))/32. - 0.5;
	double commonHorizontalMaxTerm = veinSize*(4 + MAX_DOUBLE_IN_RANGE)/32. + 0.5;
	double commonVerticalTerm = veinSize * MAX_DOUBLE_IN_RANGE*(constexprSin(constexprFloor((veinSize + static_cast<double>(Version::v1_8 <= version))/2.)*PI/veinSize) + 1)/32. + 0.5;

	return {
		{
			/* Occurs when interpoland is maximized, nextDouble is maximized, and angle is as near 0.5 as possible*/
			// version <= Version::Beta_1_5_02 ? static_cast<int32_t>(static_cast<double>(generationPoint.x) + commonHorizontalMinTerm) - generationPoint.x : static_cast<int32_t>(constexprFloor(commonHorizontalMinTerm)),
			static_cast<int32_t>(constexprFloor(commonHorizontalMinTerm)) + (version <= Version::Beta_1_5_02 && commonHorizontalMaxTerm != static_cast<int32_t>(commonHorizontalMaxTerm) && generationPoint.x < -constexprFloor(commonHorizontalMinTerm)),
			// Occurs when interpoland is as close to half as possible, nextDouble is maximized, and r1 = r2 and are minimized
			static_cast<int32_t>(constexprFloor(-commonVerticalTerm)) + (version <= Version::Beta_1_7_3 ? 2 : -2),
			/* Occurs when interpoland is maximized, nextDouble is maximized, and angle is minimized*/
			// version <= Version::Beta_1_5_02 ? static_cast<int32_t>(static_cast<double>(generationPoint.z) + commonHorizontalMinTerm) - generationPoint.z : static_cast<int32_t>(constexprFloor(commonHorizontalMinTerm)),
			static_cast<int32_t>(constexprFloor(commonHorizontalMinTerm)) + (version <= Version::Beta_1_5_02 && commonHorizontalMaxTerm != static_cast<int32_t>(commonHorizontalMaxTerm) && generationPoint.z < -constexprFloor(commonHorizontalMinTerm)),
		}, {
			// Occurs when interpoland is minimized, nextDouble is maximized, and angle is as near 0.5 as possible
			// version <= Version::Beta_1_5_02 ? static_cast<int32_t>(static_cast<double>(generationPoint.x) + commonHorizontalMaxTerm) - generationPoint.x : static_cast<int32_t>(constexprFloor(commonHorizontalMaxTerm)),
			static_cast<int32_t>(constexprFloor(commonHorizontalMaxTerm)) + (version <= Version::Beta_1_5_02 && commonHorizontalMaxTerm != static_cast<int32_t>(commonHorizontalMaxTerm) && generationPoint.x < -constexprFloor(commonHorizontalMaxTerm)),
			// Occurs when interpoland is as close to half as possible, nextDouble is maximized, and r1 = r2 and are maximized
			static_cast<int32_t>(constexprFloor(commonVerticalTerm + 2)) + (version <= Version::Beta_1_7_3 ? 2 : -2),
			// Occurs when interpoland is minimized, nextDouble is maximized, and angle is minimized
			// version <= Version::Beta_1_5_02 ? static_cast<int32_t>(static_cast<double>(generationPoint.z) + commonHorizontalMaxTerm) - generationPoint.z : static_cast<int32_t>(constexprFloor(commonHorizontalMaxTerm))
			static_cast<int32_t>(constexprFloor(commonHorizontalMaxTerm)) + (version <= Version::Beta_1_5_02 && commonHorizontalMaxTerm != static_cast<int32_t>(commonHorizontalMaxTerm) && generationPoint.z < -constexprFloor(commonHorizontalMaxTerm)),
		}
	};
}

// The maximum number of blocks away the vein's blocks can be placed from its generation point, *excluding the Beta 1.5.02- bug*.
// First is farthest in the negative directions; second is farthest in the positive directions.
constexpr [[nodiscard]] Pair<Coordinate> getMaxVeinBlockDisplacement_coordinateIndependent(const VeinMaterial material, const Version version) {
	// if (version <= Version::Beta_1_5_02) throw std::invalid_argument("Invalid version provided.");
	double veinSize = static_cast<double>(getVeinSize(material, version));
	/* In 1.7.9, the maximum interpoland is 1.; in 1.8.9, it's 1. - 1./veinSize.*/
	double commonHorizontalMinTerm = static_cast<double>(Version::v1_8 <= version)/4. - veinSize*(4 + MAX_DOUBLE_IN_RANGE*(constexprSin((1. - static_cast<double>(Version::v1_8 <= version)/veinSize)*PI) + 1))/32. - 0.5;
	double commonHorizontalMaxTerm = veinSize*(4 + MAX_DOUBLE_IN_RANGE)/32. + 0.5;
	double commonVerticalTerm = veinSize * MAX_DOUBLE_IN_RANGE * (constexprSin(constexprFloor((veinSize + static_cast<double>(Version::v1_8 <= version))/2.)*PI/veinSize) + 1)/32. + 0.5;

	return {
		{
			/* Occurs when interpoland is maximized, nextDouble is maximized, and angle is as near 0.5 as possible*/
			static_cast<int32_t>(constexprFloor(commonHorizontalMinTerm)),
			// Occurs when interpoland is as close to half as possible, nextDouble is maximized, and r1 = r2 and are minimized
			static_cast<int32_t>(constexprFloor(-commonVerticalTerm)) + (version <= Version::Beta_1_7_3 ? 2 : -2),
			/* Occurs when interpoland is maximized, nextDouble is maximized, and angle is minimized*/
			static_cast<int32_t>(constexprFloor(commonHorizontalMinTerm))
		}, {
			// Occurs when interpoland is minimized, nextDouble is maximized, and angle is as near 0.5 as possible
			static_cast<int32_t>(constexprFloor(commonHorizontalMaxTerm)),
			// Occurs when interpoland is as close to half as possible, nextDouble is maximized, and r1 = r2 and are maximized
			static_cast<int32_t>(constexprFloor(commonVerticalTerm + 2)) + (version <= Version::Beta_1_7_3 ? 2 : -2),
			// Occurs when interpoland is minimized, nextDouble is maximized, and angle is minimized
			static_cast<int32_t>(constexprFloor(commonHorizontalMaxTerm))
		}
	};
}

// The dimensions of the largest possible vein.
constexpr [[nodiscard]] Coordinate getMaxVeinDimensions(const VeinMaterial material, const Version version, const Coordinate &generationPoint) {
	Pair<Coordinate> maxVeinDisplacement = getMaxVeinBlockDisplacement(material, version, generationPoint);
	return Coordinate(
		maxVeinDisplacement.second.x - maxVeinDisplacement.first.x + 1,
		maxVeinDisplacement.second.y - maxVeinDisplacement.first.y + 1,
		maxVeinDisplacement.second.z - maxVeinDisplacement.first.z + 1
	);
};

// The dimensions of the largest possible vein, *excluding the Beta 1.5.02- bug*.
constexpr [[nodiscard]] Coordinate getMaxVeinDimensions_coordinateIndependent(const VeinMaterial material, const Version version) {
	Pair<Coordinate> maxVeinDisplacement = getMaxVeinBlockDisplacement_coordinateIndependent(material, version);
	return Coordinate(
		maxVeinDisplacement.second.x - maxVeinDisplacement.first.x + 1,
		maxVeinDisplacement.second.y - maxVeinDisplacement.first.y + 1,
		maxVeinDisplacement.second.z - maxVeinDisplacement.first.z + 1
	);
};

__device__ bool veinCanGenerateDiamonds(const Pair<int32_t> &chunk, Random &random) {
	random.setSeed(318279123*static_cast<uint64_t>(chunk.first) + 919871212*static_cast<uint64_t>(chunk.second));
	// Skips dirt and coal vein calls
	for (int32_t i = 0; i < 20 + 10; ++i) {
		random.skip<3 + 1>();
		random.nextInt(3);
		random.nextInt(3);
		random.skip<17*2>();
	}
	// Skips gold vein calls
	if (!random.nextBoolean()) {
		random.skip<3 + 1>();
		random.nextInt(3);
		random.nextInt(3);
		random.skip<17*2>();
	}
	return !random.nextInt(8);
}

__device__ Coordinate getGenerationPoint(const Pair<int32_t> &chunk, Random &random) {
	return {
		(chunk.first  << 4) + random.nextInt(16),
		random.nextInt(16),
		(chunk.second << 4) + random.nextInt(16),
	};
}

// __device__ void emulateVein(const Coordinate &generationPoint, Random &random, std::unordered_set<Coordinate, Coordinate::Hash> &blocks, bool removeBlocks=false) {
// 	double angle = static_cast<double>(random.nextFloat()) * PI;
// 	int32_t y1 = random.nextInt(3);
// 	int32_t y2 = random.nextInt(3);

// 	double maxX = sin(angle)*2.;
// 	double maxZ = cos(angle)*2.;
// 	for (int32_t k = 0; k < 17; ++k) {
// 		double interpoland = static_cast<double>(k)/16.;
// 		double xInterpolation = static_cast<double>(generationPoint.x) + maxX*(1. - 2.*interpoland);
// 		double yInterpolation = static_cast<double>(generationPoint.y) + static_cast<double>(y1) + static_cast<double>(y2 - y1) * interpoland + 2;
// 		double zInterpolation = static_cast<double>(generationPoint.z) + maxZ*(1. - 2.*interpoland);
// 		double maxRadius = (sin(PI*interpoland) + 1.)*random.nextDouble()/2. + 0.5;
// 		double maxRadiusSquared = maxRadius*maxRadius;

// 		// For those who are curious, spot the Beta 1.5.02- bug!
// 		int32_t xStart = static_cast<int32_t>(xInterpolation - maxRadius);
// 		int32_t yStart = static_cast<int32_t>(yInterpolation - maxRadius);
// 		int32_t zStart = static_cast<int32_t>(zInterpolation - maxRadius);
// 		int32_t xEnd   = static_cast<int32_t>(xInterpolation + maxRadius);
// 		int32_t yEnd   = static_cast<int32_t>(yInterpolation + maxRadius);
// 		int32_t zEnd   = static_cast<int32_t>(zInterpolation + maxRadius);

// 		for (int32_t x = xStart; x <= xEnd; ++x) {
// 			double vectorX = static_cast<double>(x) + 0.5 - xInterpolation;
// 			double vectorXSquared = vectorX*vectorX;
// 			if (vectorXSquared >= maxRadiusSquared) continue;
// 			for (int32_t y = yStart; y <= yEnd; ++y) {
// 				double vectorY = static_cast<double>(y) + 0.5 - yInterpolation;
// 				double vectorYSquared = vectorY*vectorY;
// 				if (vectorXSquared + vectorYSquared >= maxRadiusSquared) continue;
// 				for (int32_t z = zStart; z <= zEnd; ++z) {
// 					double vectorZ = static_cast<double>(z) + 0.5 - zInterpolation;
// 					double vectorZSquared = vectorZ*vectorZ;
// 					if (vectorXSquared + vectorYSquared + vectorZSquared >= maxRadiusSquared) continue;

// 					if (y < 0) continue;
// 					if (removeBlocks) blocks.erase(Coordinate(x,y,z));
// 					else blocks.insert(Coordinate(x, y, z));
// 				}
// 			}
// 		}
// 	}
// }

// __device__ void removeOtherVeinBlocks(const Pair<int32_t> &chunk, std::unordered_set<Coordinate, Coordinate::Hash> &diamondBlocks) {
// 	Random random(318279123*static_cast<uint64_t>(chunk.first) + 919871212*static_cast<uint64_t>(chunk.second));
// 	// Dirt and coal veins
// 	for (int32_t i = 0; i < 20 + 10; ++i) {
// 		Coordinate generationPoint = getGenerationPoint(chunk, random);
// 		emulateVein(generationPoint, random, diamondBlocks, true);
// 	}
// 	// Gold vein
// 	if (!random.nextBoolean()) {
// 		Coordinate generationPoint = getGenerationPoint(chunk, random);
// 		emulateVein(generationPoint, random, diamondBlocks, true);
// 	}
// }

__device__ constexpr Pair<Coordinate> MAX_DIAMOND_BOUNDING_BOX_DISPLACEMENT = getMaxVeinBlockDisplacement_coordinateIndependent(VeinMaterial::Diamond, Version::Infdev_20100617_1);

#endif