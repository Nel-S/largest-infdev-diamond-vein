#ifndef __END_PILLARS__UNDERLYING_LOGIC_CUH
#define __END_PILLARS__UNDERLYING_LOGIC_CUH

#include "../General/Pseudorandom Number Generators.cuh"
#include "Allowed Values for Input Data.cuh"

constexpr bool isSupported(const Version version) {
	return Version::v1_9 <= version;
}

constexpr bool isExperimentallySupported(const Version version) {
	MAYBE_UNUSED(version);
	return false;
}

#endif