#ifndef __END_PILLARS__INPUT_DATA_PROCESSING_CUH
#define __END_PILLARS__INPUT_DATA_PROCESSING_CUH

#include "..\..\Settings and Input Data (MODIFY THIS).cuh"
#include "..\General\General Settings Processing.cuh"
#include "Underlying Logic.cuh"

#if defined(END_PILLAR_INPUT_DATA)
	#define END_PILLAR_DATA_PROVIDED
#endif

#ifdef END_PILLAR_DATA_PROVIDED
constexpr bool orientationIsKnown = END_PILLAR_INPUT_DATA.layoutOrientation != EndPillarLayoutOrientation::Unknown;
constexpr bool eastIndexIsKnown  = 0 <= END_PILLAR_INPUT_DATA.eastmostPillarIndex && END_PILLAR_INPUT_DATA.eastmostPillarIndex < NUMBER_OF_PILLARS;
#endif

#endif