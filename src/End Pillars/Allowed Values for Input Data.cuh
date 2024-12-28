#ifndef __END_PILLARS__ALLOWED_VALUES_FOR_INPUT_DATA_CUH
#define __END_PILLARS__ALLOWED_VALUES_FOR_INPUT_DATA_CUH

#include "../General/Allowed Values for General Settings.cuh"
#include "../General/Base Logic.cuh"

/* ===== END PILLARS ===== */
enum EndPillarTypeFlags {
	Y_76__Diameter_5        = 0b0000000001,
	Y_79__Diameter_5__Caged = 0b0000000010,
	Y_82__Diameter_5__Caged = 0b0000000100,
	Y_85__Diameter_7        = 0b0000001000,
	Y_88__Diameter_7        = 0b0000010000,
	Y_91__Diameter_7        = 0b0000100000,
	Y_94__Diameter_9        = 0b0001000000,
	Y_97__Diameter_9        = 0b0010000000,
	Y_100__Diameter_9       = 0b0100000000,
	Y_103__Diameter_11      = 0b1000000000,
	Unknown                 = 0b1111111111,

	Caged                = EndPillarTypeFlags::Y_79__Diameter_5__Caged | EndPillarTypeFlags::Y_82__Diameter_5__Caged,
	Not_Caged            = ~EndPillarTypeFlags::Caged & EndPillarTypeFlags::Unknown,
	Diameter_5           = EndPillarTypeFlags::Y_76__Diameter_5 | EndPillarTypeFlags::Y_79__Diameter_5__Caged
	                       | EndPillarTypeFlags::Y_82__Diameter_5__Caged,
	Diameter_7           = EndPillarTypeFlags::Y_85__Diameter_7 | EndPillarTypeFlags::Y_88__Diameter_7
	                       | EndPillarTypeFlags::Y_91__Diameter_7,
	Diameter_9           = EndPillarTypeFlags::Y_94__Diameter_9 | EndPillarTypeFlags::Y_97__Diameter_9
	                       | EndPillarTypeFlags::Y_100__Diameter_9,
	Diameter_11          = EndPillarTypeFlags::Y_103__Diameter_11,
	Shorter_Than_Caged   = EndPillarTypeFlags::Y_76__Diameter_5,
	Taller_Than_Caged    = EndPillarTypeFlags::Y_85__Diameter_7 | EndPillarTypeFlags::Y_88__Diameter_7
	                       | EndPillarTypeFlags::Y_91__Diameter_7 | EndPillarTypeFlags::Y_94__Diameter_9
						   | EndPillarTypeFlags::Y_97__Diameter_9 | EndPillarTypeFlags::Y_100__Diameter_9
						   | EndPillarTypeFlags::Y_103__Diameter_11,
	Diagonal_Block_Width_2 = EndPillarTypeFlags::Diameter_5,
	Diagonal_Block_Width_3 = EndPillarTypeFlags::Diameter_7,
	Diagonal_Block_Width_4 = EndPillarTypeFlags::Diameter_9,
	Diagonal_Block_Width_6 = EndPillarTypeFlags::Diameter_11,

};
constexpr [[nodiscard]] EndPillarTypeFlags operator&(const EndPillarTypeFlags left, const EndPillarTypeFlags right) {
	return static_cast<EndPillarTypeFlags>((static_cast<int32_t>(left) & static_cast<int32_t>(right)) & static_cast<int32_t>(EndPillarTypeFlags::Unknown));
}
constexpr [[nodiscard]] EndPillarTypeFlags operator|(const EndPillarTypeFlags left, const EndPillarTypeFlags right) {
	return static_cast<EndPillarTypeFlags>((static_cast<int32_t>(left) | static_cast<int32_t>(right)) & static_cast<int32_t>(EndPillarTypeFlags::Unknown));
}

constexpr uint32_t NUMBER_OF_PILLARS = getNumberOfOnesIn(EndPillarTypeFlags::Unknown);

enum EndPillarLayoutOrientation {
	List_Goes_Clockwise        = 1,
	List_Goes_Counterclockwise = -1, List_Goes_Anticlockwise = EndPillarLayoutOrientation::List_Goes_Counterclockwise,
	Unknown
};

struct EndPillarsInputData : InputData {
	EndPillarLayoutOrientation layoutOrientation;
	uint8_t eastmostPillarIndex;
	EndPillarTypeFlags knownPillars[NUMBER_OF_PILLARS];
};

// The various test data.
#include "Test Data/1.9+.cuh"

#endif