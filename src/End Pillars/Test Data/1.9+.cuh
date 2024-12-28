#include "../Allowed Values for Input Data.cuh"

// Pillar seed: ??
// Structure seed: ??
__device__ constexpr EndPillarsInputData END_PILLARS_TEST_DATA_1 = {
	CrackerType::End_Pillars,
	Version::v1_9,
	EndPillarLayoutOrientation::List_Goes_Clockwise,
	-1,
	{
		EndPillarTypeFlags::Not_Caged,
		EndPillarTypeFlags::Not_Caged & EndPillarTypeFlags::Diagonal_Block_Width_4,
		EndPillarTypeFlags::Not_Caged,
		EndPillarTypeFlags::Not_Caged,
		EndPillarTypeFlags::Not_Caged & EndPillarTypeFlags::Diagonal_Block_Width_4,
		EndPillarTypeFlags::Unknown,
		EndPillarTypeFlags::Unknown,
		EndPillarTypeFlags::Unknown,
		EndPillarTypeFlags::Unknown,
		EndPillarTypeFlags::Unknown,
	}
};

// Pillar seed: ??
// Structure seed: ??
// From the background of Epic10l2's Simple Pillarcracker (https://github.com/Epic10l2/Simple-Pillarcracker/blob/59aef78878ef033882f764e7aa2734b72143d059/Pillarseed%20bruteforcer/Pillars.png)
__device__ constexpr EndPillarsInputData END_PILLARS_TEST_DATA_2 = {
	CrackerType::End_Pillars,
	Version::v1_9,
	EndPillarLayoutOrientation::List_Goes_Clockwise,
	0,
	{
		EndPillarTypeFlags::Taller_Than_Caged  & EndPillarTypeFlags::Diagonal_Block_Width_3,
		EndPillarTypeFlags::Taller_Than_Caged  & EndPillarTypeFlags::Diagonal_Block_Width_4,
		EndPillarTypeFlags::Caged              & EndPillarTypeFlags::Diagonal_Block_Width_2,
		EndPillarTypeFlags::Taller_Than_Caged  & EndPillarTypeFlags::Diagonal_Block_Width_3,
		EndPillarTypeFlags::Shorter_Than_Caged & EndPillarTypeFlags::Diagonal_Block_Width_2,
		EndPillarTypeFlags::Taller_Than_Caged  & EndPillarTypeFlags::Diagonal_Block_Width_3,
		EndPillarTypeFlags::Caged              & EndPillarTypeFlags::Diagonal_Block_Width_2,
		EndPillarTypeFlags::Taller_Than_Caged  & EndPillarTypeFlags::Diagonal_Block_Width_6,
		EndPillarTypeFlags::Taller_Than_Caged  & EndPillarTypeFlags::Diagonal_Block_Width_4,
		EndPillarTypeFlags::Taller_Than_Caged  & EndPillarTypeFlags::Diagonal_Block_Width_4,
	}
};