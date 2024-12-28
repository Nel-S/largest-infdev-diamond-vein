#ifndef __GENERAL__ALLOWED_VALUES_FOR_SETTINGS_CUH
#define __GENERAL__ALLOWED_VALUES_FOR_SETTINGS_CUH

// The list of supported seedcracking tools.
enum CrackerType {
	Veins,
	End_Pillars
};

// The list of all Java Edition versions.
enum Version {
	Rd_132211,
	Rd_132328,
	Rd_160052,
	Rd_161348,
	Classic_0_0_11a,
	Classic_0_0_12a_03,
	Classic_0_0_13a, Classic_0_0_13a_03,
	Classic_0_0_14a_08,
	Classic_0_0_15a,
	Classic_0_0_16a_02,
	Classic_0_0_17a,
	Classic_0_0_18a_02,
	Classic_0_0_19a_04, Classic_0_0_19a_06,
	Classic_0_0_20a_01, Classic_0_0_20a_02,
	Classic_0_0_21a,
	Classic_0_0_22a_05,
	Classic_0_0_23a_01,
	Classic_0_24_03,
	Classic_0_25_05,
	Classic_0_27,
	Classic_0_28_01,
	Classic_0_29, Classic_0_29_01, Classic_0_29_02,
	Classic_0_30,
	Indev_20091223_3,
	Indev_20091231_2,
	Indev_20100104,
	Indev_20100110,
	Indev_20100124_2,
	Indev_20100125,
	Indev_20100128,
	Indev_20100129,
	Indev_20100130,
	Indev_20100131,
	Indev_20100201_1, Indev_20100201_2,
	Indev_20100202,
	Indev_20100206_3,
	Indev_20100207_1, Indev_20100207_2,
	Indev_20100212_1, Indev_20100212_2,
	Indev_20100213,
	Indev_20100214,
	Indev_20100218,
	Indev_20100219,
	Indev_20100223,
	Infdev_20100227_2,
	Infdev_20100313,
	Infdev_20100316,
	Infdev_20100320,
	Infdev_20100321,
	Infdev_20100325,
	Infdev_20100327,
	Infdev_20100330_2,
	Infdev_20100413,
	Infdev_20100414,
	Infdev_20100415,
	Infdev_20100420,
	Infdev_20100607,
	Infdev_20100608,
	Infdev_20100611,
	Infdev_20100615,
	Infdev_20100616_1,
	Infdev_20100617_1, Infdev_20100617_2,
	Infdev_20100618,
	Infdev_20100624,
	Infdev_20100625_1, Infdev_20100625_2,
	Infdev_20100627,
	Infdev_20100629,
	Infdev_20100630,
	Alpha_1_0_0, Alpha_1_0_1_01, Alpha_1_0_2_01, Alpha_1_0_2_02, Alpha_1_0_3, Alpha_1_0_4, Alpha_1_0_5_01, Alpha_1_0_6, Alpha_1_0_6_01,
		Alpha_1_0_6_03 , Alpha_1_0_7    , Alpha_1_0_8_01, Alpha_1_0_9 , Alpha_1_0_10   , Alpha_1_0_11   , Alpha_1_0_12   ,
		Alpha_1_0_13   , Alpha_1_0_13_01, Alpha_1_0_14  , Alpha_1_0_15, Alpha_1_0_16_01, Alpha_1_0_16_02, Alpha_1_0_17_02, Alpha_1_0_17_03, Alpha_1_0_17_04,
	Alpha_1_1_0, Alpha_1_1_1, Alpha_1_1_2, Alpha_1_1_2_01,
	Alpha_1_2_0_01, Alpha_1_2_0_02, Alpha_1_2_1_01, Alpha_1_2_2, Alpha_1_2_3, Alpha_1_2_3_01, Alpha_1_2_3_02, Alpha_1_2_3_04,
		Alpha_1_2_3_05, Alpha_1_2_4_01, Alpha_1_2_5, Alpha_1_2_6, 
	Beta_1_0, Beta_1_0_01, Beta_1_0_2,
	Beta_1_1, Beta_1_1_01, Beta_1_1_02,
	Beta_1_2, Beta_1_2_01, Beta_1_2_02,
	Beta_1_3, Beta_1_3_01,
	Beta_1_4, Beta_1_4_01,
	Beta_1_5, Beta_1_5_01, Beta_1_5_02,
	Beta_1_6, Beta_1_6_1 , Beta_1_6_2 , Beta_1_6_3, Beta_1_6_4, Beta_1_6_5, Beta_1_6_6,
	Beta_1_7, Beta_1_7_01, Beta_1_7_2 , Beta_1_7_3,
	Beta_1_8, Beta_1_8_1,
	v1_0,
	v1_1,
	v1_2  , v1_2_1 , v1_2_2 , v1_2_3 , v1_2_4 , v1_2_5 ,
	v1_3_1, v1_3_2 ,
	v1_4_2, v1_4_4 , v1_4_5 , v1_4_6 , v1_4_7 ,
	v1_5  , v1_5_1 , v1_5_2 ,
	v1_6_1, v1_6_2 , v1_6_4 ,
	v1_7_2, v1_7_4 , v1_7_5 , v1_7_6 , v1_7_7 , v1_7_8 , v1_7_9 , v1_7_10,
	v1_8  , v1_8_1 , v1_8_2 , v1_8_3 , v1_8_4 , v1_8_5 , v1_8_6 , v1_8_7 , v1_8_8, v1_8_9,
	v1_9  , v1_9_1 , v1_9_2 , v1_9_3 , v1_9_4 ,
	v1_10 , v1_10_1, v1_10_2,
	v1_11 , v1_11_1, v1_11_2,
	v1_12 , v1_12_1, v1_12_2,
	v1_13 , v1_13_1, v1_13_2,
	v1_14 , v1_14_1, v1_14_2, v1_14_3, v1_14_4,
	v1_15 , v1_15_1, v1_15_2,
	v1_16 , v1_16_1, v1_16_2, v1_16_3, v1_16_4, v1_16_5,
	v1_17 , v1_17_1,
	v1_18 , v1_18_1, v1_18_2,
	v1_19 , v1_19_1, v1_19_2, v1_19_3, v1_19_4,
	v1_20 , v1_20_1, v1_20_2, v1_20_3, v1_20_4, v1_20_5, v1_20_6,
	v1_21 , v1_21_1, v1_21_2, v1_21_3, v1_21_4,
};

enum Biome {
	Default_or_Unknown
};


// The attributes that all input data must contain:
struct InputData {
	// The type of seedcracker to use
	CrackerType type;
	// The game version
	Version version;
};

enum InputLayoutType {

};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/* WARNING: These are experimental features that don't work yet.*/

enum ExperimentalBiome {
	Badlands = Biome::Default_or_Unknown + 1, Mesa = Badlands,
	Basalt_Deltas,
};

#endif