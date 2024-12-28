#ifndef __GENERAL__FILTERS_CUH
#define __GENERAL__FILTERS_CUH

#include "General Settings Processing.cuh"

struct Feature {
	
	virtual bool initialFilter(uint64_t *outputArray, size_t *outputArraySize, size_t outputArrayCapacity) {
		if (!outputArray) throw std::invalid_argument("Invalid pointer to output array provided.");
		if (!outputArraySize) throw std::invalid_argument("Invalid pointer to output array size provided.");
		if (!outputArrayCapacity) throw std::invalid_argument("Empty output array provided.");
	}
};

struct EndPillarFeature : Feature {
	bool initialFilter(uint64_t *outputArray, size_t *outputArraySize, size_t outputArrayCapacity) {
		Feature::initialFilter(outputArray, outputArraySize, outputArrayCapacity);
	}
};

#include <vector>

int temp() {
	EndPillarFeature endPillarFeature;
	std::vector<Feature> features;
	features.push_back(endPillarFeature);
}

#endif