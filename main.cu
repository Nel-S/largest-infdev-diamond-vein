#include "src/Veins/Filters.cuh"
#include <thread>


void deviceManager(int32_t deviceIndex) {
	TRY_CUDA(cudaSetDevice(deviceIndex));
	initialFilter();
}

int main() {
	std::thread threads[NUMBER_OF_DEVICES];
	time_t startTime = time(NULL), currentTime;
	for (int32_t i = 0; i < NUMBER_OF_DEVICES; ++i) threads[i] = std::thread(deviceManager, i);

	if (STATUS_FREQUENCY.count()) {
		// Wait until some progress is made, to avoid division by zero in ETA and to get a more accurate estimate
		while (!globalCurrentIteration) std::this_thread::sleep_for(std::chrono::seconds(10));

		while (globalCurrentIteration < GLOBAL_ITERATIONS_NEEDED) {
			time(&currentTime);
			// Calculate estimated seconds until finishing
			double eta = static_cast<double>(GLOBAL_ITERATIONS_NEEDED - globalCurrentIteration) * static_cast<double>(currentTime - startTime) / static_cast<double>(globalCurrentIteration);
			fprintf(stderr, "(%" PRIu64 "/%" PRIu64 " chunks searched; ETA = %.2f seconds)\n", globalCurrentIteration*WORKERS_PER_DEVICE, TOTAL_CHUNKS, eta);
			std::this_thread::sleep_for(STATUS_FREQUENCY);
		}
	}
	return 0;
}