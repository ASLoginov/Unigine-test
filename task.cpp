#include "task.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <thread>

#include "immintrin.h"


inline __m256 vLength2(__m256& x, __m256& y)
{
	return _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));
}

void checkForThread(int a, int b, const std::vector<unit>& all_units, std::vector<int>& results, std::vector<float>& xPositions, std::vector<float>& yPositions)
{
	int size = all_units.size();
	float floatArrX[8];
	__m256 fov_tan;
	__m256 dist2;
	__m256 dirX;
	__m256 dirY;
	__m256 obsPosX;
	__m256 obsPosY;
	__m256 posX;
	__m256 posY;
	__m256 vLen2;
	__m256 rotatedX;
	__m256 rotatedY;
	__m256 fCounter;
	__m256 zero = _mm256_setzero_ps();
	__m256 abs = _mm256_set1_ps(-0.0f);
	__m256 mask;
	__m256 oneMask = _mm256_set1_ps(1.0f);
	float forTan = M_PI / 360;
	int end;
	int result;

	unit obs;

	for (int i = a; i < b; i++)
	{
		obs = all_units[i];
		fCounter = zero;
		result = 0;
		fov_tan = _mm256_set1_ps(tanf(obs.fov_deg * forTan));
		dist2 = _mm256_set1_ps(obs.distance * obs.distance);
		obsPosX = _mm256_set1_ps(obs.position.x);
		obsPosY = _mm256_set1_ps(obs.position.y);
		dirX = _mm256_set1_ps(obs.direction.x);
		dirY = _mm256_set1_ps(obs.direction.y);

		for (int j = 0; j < size; j += 8)
		{
			end = 8;
			mask = oneMask;
			if (j + 8 > size)
			{
				end = all_units.size() - j;
				_mm256_storeu_ps(floatArrX, zero);
				for (int e = 0; e < end; e++) floatArrX[e] = 1.0f;
				mask = _mm256_loadu_ps(floatArrX);
			}

			posX = _mm256_loadu_ps(xPositions.data() + j);
			posY = _mm256_loadu_ps(yPositions.data() + j);
			posX = _mm256_sub_ps(posX, obsPosX);
			posY = _mm256_sub_ps(posY, obsPosY);

			vLen2 = vLength2(posX, posY);

			mask = _mm256_and_ps(mask, _mm256_cmp_ps(vLen2, dist2, _CMP_LT_OS));

			rotatedX = _mm256_add_ps(_mm256_mul_ps(posX, dirX), _mm256_mul_ps(posY, dirY));

			mask = _mm256_and_ps(mask, _mm256_cmp_ps(rotatedX, zero, _CMP_GT_OQ));

			rotatedY = _mm256_andnot_ps(abs, _mm256_sub_ps(_mm256_mul_ps(posY, dirX), _mm256_mul_ps(posX, dirY)));
			rotatedX = _mm256_mul_ps(rotatedX, fov_tan);

			mask = _mm256_and_ps(mask, _mm256_cmp_ps(rotatedY, rotatedX, _CMP_LT_OS));

			fCounter = _mm256_add_ps(fCounter, mask);
		}
		
		for (int m = 0; m < 8; m++) result += ((float*)&fCounter)[m];
		results[i] = result;
	}
}

void Task::checkVisible(const std::vector<unit>& input_units, std::vector<int>& result)
{
	std::vector<float> xPositions;
	xPositions.resize(input_units.size());
	std::vector<float> yPositions;
	yPositions.resize(input_units.size());
	for (int u = 0; u < input_units.size(); u++)
	{
		xPositions[u] = input_units[u].position.x;
		yPositions[u] = input_units[u].position.y;
	}
	
	size_t threadsCount = 3;
	int blockSize = input_units.size() / (threadsCount + 1);
	std::vector<std::thread> threads;

	result.resize(input_units.size());

	for (int i = 0; i < threadsCount; i++)
		threads.push_back(std::thread(checkForThread, i * blockSize, (i + 1) * blockSize, std::ref(input_units), std::ref(result), std::ref(xPositions), std::ref(yPositions)));

	checkForThread(blockSize * threadsCount, input_units.size(), input_units, result, std::ref(xPositions), std::ref(yPositions));

	for (auto& thr : threads) thr.join();
}
