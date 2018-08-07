#include <random>
#include <iostream>
#include <algorithm>
#include "range.hpp"
template <typename T, int N, int  min, int  max>
class Random
{
    public:
        Random() {};
        T* generateRandomArray()
        {
            T* _array = new T [N*N ];
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(min, max);
            for (auto i : util::lang::range(0, N*N  ).step(1))
            {
                _array[i] = dis(gen);
            }
            return _array;
        }

};
