#pragma once
#include <iostream>
#include <string>
#include <cstdio>
#include <memory>
#include <cassert>

namespace common {

	#define INFO(value,...) Info(value,##__VA_ARGS__)

	void Info(const char* value, ...);

	bool fileExists(const std::string& filePath);
}

