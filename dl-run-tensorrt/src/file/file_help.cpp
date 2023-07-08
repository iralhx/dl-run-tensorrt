#include<file\file_help.h>




bool fileExists(const std::string& filePath) {
    return std::filesystem::exists(filePath);
}
