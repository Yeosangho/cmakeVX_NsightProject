mkdir -p vsStabilizer/bin
cd vsStabilizer
#cmake -G "Eclipse CDT4 - Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug -D CMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE ../src/
cmake -G "Eclipse CDT4 - Unix Makefiles"  -D CMAKE_BUILD_TYPE=Debug ../src
