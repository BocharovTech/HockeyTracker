cmake_minimum_required(VERSION 3.22)

project(HockeyTracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Для Apple Silicon задаём архитектуру
set(CMAKE_OSX_ARCHITECTURES arm64)

# Qt6 из Homebrew (на Apple Silicon)
set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/qtbase")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Core)
find_package(OpenCV REQUIRED)

# Указываем исходные файлы с абсолютными путями через CMAKE_CURRENT_SOURCE_DIR
set(PROJECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tracker.cpp
)

qt_add_executable(HockeyTracker
    ${PROJECT_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow.ui
)

target_link_libraries(HockeyTracker PRIVATE
    Qt6::Widgets
    Qt6::Core
    ${OpenCV_LIBS}
)

target_include_directories(HockeyTracker PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)