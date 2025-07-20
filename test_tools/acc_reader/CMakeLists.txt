cmake_minimum_required(VERSION 3.16)
project(AccelerometerAudioRecorder)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows 플랫폼 확인
if(NOT WIN32)
    message(FATAL_ERROR "This project is designed for Windows only")
endif()

# 소스 파일들
set(SOURCES
    main.cpp
    acc.cpp
    audio.cpp
)

# 헤더 파일들
set(HEADERS
    acc.h
    audio.h
)

# 실행 파일 생성
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Windows 특정 라이브러리 링크
target_link_libraries(${PROJECT_NAME}
    sensorsapi
    propsys
    ole32
    avrt
)

# 컴파일러 특정 설정
if(MSVC)
    # MSVC 컴파일러 설정
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _WIN32_DCOM
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
    
    # 경고 레벨 설정
    target_compile_options(${PROJECT_NAME} PRIVATE /W3)
    
    # 멀티스레드 런타임 사용
    set_property(TARGET ${PROJECT_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# 디버그 정보 포함
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME} PRIVATE _DEBUG)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)
endif()

# 출력 디렉토리 설정
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
)

# 설치 설정
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# 프로젝트 정보 출력
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")