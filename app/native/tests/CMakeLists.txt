# fb_native tests. Build with -DFB_BUILD_TESTS=ON; run with ctest, or directly:  bin/fb_tests [name-filter]
find_program(FB_PYTHON NAMES python.exe python
  HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../.venv/Scripts" NO_DEFAULT_PATH)
if(NOT FB_PYTHON)
  find_program(FB_PYTHON NAMES python python3)
endif()

set(FB_TEST_ENGINE_DIR "${CMAKE_BINARY_DIR}/test_engines")
file(MAKE_DIRECTORY "${FB_TEST_ENGINE_DIR}")
set(FB_REAL_ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../models/engines")
file(TO_NATIVE_PATH "${FB_TEST_ENGINE_DIR}" FB_TEST_ENGINE_DIR_NATIVE)
file(TO_NATIVE_PATH "${FB_REAL_ENGINE_DIR}" FB_REAL_ENGINE_DIR_NATIVE)
string(REPLACE "\\" "\\\\" FB_TEST_ENGINE_DIR_ESC "${FB_TEST_ENGINE_DIR_NATIVE}")
string(REPLACE "\\" "\\\\" FB_REAL_ENGINE_DIR_ESC "${FB_REAL_ENGINE_DIR_NATIVE}")

add_executable(fb_tests
  test_main.cpp test_color.cpp test_tile.cpp test_misc.cpp test_engine.cpp)
target_link_libraries(fb_tests PRIVATE fb_native)
target_include_directories(fb_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(fb_tests PRIVATE
  NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
  "FB_TEST_ENGINE_DIR=\"${FB_TEST_ENGINE_DIR_ESC}\""
  "FB_REAL_ENGINE_DIR=\"${FB_REAL_ENGINE_DIR_ESC}\"")
if(MSVC)
  target_compile_options(fb_tests PRIVATE /W4 /wd4100 /utf-8)
endif()

# Test engines are built with the python tensorrt package (version-compatible, like the production engines).
if(FB_PYTHON)
  add_test(NAME make_test_engines
           COMMAND "${FB_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/make_test_engines.py" "${FB_TEST_ENGINE_DIR}")
  set_tests_properties(make_test_engines PROPERTIES FIXTURES_SETUP fb_engines TIMEOUT 300)
else()
  message(WARNING "python not found: engine tests need pre-built engines in ${FB_TEST_ENGINE_DIR}")
endif()

# One ctest entry per test group (filter = name prefix substring).
foreach(grp color_ tensor_ memory_ tiling_ device_ error_ streams_ scene_)
  add_test(NAME ${grp} COMMAND fb_tests ${grp})
  set_tests_properties(${grp} PROPERTIES TIMEOUT 300)
endforeach()
foreach(grp engine_ leak_)
  add_test(NAME ${grp} COMMAND fb_tests ${grp})
  set_tests_properties(${grp} PROPERTIES TIMEOUT 600)
  if(FB_PYTHON)
    set_tests_properties(${grp} PROPERTIES FIXTURES_REQUIRED fb_engines)
  endif()
endforeach()
