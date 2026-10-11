# use this file to set the compiler and flags for SYCL

set(CMPLR_ROOT $ENV{CMPLR_ROOT})
message(STATUS "CMPLR_ROOT: ${CMPLR_ROOT}")
if(WIN32)
  # Windows: CMake's Platform/Windows-IntelLLVM.cmake includes Windows-MSVC unconditionally and emits MSVC-style
  # flags (-QMD, /link, -fuse-ld=llvm-lib) for every IntelLLVM driver, so the GNU-command-line icpx.exe cannot be
  # driven by CMake here (2026-10-10: "icpx: error: unknown argument: '-nologo' / '-EHsc' / '-QMD'"). icx.exe is the
  # MSVC-command-line driver and the one torch's cpp_extension uses on Windows; it compiles SYCL with -fsycl too.
  set(CMAKE_CXX_COMPILER ${CMPLR_ROOT}/bin/icx.exe)
  set(CMAKE_C_COMPILER ${CMPLR_ROOT}/bin/icx.exe)
else()
  set(CMAKE_CXX_COMPILER ${CMPLR_ROOT}/bin/icpx)
  set(CMAKE_C_COMPILER ${CMPLR_ROOT}/bin/icx)
endif()
