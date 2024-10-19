cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(Pyrometheus)

option(TEST_python    "Test the Python  Backend." ON)
option(TEST_cpp       "Test the C++     Backend." ON)
option(TEST_fortran90 "Test the Fortran Backend." ON)

set(TEST_MECHANISMS "uiuc;sandiego;uconn32;gri30;hong")

enable_testing()

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_program(PYTEST_EXE pytest REQUIRED)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/mechs")

function (pyro_generate_backend_impl)
    cmake_parse_arguments(ARGS "" "MECHANISM;BACKEND;NAME;OUTPUT" "" ${ARGN})

    set(PYRO_DIR "${CMAKE_SOURCE_DIR}/../pyrometheus")

    add_custom_command(
        OUTPUT  "${ARGS_OUTPUT}"
        DEPENDS "${ARGS_MECHANISM}"
                "${PYRO_DIR}/chem_expr.py"
                "${PYRO_DIR}/__init__.py"
                "${PYRO_DIR}/codegen/__init__.py"
                "${PYRO_DIR}/codegen/${ARGS_BACKEND}.py"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/codegen.py"
            --mech "${ARGS_MECHANISM}"  --backend "${ARGS_BACKEND}"
            --name "${ARGS_NAME}"       --output  "${ARGS_OUTPUT}"   
    )
endfunction()


function (pyro_generate_test)
    cmake_parse_arguments(ARGS "" "BACKEND" "" ${ARGN})

    add_test(
        NAME    pyro_${ARGS_BACKEND}
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_BINARY_DIR}"
            "${PYTEST_EXE}" "${CMAKE_SOURCE_DIR}/test.py"
                            --backend ${ARGS_BACKEND}
    )
endfunction()


if (TEST_python)
    add_custom_target(pyro_python ALL)

    foreach (mechanism ${TEST_MECHANISMS})
        set(mech_in "${CMAKE_SOURCE_DIR}/mechs/${mechanism}.yaml")
        set(py_out  "${CMAKE_BINARY_DIR}/libpyro_python_${mechanism}.py")

        pyro_generate_backend_impl(
            MECHANISM "${mech_in}"    BACKEND python
            NAME      "${mechanism}"  OUTPUT  "${py_out}"
        )

        add_custom_target(pyro_python_${mechanism} ALL DEPENDS "${py_out}")
        add_dependencies(pyro_python pyro_python_${mechanism})
    endforeach()

    pyro_generate_test(BACKEND python)
endif()


if (TEST_cpp)
    enable_language(CXX)
    find_package(pybind11 REQUIRED)

    add_custom_target(pyro_cpp ALL)

    foreach (mechanism ${TEST_MECHANISMS})
        set(mech_in     "${CMAKE_SOURCE_DIR}/mechs/${mechanism}.yaml")
        set(header_out  "${CMAKE_BINARY_DIR}/mechs/${mechanism}.hxx")
        set(binding_out "${CMAKE_BINARY_DIR}/binding_${mechanism}.cxx")

        pyro_generate_backend_impl(
            MECHANISM "${mech_in}"    BACKEND cpp
            NAME      "${mechanism}"  OUTPUT  "${header_out}"
        )

        configure_file("${CMAKE_SOURCE_DIR}/binding.cxx.in" "${binding_out}")

        add_library(pyro_cpp_${mechanism} MODULE "${binding_out}" "${header_out}")
        target_include_directories(pyro_cpp_${mechanism} PRIVATE "${CMAKE_BINARY_DIR}/mechs")
        target_link_libraries(pyro_cpp_${mechanism} PRIVATE pybind11::embed)

        add_dependencies(pyro_cpp pyro_cpp_${mechanism})
    endforeach()

    pyro_generate_test(BACKEND cpp)
endif()


if (TEST_fortran90)
    enable_language(Fortran)

    execute_process(COMMAND "${Python3_EXECUTABLE}"
        -c "import numpy.f2py; print(numpy.f2py.get_include())"
        OUTPUT_VARIABLE F2PY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    add_custom_target(pyro_fortran90 ALL)

    foreach (mechanism ${TEST_MECHANISMS})
        set(lib_name   "libpyro_fortran90_${mechanism}")
        set(mech_in    "${CMAKE_SOURCE_DIR}/mechs/${mechanism}.yaml")
        set(module_out "${CMAKE_BINARY_DIR}/mechs/${mechanism}.f90")
        set(f2py_out_c "${CMAKE_BINARY_DIR}/${lib_name}module.c")
        set(f2py_out_f "${CMAKE_BINARY_DIR}/${lib_name}-f2pywrappers2.f90")

        pyro_generate_backend_impl(
            MECHANISM "${mech_in}"    BACKEND fortran90
            NAME      "${lib_name}"   OUTPUT  "${module_out}"
        )

        add_custom_command(
            OUTPUT  "${f2py_out_c}" "${f2py_out_f}"
            COMMAND ${Python3_EXECUTABLE} -m "numpy.f2py" "${module_out}"
                --lower  -m "${lib_name}" 
                --f2cmap "${CMAKE_SOURCE_DIR}/.f2py_f2cmap"
            DEPENDS ${module_out}
        )

        add_library(pyro_fortran90_${mechanism} MODULE
            "${f2py_out_c}" "${f2py_out_f}"
            "${F2PY_INCLUDE_DIR}/fortranobject.c" "${module_out}")
        target_link_libraries(pyro_fortran90_${mechanism} PRIVATE Python3::NumPy Python3::Module)
        target_compile_definitions(pyro_fortran90_${mechanism} PRIVATE "PYROMETHEUS_CALLER_INDEXING=1")
        target_include_directories(pyro_fortran90_${mechanism} PRIVATE "${F2PY_INCLUDE_DIR}")

        set_target_properties(pyro_fortran90_${mechanism} PROPERTIES
            Fortran_PREPROCESS ON
        )

        add_dependencies(pyro_fortran90 pyro_fortran90_${mechanism})
    endforeach()

    pyro_generate_test(BACKEND fortran90)
endif()
