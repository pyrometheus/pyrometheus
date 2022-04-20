//  c++ -fPIC -shared -I. wrapper.cpp -I ~/src/env-3.9/lib/python3.9/site-packages/pybind11/include -I/usr/include/python3.9 -lpython3.9 -o mech_test.so           

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <mechs/sanDiego.hpp>
#include <vector>


namespace py = pybind11;


typedef std::vector<double> container_t;
PYBIND11_MAKE_OPAQUE(container_t);


typedef thermochemistry<container_t, double> tchem_t;


#define DEF_STATIC_METHOD(name) \
    def_static(#name, &cls::name)
PYBIND11_MODULE(mech_test, m)
{
    {
        typedef tchem_t cls ;
        py::class_<cls>(m, "Thermochemistry")
            .def(py::init<>())
            .DEF_STATIC_METHOD(get_specific_gas_constant)
            .DEF_STATIC_METHOD(get_density)
            .DEF_STATIC_METHOD(get_species_specific_heats_r)
            ;
    }

    py::bind_vector<container_t>(m, "VectorDouble");
}


