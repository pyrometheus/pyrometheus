//  c++ -std=c++11 -Xlinker -undefined -Xlinker dynamic_lookup -fPIC -shared -I. wrapper.cpp -I ~/Packages/miniconda3/lib/python3.8/site-packages/pybind11/include -I ~/Packages/miniconda3/include/python3.8  -L ~/Packages/miniconda3/lib/ -o pyro_cpp.so

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <include/uiuc.h>
#include <include/sandiego.h>
#include <include/uconn32.h>
#include <vector>


namespace py = pybind11;


typedef std::vector<double> container_t;
PYBIND11_MAKE_OPAQUE(container_t);


typedef uiuc::thermochemistry<container_t, double> uiuc_t;
typedef sandiego::thermochemistry<container_t, double> sandiego_t;
typedef uconn32::thermochemistry<container_t, double> uconn_t;


#define DEF_STATIC_METHOD(name)			\
  def_static(#name, &cls::name)
PYBIND11_MODULE(pyro_cpp, m)
{
  {
    typedef uiuc_t cls;
    py::class_<cls>(m, "uiuc")
      .def(py::init<>())
      .DEF_STATIC_METHOD(get_specific_gas_constant)
      .DEF_STATIC_METHOD(get_mix_molecular_weight)
      .DEF_STATIC_METHOD(get_concentrations)
      .DEF_STATIC_METHOD(get_mass_average_property)
      .DEF_STATIC_METHOD(get_mixture_specific_heat_cp_mass)
      .DEF_STATIC_METHOD(get_mixture_enthalpy_mass)
      .DEF_STATIC_METHOD(get_density)
      .DEF_STATIC_METHOD(get_species_specific_heats_r)
      .DEF_STATIC_METHOD(get_species_enthalpies_rt)
      .DEF_STATIC_METHOD(get_species_entropies_r)
      .DEF_STATIC_METHOD(get_species_gibbs_rt)
      .DEF_STATIC_METHOD(get_equilibrium_constants)
      .DEF_STATIC_METHOD(get_temperature)
      .DEF_STATIC_METHOD(get_fwd_rate_coefficients)
      .DEF_STATIC_METHOD(get_net_rates_of_progress)
      .DEF_STATIC_METHOD(get_net_production_rates)
      ;
  }
  
  {
    typedef sandiego_t cls;
    py::class_<cls>(m, "sandiego")
      .def(py::init<>())
      .DEF_STATIC_METHOD(get_specific_gas_constant)
      .DEF_STATIC_METHOD(get_mix_molecular_weight)
      .DEF_STATIC_METHOD(get_concentrations)
      .DEF_STATIC_METHOD(get_mass_average_property)
      .DEF_STATIC_METHOD(get_mixture_specific_heat_cp_mass)
      .DEF_STATIC_METHOD(get_mixture_enthalpy_mass)
      .DEF_STATIC_METHOD(get_density)
      .DEF_STATIC_METHOD(get_species_specific_heats_r)
      .DEF_STATIC_METHOD(get_species_enthalpies_rt)
      .DEF_STATIC_METHOD(get_species_entropies_r)
      .DEF_STATIC_METHOD(get_species_gibbs_rt)
      .DEF_STATIC_METHOD(get_equilibrium_constants)
      .DEF_STATIC_METHOD(get_temperature)
      .DEF_STATIC_METHOD(get_fwd_rate_coefficients)
      .DEF_STATIC_METHOD(get_net_rates_of_progress)
      .DEF_STATIC_METHOD(get_net_production_rates)
      ;
  }

  {
    typedef uconn_t cls;
    py::class_<cls>(m, "uconn32")
      .def(py::init<>())
      .DEF_STATIC_METHOD(get_specific_gas_constant)
      .DEF_STATIC_METHOD(get_mix_molecular_weight)
      .DEF_STATIC_METHOD(get_concentrations)
      .DEF_STATIC_METHOD(get_mass_average_property)
      .DEF_STATIC_METHOD(get_mixture_specific_heat_cp_mass)
      .DEF_STATIC_METHOD(get_mixture_enthalpy_mass)
      .DEF_STATIC_METHOD(get_density)
      .DEF_STATIC_METHOD(get_species_specific_heats_r)
      .DEF_STATIC_METHOD(get_species_enthalpies_rt)
      .DEF_STATIC_METHOD(get_species_entropies_r)
      .DEF_STATIC_METHOD(get_species_gibbs_rt)
      .DEF_STATIC_METHOD(get_equilibrium_constants)
      .DEF_STATIC_METHOD(get_temperature)
      .DEF_STATIC_METHOD(get_fwd_rate_coefficients)
      .DEF_STATIC_METHOD(get_net_rates_of_progress)
      .DEF_STATIC_METHOD(get_net_production_rates)
      ;
  }

  py::bind_vector<container_t>(m, "VectorDouble");
}


