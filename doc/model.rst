.. _sec:kinetics:

Chemical Kinetics
=================

Pyrometheus generates code to evaluate chemical source terms.
These appear in the conservation equations of reacting flows.
Here, we lay out the corresponding equations.
We focus on a homogeneous adiabatic reactor for simplicify. Yet, the
systems explained here can easily be adapted to other configurations
(e.g., isochoric or inhomogeneous reactors).

.. _subsec:species:

Species Conservation and Chemical Source Terms
----------------------------------------------

Our goal is to express, in as much detail, the chemical kinetics of
reactive flows. We assume a homogeneous mixture of ideal gases evolves
at constant pressure :math:`p` and enthalpy :math:`h_{0}`. We
characterize its chemical composition by the species mass fractions
:math:`\boldsymbol{y} = \{ y_{i} \}_{i = 1}^{N}`. These evolve from an initial
condition :math:`y_{i}(0) = y_{i}^{0}` according to

.. math::

   \label{eq:species_conservation}
     \frac{dy_{i}}{dt} = S_{i} \equiv \frac{ W_{i}\dot{\omega}_{i} }{ \rho },

where :math:`S_{i}` is the chemical source term of the
:math:`i^{\mathrm{th}}` species (in :math:`\mathrm{s}^{-1}`),
:math:`W_{i}` its molecular weight (in :math:`\mathrm{kg/kmol})` and
:math:`\dot{\omega}_{i}` its molar production rate (in
:math:`\mathrm{kmol/m^{3}-s}`). The mixture density :math:`\rho` (in
:math:`\mathrm{kg/m^{3}}`) is obtained from

.. math:: pW = \rho RT,

where

.. math:: W = \sum_{i = 1}^{N}W_{i}y_{i}

is the mixture molecular weight, :math:`R` the universal gas constant
(in :math:`\mathrm{J/kmol-K}`), and :math:`T` the temperature (in
:math:`\mathrm{K}`). We explain how to obtain the mixture temperature in
detail in Section `1.2 <#subsec:energy>`__.

To evaluate `[species_conservation] <#species_conservation>`__, we need
the net production rates
:math:`\dot{\boldsymbol{\omega}} = \{ \dot{\omega}_{i} \}_{i = 1}^{N}`. These
represent changes in composition due to chemical reactions

.. math::

   \label{eq:reactions}
     \sum_{\ell = 1}^{N}\nu_{i\ell}^{\prime}\mathcal{S}_{\ell} \rightleftharpoons \sum_{k = 1}^{N}\nu_{ik}^{\prime\prime}\mathcal{S}_{k},\qquad j = 1,\dots,M,

where :math:`\nu_{ij}^{\prime}` and :math:`\nu_{ij}^{\prime\prime}` are
the forward and reverse stoichiometric coefficients of species
:math:`\mathcal{S}_{i}` in the :math:`j^{\mathrm{th}}` reaction. Per
`[reactions] <#reactions>`__, species :math:`\mathcal{S}_{i}` can only
be produced (or destroyed) by an amount :math:`\nu_{ij}^{\prime\prime}`
(or :math:`\nu_{ij}^{\prime}`) in the :math:`j^{\mathrm{th}}` reaction.
Thus, :math:`\{ \dot{\omega}_{i} \}_{i = 1}^{N}` are linear combinations
of the reaction rates of progress :math:`R_{j}`,

.. math::

   \label{eq:production_rates}
     \dot{\omega}_{i} = \sum_{j = 1}^{M}\nu_{ij}R_{j},\qquad i = 1,\dots,N,

where :math:`\nu_{ij} = \nu_{ij}^{\prime\prime} - \nu_{ij}^{\prime}` is
the net stoichiometric coefficient of the :math:`i^{\mathrm{th}}`
species in the :math:`j^{\mathrm{j}}` reaction. The rates of progress
are given by the law of mass-action,

.. math::

   \label{eq:reaction_rates}
     R_{j} = k_{j}(T)\left[  \prod_{\ell = 1}^{N}\left(\frac{ \rho y_{\ell} }{ W_{\ell} }\right)^{\nu_{mj}^{\prime}} - \frac{1}{K_{j}(T)}\prod_{k = 1}^{N}\left(\frac{ \rho y_{k} }{ W_{k} }\right)^{\nu_{mj}^{\prime\prime}} \right],\qquad j = 1,\dots,M,

where :math:`k_{j}(T)` is the rate coefficient of the
:math:`j^{\mathrm{th}}` reaction and :math:`K_{j}(T)` its equilibrium
constant. Depending on the reaction, the rate coefficient
:math:`k_{j}(T)` may take different forms (and even become a function of
pressure). Its simplest form is the Arrhenius expression,

.. math::

   \label{eq:rate_coeff}
     k_{j}(T) = A_{j}T^{b_{j}}\exp\left({ -\frac{\theta_{a,j}}{T} }\right),\qquad j = 1,\dots,M

where :math:`A_{j}` is the pre-exponential, :math:`b_{j}` is the
temperature exponent, and :math:`\theta_{a,j}` is the activation
temperature.

The equilibrium constant is evaluated through equilibrium thermodynamics

.. math::

   \label{eq:equil_constants}
     K_{j}(T) = \left( \frac{p_{0}}{RT} \right)^{\sum_{i = 0}^{\nu_{ij}}}\exp\left( -\sum_{i = 1}^{N}\frac{\nu_{ij}g_{i}(T)}{RT} \right),\qquad j = 1,\dots,M,

where :math:`p_{0} = 1` :math:`\mathrm{atm}` and

.. math:: g_{i}(T) = h_{i}(T) - T\,s_{i}(T),\qquad i = 1,\dots,N

are the species Gibbs functions, with :math:`h_{i}` and :math:`s_{i}`
the species enthalpies and entropies.

.. _subsec:thermo:

Species Thermodynamics
~~~~~~~~~~~~~~~~~~~~~~

.. _subsec:energy:

Conservation of Energy
----------------------

To evaluate the rates of
progress `[reaction_rates] <#reaction_rates>`__, we need the
temperature. Yet, we have defered any discussion on how to compute it
from other state variables.

.. _sec:transport:

Transport coefficients
======================

Pyrometheus generates code to evaluate transport coefficients of single species and mixtures based on the kinetic theory of gases where the electrical potential between the atoms and molecules dictates the macroscopic properties of the flow. A complete overview and description of the formulation is presented in chapter 12 of [Kee_2003]_.

.. _subsec:Viscosity:

The fluid viscosity depends on the mixture composition given by :math:`X_k` mole fraction and pure species viscosity :math:`\mu_k` of the individual species. The latter are obtained according to 

.. math::

    \mu_k = 2.6693 \times 10^{-6} \frac{[T W_k]^{\frac{1}{2}}}{\sigma_{i}^2 \Omega^{(2,2)}_{i}(T, \epsilon, k_B, \delta_k)}

with units :math:`\frac{kg}{m-s}`. In this equation, :math:`W` is the molecular weight, :math:`\sigma` is the net collision diameter according to Lennard-Jones potential, :math:`\Omega^{(2,2)}` is the collision integral as a function of temperature :math:`T`, dipole moment :math:`\delta`, well-depth :math:`\epsilon` and the Boltzmann constant :math:`k_B`. In practice, the atom or molecule geometrical properties are given in the mechanism file in the variables ``geometry``, ``diameter``, ``well-depth``, ``polarizability`` and ``rotational-relaxation``.

Finally, the collision integral is tabulated for fast evaluation of the transport coefficients. Thus, with all these variables, an interpolation function can be obtained, yielding the respective species viscosities:

.. math::

    \mu_k(T) = \sqrt{T} (A + B \, log(T) + C \, log(T)^2 + D \, log(T)^3 + E \, log(T)^4)^n

The coefficients :math:`A` to :math:`E` depends on the respective species as a function of all the aforementioned variables. For the viscosity, the exponent is :math:`n=2`. 

Then, a mixture rule is employed to weight the contribution of the individual species to the fluid viscosity and it is given by

.. math::

    \mu^{(m)} = \sum_{k=1}^{K} \frac{X_k \mu_k}{\sum_{j=1}^{K} X_j\phi_{kj}}

where

.. math::

    \phi_{kj} = \frac{1}{\sqrt{8}}
    \left( 1 + \frac{W_k}{W_j} \right)^{-\frac{1}{2}}
    \left( 1 + \left[ \frac{\mu_k}{\mu_j} \right]^{\frac{1}{2}}
    \left[ \frac{W_j}{W_k} \right]^{\frac{1}{4}} \right)^2

.. _subsec:Thermal conductivity:

The thermal conductivity of the indidividual species can be obtained from the viscosity according to

.. math::

    \lambda = \mu c_v

where :math:`cv` is the specific heat at constant volume. Assuming that the individual species conductivities are composed of translational, rotational, and vibrational contributions, the thermal conductivity is evaluated as

.. math::

    \lambda = \mu (f_{trans} c_{v_{trans}} + f_{rot} c_{v_{rot}} + f_{vib} c_{v_{vib}})

The reader is referred to [Kee_2003]_ for the exact expression of each one of the above arguments. The interpolating function with an the exponent is :math:`n=1` is given by

.. math::

    \lambda_k(T) = \sqrt{T} (A + B \, log(T) + C \, log(T)^2 + D \, log(T)^3 + E \, log(T)^4)^n

Using a mixture averaged rule based on its composition in terms of mole fractions is given by

.. math::

    \lambda^{(m)} = \frac{1}{2} \left( \sum_{k=1}^{K} X_k \lambda_k +
       \frac{1}{\sum_{k=1}^{K} \frac{X_k}{\lambda_k} }\right)

.. _subsec:Species mass diffusivities:

The species mass diffusivities in :math:`\frac{m^2}{s}` are evaluated according to 

.. math::

    D_{ij} = 1.8583 \times 10^{-7} \frac{[T^3 W_ij]^{\frac{1}{2}}}{P \sigma_{ij}^2 \Omega^{(1,1)}_{ij}(T, \epsilon, k_B, \delta_i)}

In this equation, :math:`P` is the pressure and :math:`\Omega^{(1,1)}` is another collision integral. Similarly to the viscosity and thermal conductivity, a interpolating function is used:

.. math::

     D_{ij}(T) = \frac{T^{3/2}}{P} (A + B \, log(T) + C \, log(T)^2 + D \, log(T)^3 + E \, log(T)^4)^n

Here, the exponent is :math:`n=1`. 

Each species has a respective mass diffusivity relative to the mixture, which is given by a weighting rule considering the species binary mass diffusivities
:math:`D_{ij}` and the mass fractions :math:`Y_i`

.. math::

    D_{i}^{(m)} = \frac{1 - Y_i}{\sum_{j\ne i} \frac{X_j}{D_{ij}}}

This mixture rule becomes singular in regions of a single species, when :math:`1 - Y_i \to 0` and :math:`\sum_{j\ne i} X_j \to 0`. In this case, the species self-diffusivity :math:`D_{ii}` is used instead as the limit value.
