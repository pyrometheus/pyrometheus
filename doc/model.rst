.. _sec:thermochemistry:

Thermochemistry and Transport
=============================

Pyrometheus generates code to evaluate chemical source terms.
These appear in the conservation equations of reacting flows.
Here, we lay out the corresponding equations.
We focus on a homogeneous adiabatic reactor for simplicify. Yet, the
systems explained here can easily be adapted to other configurations
(e.g., isochoric or inhomogeneous reactors).

.. _subsec:thermokinetics:

Chemical Kinetics and Thermodynamics
------------------------------------

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
(in :math:`\mathrm{J/kmol\cdot K}`), and :math:`T` the temperature (in
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

.. _subsec:transport:

Transport Properties
----------------------------------------------

Pyrometheus-generated code provides routines to evaluate species and mixture transport properties. These follow most closely the Cantera implementation, which is based on polynomial fits to collision integrals. This approach is based on the kinetic theory of gases, for which a complete overview can be found in chapter 12 of [Kee_2003]_.

.. _subsec:Viscosity:

The viscosity of the :math:`n^{\mathrm{th}}` species in the mixture is:

.. math::

    \mu_n = \sqrt{T} \left[\sum_{m = 0}^{4} a_{m, n}\, (\log\, T)^{m}\right]^2,

where the coefficients :math:`a_{m, n}` are provided by Cantera. The viscosity of the mixture is then obtained via the mixture rule

.. math::

    \mu = \sum_{n = 1}^{N} \frac{X_n \mu_n}{\sum_{j = 1}^{N} X_j\Phi_{nj}}

where :math:`X_{n} = W Y_{n} / W_{(n)}` is the mole fraction of species :math:`n`, and

.. math::

    \Phi_{nj} = \frac{1}{\sqrt{8}}
    \left( 1 + \frac{W_n}{W_j} \right)^{-\frac{1}{2}}
    \left( 1 + \left[ \frac{\mu_n}{\mu_j} \right]^{\frac{1}{2}}
    \left[ \frac{W_j}{W_n} \right]^{\frac{1}{4}} \right)^2.

.. _subsec:Thermal conductivity:

The thermal conductivity of species :math:`n` is

.. math::

    \lambda_n = \sqrt{T} \sum_{m = 0}^{0} b_{m, n}\, (\log\, T)^{m}.

The mixture viscosity is

.. math::

    \lambda = \frac{1}{2} \left( \sum_{n = 1}^{N} X_n \lambda_n +
       \frac{1}{\sum_{n = 1}^{N} \frac{X_n}{\lambda_n} } \right).

.. _subsec:Species mass diffusivities:

The binary mass diffusivities, in :math:`\frac{m^2}{s}`, for species $i$ and $j$ are

.. math::

     D_{i,j}(T) = \frac{T^{3/2}}{p} \sum_{m = 0}^{4}c_{i,j,m}\, (\log\, T)^m

The mixture-averaged diffusivity of species :math:`n` is

.. math::

    \mathscr{D}_{n} = \frac{W - X_{(n)}W_{n}}{W}\left\lbrace \sum_{m \neq n}\frac{X_{m}}{D_{nm}}  \right\rbrace^{-1}

This expression becomes singular for :math:`X_n = 1` (for any :math:`n`, so :math:`\sum_{m \neq n} X_m/D_{nm} = 0`). Thus, following Cantera, it only returns the mixture-averaged diffusivity if

.. math::

   \sum_{m \neq n} \frac{X_{m}}{D_nm} > 0,

and :math:`D_{nn}` otherwise. The conditional is implemented using :func:`numpy.where` and, of course, it is difficult to satisfy in finite-precision calculations. It can lead to round-off errors in :math:`\mathscr{D}_{n}` but, like Cantera, Pyrometheus does not attempt to correct this behavior to avoid the use of arbitrary thresholds.
