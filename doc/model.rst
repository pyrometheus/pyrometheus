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
     R_{j} = k_{j}(T)\psq{ \prod_{\ell = 1}^{N}\pp{\frac{ \rho y_{\ell} }{ W_{\ell} }}^{\nu_{mj}^{\prime}} - \frac{1}{K_{j}(T)}\prod_{k = 1}^{N}\pp{\frac{ \rho y_{k} }{ W_{k} }}^{\nu_{mj}^{\prime\prime}} },\qquad j = 1,\dots,M,

where :math:`k_{j}(T)` is the rate coefficient of the
:math:`j^{\mathrm{th}}` reaction and :math:`K_{j}(T)` its equilibrium
constant. Depending on the reaction, the rate coefficient
:math:`k_{j}(T)` may take different forms (and even become a function of
pressure). Its simplest form is the Arrhenius expression,

.. math::

   \label{eq:rate_coeff}
     k_{j}(T) = A_{j}T^{b_{j}}\exp\pp{ -\frac{\theta_{a,j}}{T} },\qquad j = 1,\dots,M

where :math:`A_{j}` is the pre-exponential, :math:`b_{j}` is the
temperature exponent, and :math:`\theta_{a,j}` is the activation
temperature.

The equilibrium constant is evaluated through equilibrium thermodynamics

.. math::

   \label{eq:equil_constants}
     K_{j}(T) = \pp{ \frac{p_{0}}{RT} }^{\sum_{i = 0}^{\nu_{ij}}}\exp\pp{ -\sum_{i = 1}^{N}\frac{\nu_{ij}g_{i}(T)}{RT} },\qquad j = 1,\dots,M,

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
