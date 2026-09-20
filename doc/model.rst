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

.. _subsec:surface:

Heterogeneous Kinetics
----------------------

Pyrometheus also generates code for reactions that take place on a surface rather
than in the bulk gas. These are described by a Cantera *interface* phase, which
couples the surface sites to the phases that react at them: a gas phase and,
for mechanisms such as char gasification, one or more bulk (solid) phases.

Throughout, :math:`N_{s}` is the number of surface species, :math:`N_{c}` the
number of species the interface couples in total, and :math:`M_{s}` the number of
heterogeneous reactions. Concentrations and production rates over all coupled
phases are ordered as Cantera orders them for kinetics: the interface's own
species first, then those of each adjacent phase in turn. This is *not* the
order in which the phases are written in the mechanism file, and it is the
order the generated routines expect.

.. _subsec:surface_state:

Surface State
~~~~~~~~~~~~~

The state of the surface is given by the site fractions, or coverages,
:math:`\boldsymbol{\theta} = \{ \theta_{k} \}_{k = 1}^{N_{s}}`, which satisfy
:math:`\sum_{k}\theta_{k} = 1`. A surface species may occupy more than one site;
writing :math:`\sigma_{k}` for the number it occupies and :math:`\Gamma_{0}` for
the site density of the interface (in :math:`\mathrm{kmol/m^{2}}`), its
concentration is

.. math::

   \label{eq:site_concentration}
     [\mathcal{S}_{k}] = \frac{ \theta_{k}\Gamma_{0} }{ \sigma_{k} },\qquad
     k = 1,\dots,N_{s}.

The factor :math:`\sigma_{k}` is easy to overlook, since the great majority of
mechanisms give every surface species a single site.

Because the species that meet at an interface live in phases of different
dimensionality, the quantity that enters a rate of progress is not the molar
concentration for all of them but the *activity concentration*: the molar
concentration for a gas species (in :math:`\mathrm{kmol/m^{3}}`), the site
concentration `[site_concentration] <#site_concentration>`__ for a surface
species (in :math:`\mathrm{kmol/m^{2}}`), and the activity for a bulk species,
which is unity for the pure solids these mechanisms react. Passing a bulk
species' molar density in its place, which is what Cantera's
:attr:`~cantera.ThermoPhase.concentrations` returns for that phase, is wrong by
that density, some two hundred :math:`\mathrm{kmol/m^{3}}` for graphite.

.. _subsec:surface_rates:

Rates of Progress
~~~~~~~~~~~~~~~~~

The rates of progress follow the law of mass action, as in the gas phase, but
over the activity concentrations :math:`C_{i}` of every species the interface
couples,

.. math::

   \label{eq:surface_reaction_rates}
     R_{j} = k_{j}(T)\prod_{\ell = 1}^{N_{c}}C_{\ell}^{\nu_{\ell j}^{\prime}}
     - \frac{ k_{j}(T) }{ K_{j}(T) }
     \prod_{k = 1}^{N_{c}}C_{k}^{\nu_{kj}^{\prime\prime}},\qquad
     j = 1,\dots,M_{s},

in :math:`\mathrm{kmol/m^{2}\textrm{-}s}`. Where a mechanism declares explicit
reaction orders, they replace the reactant stoichiometric coefficients
:math:`\nu_{\ell j}^{\prime}` in the forward product. Orders are a property of
the forward direction alone, so the reverse product always uses
:math:`\nu_{kj}^{\prime\prime}`. The production rates then follow exactly as in
the gas phase,

.. math::

   \dot{\omega}_{i} = \sum_{j = 1}^{M_{s}}\nu_{ij}R_{j},\qquad i = 1,\dots,N_{c},

and cover every phase the interface couples: gas species consumed or released at
the wall, surface species whose coverages evolve, and bulk species consumed as
the solid is eaten away.

.. _subsec:surface_rate_coeffs:

Rate Coefficients
~~~~~~~~~~~~~~~~~

Three forms of rate coefficient appear in surface mechanisms.

The first is the modified Arrhenius expression `[rate_coeff] <#rate_coeff>`__,
unchanged from the gas phase.

The second is the *sticking coefficient*, where the Arrhenius parameters give not
a rate coefficient but a dimensionless sticking probability,

.. math::

   \gamma_{j}(T) = A_{j}T^{b_{j}}\exp\left({ -\frac{\theta_{a,j}}{T} }\right),

the fraction of collisions with the surface that react. Converting it to a rate
coefficient is a result of kinetic theory: the flux of a gas species of molecular
weight :math:`W_{m}` onto a surface is :math:`\sqrt{RT/2\pi W_{m}}` per unit
concentration, and each of the :math:`n_{j}` sites the reaction consumes divides
by the site density, so

.. math::

   \label{eq:sticking}
     k_{j}(T) = \frac{ \gamma_{j}(T) }{ \Gamma_{0}^{n_{j}} }
     \sqrt{ \frac{RT}{2\pi W_{m}} }.

Reading :math:`A_{j}`, :math:`b_{j}` and :math:`\theta_{a,j}` as though they were
an ordinary Arrhenius rate is therefore wrong by orders of magnitude, and
nothing reports it. When the mechanism asks for the Motz-Wise correction,
which matters once :math:`\gamma_{j}` is no longer small compared to one, the
leading factor becomes :math:`\gamma_{j}/(1 - \gamma_{j}/2)`.

The third is coverage dependence, which multiplies either of the above by

.. math::

   \label{eq:coverage_dependence}
     \prod_{k}10^{\,a_{jk}\theta_{k}}\;\theta_{k}^{\,m_{jk}}\;
     \exp\left( -\frac{ E_{jk}\theta_{k} }{ RT } \right)

over the species :math:`k` the reaction declares a dependence on. It expresses
how the binding energy of an adsorbate changes as the surface fills up.

.. _subsec:surface_heat:

Surface Heat Release
~~~~~~~~~~~~~~~~~~~~

Heterogeneous reactions release or absorb heat at the wall, and a surface energy
balance needs that rate. It follows from the production rates and the species
enthalpies,

.. math::

   \label{eq:surface_heat_release}
     \dot{q} = -\sum_{i = 1}^{N_{c}}\dot{\omega}_{i}h_{i}(T)
     = -\sum_{j = 1}^{M_{s}}\Delta h_{j}(T)R_{j},\qquad
     \Delta h_{j} = \sum_{i = 1}^{N_{c}}\nu_{ij}h_{i}(T),

in :math:`\mathrm{W/m^{2}}`, since the production rates are per unit *area*,
and positive when the surface chemistry is exothermic. The two forms are
identical, the second following from the first by
`[production_rates] <#production_rates>`__; the first is what the generated code
evaluates, because it needs the stoichiometry only once, inside
:math:`\dot{\omega}`.

The sum runs over every species the interface couples, so a reaction that
consumes solid carries the enthalpy of the solid it consumes. Like the Gibbs
functions of `[surface_equil_constants] <#surface_equil_constants>`__, the gas-phase
enthalpies come from the separately generated gas-phase code and the surface and
bulk ones are generated alongside the surface mechanism.

.. _subsec:surface_equilibrium:

Equilibrium Constants
~~~~~~~~~~~~~~~~~~~~~

The equilibrium constant follows from equilibrium thermodynamics as in the gas
phase, but the factor that converts between the activity of a species and its
activity concentration, the standard concentration :math:`c^{0}_{i}`, is not
the same for every species, because the phases have different dimensionality:

.. math::

   \label{eq:surface_equil_constants}
     K_{j}(T) = \left( \prod_{i = 1}^{N_{c}}
     \left( c^{0}_{i} \right)^{\nu_{ij}} \right)
     \exp\left( -\sum_{i = 1}^{N_{c}}\frac{\nu_{ij}g_{i}(T)}{RT} \right),\qquad
     j = 1,\dots,M_{s},

with

.. math::

   c^{0}_{i} = \begin{cases}
     p_{0}/RT, & \text{$i$ a gas species,} \\
     \Gamma_{0}/\sigma_{i}, & \text{$i$ a surface species,} \\
     1, & \text{$i$ a bulk species.}
   \end{cases}

A reaction that changes the moles of gas thus carries
:math:`(p_{0}/RT)^{\Delta n_{g,j}}` as it does in the gas phase, one that changes
the number of occupied sites carries the corresponding power of the site density,
and a bulk species contributes nothing at all. Treating a bulk species as though
it were a gas species leaves a spurious factor of
:math:`(p_{0}/RT)^{\Delta n_{b,j}}`, roughly two orders of magnitude per mole
of solid at combustion temperatures.

The Gibbs functions :math:`g_{i}` are needed for every species the interface
couples. For the gas species they come from the separately generated gas-phase
code, which the generated surface code refers to; for the surface and
bulk species, which have no Pyrometheus code of their own, they are generated
alongside the surface mechanism.

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
