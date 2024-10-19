About Pyrometheus
=================

License
=======

.. include:: ../LICENSE

Citing Pyrometheus
==================

(TBD)

Acknowledgment
==============

This material is based in part upon work supported by the Department of Energy,
National Nuclear Security Administration, under Award Number DE-NA0003963.

Disclaimer
==========

This report was prepared as an account of work sponsored by an agency of the
United States Government.   Neither  the  United  States  Government  nor  any
agency  thereof,  nor  any  of  their employees, makes any warranty, express or
implied, or assumes any legal liability or responsibility for the accuracy,
completeness,  or usefulness of any information,  apparatus,  product,  or
process disclosed, or represents that its use would not infringe privately owned
rights.  Reference herein to any specific commercial product, process, or
service by trade name, trademark, manufacturer, or otherwise does not
necessarily constitute or imply its endorsement, recommendation, or favoring
by the United States Government or any agency thereof.  The views and opinions
of authors expressed herein  do  not  necessarily  state  or  reflect  those  of
the  United  States  Government  or any  agency thereof.

Building the Documentation
==========================

You need to install Sphinx (a Python documentation tool) and two third-party
packages to build the docs::

    pip install .[docs]

Then simply run::

    cd doc
    make html

and browse the docs starting from ``doc/_build/html/index.html``.

References
==========

..
    When adding references here, please use the demonstrated format:
    [FirstAuthor_pubyear]

    Example: (unindent to use)

    .. [Hesthaven_2008] Hesthaven and Warburton (2008), Nodal DG Methods, Springer \
        `DOI: <https://doi.org/10.1007/978-0-387-72067-8>`__
