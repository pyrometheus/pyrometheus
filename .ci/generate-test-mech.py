import cantera as ct
import pyrometheus as pyro
with open("pyrometheus/thermochem_example.py", "w") as outf:
    outf.write(pyro.gen_thermochem_code(ct.Solution("test/sanDiego.cti", "gas")))
