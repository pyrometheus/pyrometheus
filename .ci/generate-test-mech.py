import cantera as ct
import pyrometheus as pyro
import glob

for mech in glob.glob("test/mechs/*.cti"):
    mechname = mech[11:-4]
    with open(f"test/mechs/{mechname}_mech.py", "w") as outf:
        print(pyro.gen_thermochem_code(ct.Solution(f"{mech}", "gas")), file=outf)
