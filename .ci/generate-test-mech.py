import cantera as ct
import pyrometheus as pyro
import glob

test_mechs = glob.glob("test/mechs/*.cti")
for mech in test_mechs:
    mechname = mech[11:-4]
    with open(f"test/mechs/{mechname}_mech.py", "w") as outf:
        print(pyro.gen_thermochem_code(ct.Solution(f"{mech}", "gas")), file=outf)
