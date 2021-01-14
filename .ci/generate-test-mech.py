import cantera as ct
import pyrometheus as pyro
import glob

test_mechs = glob.glob("test/*.cti")
for mech in test_mechs:
    mechname = mech[5:-4]
    with open(f"test/{mechname}_mech.py", "w") as outf:
        code = pyro.gen_thermochem_code(ct.Solution(f"{mech}", "gas"))
        print(code, file=outf)
