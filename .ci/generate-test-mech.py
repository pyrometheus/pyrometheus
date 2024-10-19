import pyrometheus.codegen.python as pyro
import glob

test_mechs = glob.glob("test/mechs/*.yaml")
for cti_file_name in test_mechs:
    mechname = cti_file_name[11:-4]
    mech_file_name = f"test/mechs/{mechname}_mech.py"
    pyro.cti_to_mech_file(cti_file_name, mech_file_name)
