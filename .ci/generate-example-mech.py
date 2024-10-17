import sys
import pyrometheus.codegen.python as pyro


narg = len(sys.argv)
if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} cti_file_name mech_file_name")
else:
    cti_file_name = sys.argv[1]
    mech_file_name = sys.argv[2]
    pyro.cti_to_mech_file(cti_file_name, mech_file_name)
