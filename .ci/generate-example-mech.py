import sys
import pyrometheus as pyro


if __name__ == "__main__":
    narg = len(sys.argv)
    if narg < 3:
        print(f"Usage: {sys.argv[0]} cti_file_name mech_file_name")
    cti_file_name = sys.argv[1]
    mech_file_name = sys.argv[2]
    pyro.cti_to_mech_file(cti_file_name, mech_file_name)
