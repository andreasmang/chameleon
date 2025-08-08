def add(a, b):
    """return sum of a and b"""
    return a + b

def test_add():
    """test add function"""
    assert add( 2, 3) == 5
    assert add(-1, 1) == 0
    assert add( 0, 0) == 0
    print("all tests passed")

if __name__ == "__main__":
    test_add()


###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
