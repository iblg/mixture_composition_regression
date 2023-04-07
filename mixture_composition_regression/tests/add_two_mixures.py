from mixture_composition_regression.import_spectrum import clean_data
from mixture_composition_regression.sample import Sample
from mixture_composition_regression.mixture import Mixture


def get_005_w(m_sol, m_dipa):
    w_005 = 0.6539 / (0.6539 + 65.0868)

    mnacl = w_005 * m_sol
    mwater = (1 - w_005) * m_sol
    mtot = mnacl + mwater + m_dipa
    return [mwater / mtot, m_dipa / mtot, mnacl / mtot]


def get_003_w(m_sol, m_dipa):
    w = 0.6539 / (0.6539 + 65.0868) * 5.9860 / (5.9860 + 4.1022)
    mnacl = w * m_sol
    mwater = (1 - w) * m_sol
    mtot = mnacl + mwater + m_dipa
    return [mwater / mtot, m_dipa / mtot, mnacl / mtot]


def import_training_set():
    cp = {'name': ['water', 'dipa', 'nacl'],
          'mw': [18.015, 101.19, 58.44],
          'nu': [1, 1, 2]}

    # 03-03-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-03-03/2023-03-03.csv'
    df = clean_data(file)
    water1 = Sample('water1', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])
    dipa1 = Sample('dipa1', df, 4, 5, chem_properties=cp, w=[0., 1., 0.])

    # 03-07-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-03-07/2023-03-07.csv'
    df = clean_data(file)
    water2 = Sample('water2', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])
    dipa2 = Sample('dipa2', df, 4, 5, chem_properties=cp, w=[0., 1., 0.])
    dipa_w1 = Sample('dipa_w1', df, 6, 7, chem_properties=cp,
                     w=[0.0910 / (0.0910 + 0.9474), 0.9474 / (0.0910 + 0.9474), 0.])
    dipa_w2 = Sample('dipa_w2', df, 8, 9, chem_properties=cp,
                     w=[0.1510 / (0.1510 + 1.0358), 1.0358 / (0.1510 + 1.0358), 0.])

    # 03-09-2023
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-03-09/2023-03-09.csv'
    df = clean_data(file)
    dipa_w1a = Sample('dipa_w1a', df, 0, 1, chem_properties=cp,
                      w=[0.0910 / (0.0910 + 0.9474), 0.9474 / (0.0910 + 0.9474), 0.])
    dipa_w2a = Sample('dipa_w2a', df, 2, 3, chem_properties=cp,
                      w=[0.1510 / (0.1510 + 1.0358), 1.0358 / (0.1510 + 1.0358), 0.])
    dipa_w3 = Sample('dipa_w3', df, 4, 5, chem_properties=cp,
                     w=[0.0382 / (0.0382 + 0.8671), 0.8671 / (0.0382 + 0.8671), 0.])
    dipa_w4 = Sample('dipa_w4', df, 6, 7, chem_properties=cp,
                     w=[0.3690 / (0.3690 + 1.1550), 1.1550 / (0.3690 + 1.1550), 0.])

    # 03-22-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-03-22/2023-03-22.csv'
    df = clean_data(file)

    water3 = Sample('water3', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])
    five_M = Sample('5M', df, 4, 5, chem_properties=cp, w=[1. - 0.2470, 0., 0.2470])
    five_M_2 = Sample('5M_2', df, 6, 7, chem_properties=cp, w=[1. - 0.2470, 0., 0.2470])
    two_M = Sample('2M', df, 8, 9, chem_properties=cp, w=[1. - 0.1087, 0., 0.1087])
    two_M_2 = Sample('2M_2', df, 10, 11, chem_properties=cp, w=[1. - 0.1087, 0., 0.1087])
    four_M = Sample('4M', df, 12, 13, chem_properties=cp, w=[1. - 0.2036, 0., 0.2036])
    four_M_2 = Sample('4M_2', df, 14, 15, chem_properties=cp, w=[1. - 0.2036, 0., 0.2036])

    # 03-30-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-03-30/2023-03-30.csv'
    df = clean_data(file)
    # these samples are mixtures of dipa and 0.5 wt % nacl
    nacl_005_1a = Sample('dipa_nacl_005_1a', df, 4, 5, chem_properties=cp, w=get_005_w(6.7168, 0.1519), background=None)
    nacl_005_1b = Sample('dipa_nacl_005_1b', df, 6, 7, chem_properties=cp, w=get_005_w(6.7168, 0.1519), background=None)
    nacl_005_6a = Sample('dipa_nacl_005_6a', df, 8, 9, chem_properties=cp, w=get_005_w(0.3404, 6.3267), background=None)
    nacl_005_6b = Sample('dipa_nacl_005_6b', df, 10, 11, chem_properties=cp, w=get_005_w(0.3404, 6.3267),
                         background=None)
    nacl_005_2a = Sample('dipa_nacl_005_2a', df, 12, 13, chem_properties=cp, w=get_005_w(4.6713, 1.6821),
                         background=None)
    nacl_005_2b = Sample('dipa_nacl_005_2b', df, 14, 15, chem_properties=cp, w=get_005_w(4.6713, 1.6821),
                         background=None)
    nacl_005_5a = Sample('dipa_nacl_005_5a', df, 16, 17, chem_properties=cp, w=get_005_w(0.4695, 5.5782),
                         background=None)
    nacl_005_5b = Sample('dipa_nacl_005_5b', df, 18, 19, chem_properties=cp, w=get_005_w(0.4695, 5.5782),
                         background=None)
    nacl_005_4a = Sample('dipa_nacl_005_4a', df, 20, 21, chem_properties=cp, w=get_005_w(1.1842, 3.6719),
                         background=None)
    nacl_005_4b = Sample('dipa_nacl_005_4b', df, 22, 23, chem_properties=cp, w=get_005_w(1.1842, 3.6719),
                         background=None)
    nacl_005_3a = Sample('dipa_nacl_005_3a', df, 24, 25, chem_properties=cp, w=get_005_w(2.1407, 2.9448),
                         background=None)

    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-04-04/2023-04-04.csv'
    df = clean_data(file)
    s7 = Sample('water_dipa_nacl_s7', df, 10, 11, chem_properties=cp, w=get_003_w(0.4246, 0.9616), background=None)
    s8 = Sample('water_dipa_nacl_s8', df, 8, 9, chem_properties=cp, w=get_003_w(0.7291, 0.7668), background=None)
    s9 = Sample('water_dipa_nacl_s9', df, 6, 7, chem_properties=cp, w=get_003_w(0.3236, 0.8704), background=None)
    s10 = Sample('water_dipa_nacl_s10', df, 4, 5, chem_properties=cp, w=get_003_w(0.2493, 0.7336), background=None)
    s11 = Sample('water_dipa_nacl_s11', df, 2, 3, chem_properties=cp, w=get_003_w(0.1462, 0.8066), background=None)
    s12 = Sample('water_dipa_nacl_s12', df, 12, 13, chem_properties=cp, w=get_003_w(0.0611, 2.3315), background=None)


    dipa_water = Mixture([], savefile='dipa_water.nc')

    m1 = Mixture([water1,
                  # dipa1, water2, water3, dipa2, dipa_w1, dipa_w1a, dipa_w2, dipa_w2a, dipa_w3, dipa_w4,
                  # five_M, five_M_2, two_M, two_M_2, four_M, four_M_2, nacl_005_1a, nacl_005_1b, nacl_005_2a,
                  # nacl_005_2b
                  ])
    m2 = Mixture([dipa1
                  # nacl_005_3a, nacl_005_4a, nacl_005_4b, nacl_005_5a, nacl_005_5b, nacl_005_6a, nacl_005_6b,
                  # s7, s8, s9, s10, s11, s12
                  ])

    return m1, m2

def main():
    m1, m2 = import_training_set()
    m3 = m1 + m2
    print(m3.da)
    return

if __name__ == '__main__':
    main()