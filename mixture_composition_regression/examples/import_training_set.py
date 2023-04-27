from mixture_composition_regression.import_spectrum import clean_data
from mixture_composition_regression.sample import Sample
from mixture_composition_regression.mixture import Mixture


def get_005_w(m_sol, m_dipa):
    w = 0.6539 / (0.6539 + 65.0868)

    mnacl = w * m_sol
    mwater = (1 - w) * m_sol
    mtot = mnacl + mwater + m_dipa
    return [mwater / mtot, m_dipa / mtot, mnacl / mtot]


def get_003_w(m_sol, m_dipa):
    w = 0.6539 / (0.6539 + 65.0868)
    w = w * 5.9860 / (5.9860 + 4.1022)
    mnacl = w * m_sol
    mwater = (1 - w) * m_sol
    mtot = mnacl + mwater + m_dipa
    return [mwater / mtot, m_dipa / mtot, mnacl / mtot]


def get_2023_04_06(m_sol, m_water):
    w = 0.6539 / (0.6539 + 65.0868)
    m_nacl = w * m_sol
    m_dipa = 0
    m_water = m_water + (1 - w) * m_sol
    m_tot = m_nacl + m_dipa + m_water

    w_water = m_water / m_tot
    w_dipa = m_dipa / m_tot
    w_nacl = m_nacl / m_tot
    return [w_water, w_dipa, w_nacl]


def import_training_set():
    cp = {'name': ['water', 'dipa', 'nacl'],
          'mw': [18.015, 101.19, 58.44],
          'nu': [1, 1, 2]}

    # 03-03-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-03-03/2023-03-03.csv'
    df = clean_data(file)

    back = Sample('background', df, 0, 1, chem_properties=cp, w=[1, 0, 0])
    water1 = Sample('water1', df, 2, 3, chem_properties=cp, w=[1., 0., 0.], background=back)
    dipa1 = Sample('dipa1', df, 4, 5, chem_properties=cp, w=[0., 1., 0.], background=back)

    # 03-07-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-03-07/2023-03-07.csv'
    df = clean_data(file)

    back = Sample('background', df, 0, 1, chem_properties=cp, w=[1, 0, 0])
    water2 = Sample('water2', df, 2, 3, chem_properties=cp, w=[1., 0., 0.], background=back)
    dipa2 = Sample('dipa2', df, 4, 5, chem_properties=cp, w=[0., 1., 0.], background=back)
    dipa_w1 = Sample('dipa_w1', df, 6, 7, chem_properties=cp,
                     w=[0.0910 / (0.0910 + 0.9474), 0.9474 / (0.0910 + 0.9474), 0.], background=back)
    dipa_w2 = Sample('dipa_w2', df, 8, 9, chem_properties=cp,
                     w=[0.1510 / (0.1510 + 1.0358), 1.0358 / (0.1510 + 1.0358), 0.], background=back)

    # 03-09-2023
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-03-09/2023-03-09.csv'
    df = clean_data(file)

    # no background taken on this date.
    dipa_w1a = Sample('dipa_w1a', df, 0, 1, chem_properties=cp,
                      w=[0.0910 / (0.0910 + 0.9474), 0.9474 / (0.0910 + 0.9474), 0.], background=back)
    dipa_w2a = Sample('dipa_w2a', df, 2, 3, chem_properties=cp,
                      w=[0.1510 / (0.1510 + 1.0358), 1.0358 / (0.1510 + 1.0358), 0.], background=back)
    dipa_w3 = Sample('dipa_w3', df, 4, 5, chem_properties=cp,
                     w=[0.0382 / (0.0382 + 0.8671), 0.8671 / (0.0382 + 0.8671), 0.], background=back)
    dipa_w4 = Sample('dipa_w4', df, 6, 7, chem_properties=cp,
                     w=[0.3690 / (0.3690 + 1.1550), 1.1550 / (0.3690 + 1.1550), 0.], background=back)

    # # 03-22-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-03-22/2023-03-22.csv'
    df = clean_data(file)

    # no background taken on this date.
    water3 = Sample('water3', df, 2, 3, chem_properties=cp, w=[1., 0., 0.], background=back)
    five_M = Sample('5M', df, 4, 5, chem_properties=cp, w=[1. - 0.2470, 0., 0.2470], background=back)
    five_M_2 = Sample('5M_2', df, 6, 7, chem_properties=cp, w=[1. - 0.2470, 0., 0.2470], background=back)
    two_M = Sample('2M', df, 8, 9, chem_properties=cp, w=[1. - 0.1087, 0., 0.1087], background=back)
    two_M_2 = Sample('2M_2', df, 10, 11, chem_properties=cp, w=[1. - 0.1087, 0., 0.1087], background=back)
    four_M = Sample('4M', df, 12, 13, chem_properties=cp, w=[1. - 0.2036, 0., 0.2036], background=back)
    four_M_2 = Sample('4M_2', df, 14, 15, chem_properties=cp, w=[1. - 0.2036, 0., 0.2036], background=back)

    # 03-30-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-03-30/2023-03-30.csv'
    df = clean_data(file)

    back = Sample('background', df, 0, 1, chem_properties=cp, w=[1, 0, 0])

    # these samples are mixtures of dipa and 0.5 wt % nacl
    nacl_005_1a = Sample('dipa_nacl_005_1a', df, 4, 5, chem_properties=cp, w=get_005_w(6.7168, 0.1519), background=back)
    nacl_005_1b = Sample('dipa_nacl_005_1b', df, 6, 7, chem_properties=cp, w=get_005_w(6.7168, 0.1519), background=back)
    nacl_005_6a = Sample('dipa_nacl_005_6a', df, 8, 9, chem_properties=cp, w=get_005_w(0.3404, 6.3267), background=back)
    nacl_005_6b = Sample('dipa_nacl_005_6b', df, 10, 11, chem_properties=cp, w=get_005_w(0.3404, 6.3267),
                         background=back)
    nacl_005_2a = Sample('dipa_nacl_005_2a', df, 12, 13, chem_properties=cp, w=get_005_w(4.6713, 1.6821),
                         background=back)
    nacl_005_2b = Sample('dipa_nacl_005_2b', df, 14, 15, chem_properties=cp, w=get_005_w(4.6713, 1.6821),
                         background=back)
    nacl_005_5a = Sample('dipa_nacl_005_5a', df, 16, 17, chem_properties=cp, w=get_005_w(0.4695, 5.5782),
                         background=back)
    nacl_005_5b = Sample('dipa_nacl_005_5b', df, 18, 19, chem_properties=cp, w=get_005_w(0.4695, 5.5782),
                         background=back)
    nacl_005_4a = Sample('dipa_nacl_005_4a', df, 20, 21, chem_properties=cp, w=get_005_w(1.1842, 3.6719),
                         background=back)
    nacl_005_4b = Sample('dipa_nacl_005_4b', df, 22, 23, chem_properties=cp, w=get_005_w(1.1842, 3.6719),
                         background=back)
    nacl_005_3a = Sample('dipa_nacl_005_3a', df, 24, 25, chem_properties=cp, w=get_005_w(2.1407, 2.9448),
                         background=back)

    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-04-04/2023-04-04.csv'
    df = clean_data(file)
    back = Sample('background', df, 0, 1, chem_properties=cp, w=[1, 0, 0])

    s7 = Sample('water_dipa_nacl_s7', df, 10, 11, chem_properties=cp, w=get_003_w(0.4246, 0.9616), background=back)
    s8 = Sample('water_dipa_nacl_s8', df, 8, 9, chem_properties=cp, w=get_003_w(0.7291, 0.7668), background=back)
    s9 = Sample('water_dipa_nacl_s9', df, 6, 7, chem_properties=cp, w=get_003_w(0.3236, 0.8704), background=back)
    s10 = Sample('water_dipa_nacl_s10', df, 4, 5, chem_properties=cp, w=get_003_w(0.2493, 0.7336), background=back)
    s11 = Sample('water_dipa_nacl_s11', df, 2, 3, chem_properties=cp, w=get_003_w(0.1462, 0.8066), background=back)
    s12 = Sample('water_dipa_nacl_s12', df, 12, 13, chem_properties=cp, w=get_003_w(0.0611, 2.3315), background=back)

    # 2023-04-06
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-04-06/2023-04-06.csv'
    df = clean_data(file)
    back = Sample('background', df, 0, 1, chem_properties=cp, w=[1, 0, 0])
    a2 = Sample('a2', df, 2, 3, chem_properties=cp, w=get_2023_04_06(3.8376, 2.7831), background=back)
    irina = Sample('irina', df, 4, 5, chem_properties=cp, w=get_2023_04_06(2.7387, 10.5606), background=back)
    a3 = Sample('a3', df, 8, 9, chem_properties=cp, w=get_2023_04_06(3.3168, 1.9031), background=back)
    a4 = Sample('a4', df, 6, 7, chem_properties=cp, w=get_2023_04_06(1., 0.), background=back)

    # 2023-04-10
    file = '/Users/ianbillinge/Documents/yiplab/projects/ir/data/1mm_pl/2023-04-10/2023-04-10.csv'
    df = clean_data(file)
    back = Sample('background', df, 0, 1, chem_properties=cp, w=[1, 0, 0])
    s11a = Sample('s11a', df, 4, 5, chem_properties=cp, w=[6.9871 / (6.9871 + 0.0068), 0.0068 / (6.9871 + 0.0068), 0], background=back)
    s12a = Sample('s12a', df, 2, 3, chem_properties=cp, w=[7.2492 / (7.2492 + 0.0179), 0.0179 / (7.2492 + 0.0179), 0], background=back)
    s13 = Sample('s13', df, 6, 7, chem_properties=cp, w=[6.1994 / (6.1994 + 0.0211), 0.0211 / (6.1994 + 0.0211), 0], background=back)
    s14 = Sample('s14', df, 8, 9, chem_properties=cp, w=[5.9043 / (5.9043 + 0.0114), 0.0114 / (5.9043 + 0.0114), 0], background=back)
    s15 = Sample('s15', df, 10, 11, chem_properties=cp, w=[5.0079 / (5.0079 + 0.0098), 0.0098 / (5.0079 + 0.0098), 0], background=back)
    s16 = Sample('s16', df, 12, 13, chem_properties=cp, w=[(5.4557 + 17.3049) / (5.4557 + 17.3049 + 0.0337), 0.0337 / (5.4557 + 17.3049 + 0.0337), 0], background=back)


    water_dipa = Mixture([water1, dipa1, water2,
                          water3,
                          dipa2, dipa_w1, dipa_w1a, dipa_w2, dipa_w2a, dipa_w3, dipa_w4,
                          s11a, s12a, s13, s14, s15, s16
                        ],
                         name='water_dipa')
    water_dipa.savefile('water_dipa.nc', mode='w')

    water_nacl = Mixture([water1, water2,
                          water3, five_M, five_M_2, two_M, two_M_2, four_M, four_M_2,
                          a2, irina, a3, a4
                          ],
                         name='water_nacl')
    water_nacl.savefile('water_nacl.nc', mode='w')

    all_3 = Mixture([nacl_005_1a, nacl_005_1b, nacl_005_2a,
                                                         nacl_005_2b,
                                                         nacl_005_3a, nacl_005_4a, nacl_005_4b, nacl_005_5a,
                                                         nacl_005_5b, nacl_005_6a, nacl_005_6b,
                                                         s7, s8, s9, s10, s11, s12],
                                                        name = 'all_three')
    all_3.savefile('all_3.nc', mode='w')

    water_dipa_nacl = water_dipa + water_nacl + all_3
    water_dipa_nacl = water_dipa_nacl.set_name('water_dipa_nacl')
    water_dipa_nacl.savefile('water_dipa_nacl.nc', mode='w')

    return water_dipa_nacl, water_dipa, water_nacl
