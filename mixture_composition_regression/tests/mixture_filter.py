from mixture_composition_regression.tests.import_training_set import import_training_set


def main():
    wdn, wd, wn = import_training_set()
    # print(wn.da.coords['name'])
    criteria = {'nacl': [0, 0.10], 'water': [0, 0.1]}
    # for name, bds in criteria.items():
    #     print(name)
    #     print(bds)
    wn2 = wdn.filter(criteria)
    print(wn2.da.coords['name'])

    return


if __name__ == '__main__':
    main()
