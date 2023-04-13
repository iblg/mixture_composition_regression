from mixture_composition_regression.tests.import_training_set import import_training_set


def main():
    wdn, wd, wn = import_training_set()
    # print(wn.da.coords['name'])
    wn2 = wdn.filter([0,[0,0]],[1, [0, 0.0]], [2, [0, 0.10]])
    print(wn2.da.coords['name'])

    return


if __name__ == '__main__':
    main()
