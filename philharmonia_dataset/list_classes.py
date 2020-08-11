from labeler import datasets

if __name__ == "__main__":
    path_to_csv =  './data/philharmonia/all-samples/metadata.csv'
    dset = datasets.PhilharmoniaSet(path_to_csv)
    classes = dset.get_class_data()
    print(f'classname\t\tsamples')
    for c, n in classes:
        print(f'{c:<24}{n:<10}')