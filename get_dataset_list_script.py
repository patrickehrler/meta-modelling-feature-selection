import openml

def get_datasets(number_of_classes):
    dataset_overview = openml.datasets.list_datasets(status='active', output_format='dataframe')
    dataset_overview = dataset_overview[
                (dataset_overview['format'] != 'Sparse_ARFF') &  # would break when calling get_data()
                (dataset_overview['NumberOfClasses'] == number_of_classes) &
                (dataset_overview['NumberOfNumericFeatures'] >= 1500) &  # includes target variable
                (dataset_overview['NumberOfNumericFeatures'] <= 13000) &
                (dataset_overview['NumberOfInstances'] >= 500) &
                (dataset_overview['NumberOfInstances'] <= 10000) &
                (dataset_overview['NumberOfMissingValues'] == 0)
            ]
    return dataset_overview

classification_binary_datasets = get_datasets(2)
#regression_datasets = get_datasets(0)

print(classification_binary_datasets)
classification_binary_datasets.to_csv("datasets/classification_binary_datasets.csv", index=False)
#regression_datasets.to_csv("datasets/regression_datasets.csv", index=False)
