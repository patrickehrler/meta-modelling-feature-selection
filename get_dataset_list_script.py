import openml

def get_datasets(number_of_classes):
    dataset_overview = openml.datasets.list_datasets(status='active', output_format='dataframe')
    dataset_overview = dataset_overview[
                (dataset_overview['format'] != 'Sparse_ARFF') &  # would break when calling get_data()
                (dataset_overview['NumberOfClasses'] >= number_of_classes) &
                (dataset_overview['NumberOfNumericFeatures'] >= 102) &  # includes target variable
                (dataset_overview['NumberOfNumericFeatures'] <= 10000) &
                (dataset_overview['NumberOfInstances'] >= 200) &
                (dataset_overview['NumberOfInstances'] <= 5000) &
                (dataset_overview['NumberOfMissingValues'] == 0)
            ]
    return dataset_overview

classification_binary_datasets = get_datasets(2)

# print to csv
classification_binary_datasets.to_csv("results/datasets/classification_binary_datasets.csv", index=False)

