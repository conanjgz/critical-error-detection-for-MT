from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def print_stat(dataframe, true_label_column, prediction_column):
    print(f"Accuracy: {accuracy_score(dataframe['labels'], dataframe['predictions'])}")
    print(f"Precision: {precision_score(dataframe['labels'], dataframe['predictions'], pos_label='ERR')}")
    print(f"Recall: {recall_score(dataframe['labels'], dataframe['predictions'], pos_label='ERR')}")
    print(f"F1 score: {f1_score(dataframe['labels'], dataframe['predictions'], pos_label='ERR')}")
    CM = confusion_matrix(dataframe['labels'], dataframe['predictions'], labels=['ERR', 'NOT'])
    print("Confusion matrix:")
    print(f"         \tPred ERR\tPred NOT")
    print(f"True ERR:\t{CM[0][0]}\t\t{CM[0][1]}")
    print(f"True NOT:\t{CM[1][0]}\t\t{CM[1][1]}")

def format_submission(df, language_pair, method, index, path, index_type=None):

    if index_type is None:
        index = index

    elif index_type == "Auto":
        index = range(0, df.shape[0])

    predictions = df['predictions']
    with open(path, 'w') as f:
        for number, prediction in zip(index, predictions):
            text = language_pair + "\t" + method + "\t" + str(number) + "\t" + str(prediction)
            f.write("%s\n" % text)