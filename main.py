# Made by Kyrylo Krocha
import matplotlib.pyplot as plt
import csv
import random

# Define the list of words
words = ["transfer", "greetings", "diplomat", "winner", "name", "credit", "card", "house", "market", "chat"]
num_rows = 20   # Number of rows to generate full dataset
words_in_row = 4
csv_filename = "generated_data.csv"   # CSV file name
train_size = 0.3  # size of train sample
random.seed(10)  # setting seed

# Generate data
data = []
for _ in range(num_rows):
    row_words = random.sample(words, words_in_row)  # Select 4 random words
    label = "not spam" if "card" in row_words else "spam"
    data.append(row_words + [label])


# Save to CSV file
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)


lexicon = []
for j in range(len(data)):
    lexicon.extend(data[j])
lexicon = list(set(lexicon))


# training
def training(spam_data):
    spam = 0.0
    not_spam = 0.0
    total_spam = 0.0
    total_not_spam = 0.0

    appearance_count_spam = {lexicon[i]: 0 for i in range(0, len(lexicon))}
    appearance_count_not_spam = {lexicon[i]: 0 for i in range(0, len(lexicon))}
    for i in range(len(spam_data)):
        row = spam_data[i]
        if row[4] == "not spam":
            not_spam += 1
            for k in range(0, len(row) - 1):
                appearance_count_not_spam[row[k]] += 1
        else:
            spam += 1
            for k in range(0, len(row) - 1):
                appearance_count_spam[row[k]] += 1

    for word in appearance_count_spam:
        total_spam += appearance_count_spam[word]
    for word in appearance_count_not_spam:
        total_not_spam += appearance_count_not_spam[word]

    p_spam = spam / num_rows
    p_not_spam = not_spam / num_rows

    p_spam_words = {lexicon[i]: 0 for i in range(0, len(lexicon))}
    p_not_spam_words = {lexicon[i]: 0 for i in range(0, len(lexicon))}

    for word in appearance_count_spam:
        p_spam_words[word] = (appearance_count_spam[word] + 1) / (total_spam + len(lexicon))
    for word in appearance_count_not_spam:
        p_not_spam_words[word] = (appearance_count_not_spam[word] + 1) / (total_not_spam + len(lexicon))
    return p_spam, p_spam_words, p_not_spam, p_not_spam_words


def classification(p_spam, p_spam_words, p_not_spam, p_not_spam_words, spam_data):
    spam_or_not = []
    for row in spam_data:
        p_s = p_spam  # probability of spam
        p = p_not_spam   # probability of not spam
        for word in row[0:len(row) - 1]:
            p_s *= p_spam_words[word]
            p *= p_not_spam_words[word]
        if p_s >= p:
            spam_or_not.append("spam")
        else:
            spam_or_not.append("not spam")
    return spam_or_not


def roc_curve(spam_data):
    tp, tn, fp, fn = 0, 0, 0, 0
    roc_x, roc_y = [], []
    for row in range(len(spam_data)):
        if spam_data[row][len(spam_data[row])-1] == spam_data[row][len(spam_data[row])-2]:
            if spam_data[row][len(spam_data[row])-1] == "spam":
                tp += 1
            else:
                tn += 1
        else:
            if spam_data[row][len(spam_data[row])-1] == "spam":
                fp += 1
            else:
                fn += 1
        roc_x.append(fp)
        roc_y.append(tp)

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fnr = fp / (tn + fp)
    x_path = [x / fp if fp != 0 else 0 for x in roc_x]
    y_path = [y / tp if tp != 0 else 0 for y in roc_y]
    if x_path[0] != 0:
        x_path.insert(0, 0)
        y_path.insert(0, 0)
    if x_path[len(x_path)-1] != 1:
        x_path.append(1)
        y_path.append(1)
    plt.style.use('bmh')
    plt.plot(x_path, y_path, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC curve')
    plt.xlabel('1-TNR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()
    print("TPR is: " + str(tpr))
    print("TNR is: " + str(tnr))
    print("FNR is: " + str(fnr))
    return tpr, tnr, fnr


def main():
    train_data = random.sample(data, int(len(data)*train_size))
    p_spam, p_spam_words, p_not_spam, p_not_spam_words = training(train_data)
    test_data = [item for item in data if item not in train_data]
    spam_or_not = classification(p_spam, p_spam_words, p_not_spam, p_not_spam_words, test_data)
    for row in range(len(test_data)):
        test_data[row].append(spam_or_not[row])
        print(test_data[row])
    roc_curve(test_data)


main()
