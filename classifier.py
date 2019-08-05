import math
import sys
import csv

train_file = sys.argv[2]
test_file = sys.argv[4]

# open the train file
with open(train_file) as f:
    content = f.readlines()
content = [x.strip() for x in content]

train_emails = list()
for i in content:
    train_emails.append(i.split(" ", 2))

# init all variables
prob_spam = 0.0
prob_ham = 0.0
spam_words = 0
ham_words = 0
spam_list = list()
ham_list = list()
spam_words_dict = dict()
ham_words_dict = dict()

# loop through all training mails and count the number of spam and non spam mails
for i in train_emails:
    if i[1] == 'spam':
        # increment count of spam mail
        spam_words += 1
        # append word to the spam list
        spam_list.append(i[2])
        # split all the email words on space
        i[2] = i[2].split(" ")
        # keep track of the word and its occurances
        spam_word, occurances = i[2][::2], i[2][1::2]
        # for each word in the spam list, append its count to the spam word dict
        for word in spam_word:
            idx = spam_word.index(word)
            if word in spam_words_dict:
                spam_words_dict[word] += int(occurances[idx])
            else:
                spam_words_dict[word] = int(occurances[idx])
    else:
        # increment count of spam mail
        ham_words += 1
        # append word to the ham list
        ham_list.append(i[2])
        # split all the email words on space
        i[2] = i[2].split(" ")
        # keep track of the word and its occurances
        ham_word, occurances = i[2][::2], i[2][1::2]
        # for each word in the spam list, append its count to the ham word dict
        for word in ham_word:
            idx = ham_word.index(word)
            if word in ham_words_dict:
                ham_words_dict[word] += int(occurances[idx])
            else:
                ham_words_dict[word] = int(occurances[idx])

# keep track of all unique words in the train set
temp_list1 = list(spam_words_dict)
temp_list2 = list(ham_words_dict)
temp_merged = temp_list1 + temp_list2
unique_elements = len(set(temp_merged))
#calculate the probability of spam and ham
prob_spam = float(spam_words)/float(len(train_emails))
prob_ham = float(ham_words)/float(len(train_emails))

# open the test file
with open(test_file) as f:
    content = f.readlines()
content = [x.strip() for x in content]

test_words = list()
for i in content:
    test_words.append(i.split(" ", 2))

# init the variables to keep track of all the correct predictions of spam and ham
correct = 0
correct_spam = 0
correct_ham = 0
csv_output = list()

# for each mail in the test set
for word in test_words:
    word1 = word[2].split(" ")
    w = word1[::2]
    test_spam_prob = math.log(prob_spam)
    # for each word in the mail
    for w1 in w:
        # if word previously encountered in the train dataset
        if w1 in spam_words_dict:
            # uses bayes net concept to compute the required probability
            test_spam_prob += math.log((spam_words_dict[w1] + 100) / float(sum(spam_words_dict.values()) + unique_elements * 100))
        else:
            # if word not encountered before in train
            test_spam_prob += math.log((100) / float(sum(spam_words_dict.values()) + unique_elements * 100))

    test_ham_prob = math.log(prob_ham)
    # for each word in the mail
    for w1 in w:
        # if word previously encountered in the train dataset
        if w1 in ham_words_dict:
            # uses bayes net concept to compute the required probability.
            test_ham_prob += math.log((ham_words_dict[w1]+ 100) / float(sum(ham_words_dict.values()) + unique_elements * 100))
        else:
            # if word not encountered before in train
            test_ham_prob += math.log((100) / float(sum(ham_words_dict.values()) + unique_elements * 100))

    # if ham probability is more than ham probability then it is a spam mail
    if test_spam_prob > test_ham_prob:
        correct_spam += 1
        pred = 'spam'

    else:
        correct_ham += 1
        pred = 'ham'

    # append so as to store in the output csv file
    csv_output.append((word[0], pred))

    # if prediction matches then increase correct value
    if pred == word[1]:
        correct += 1

# print "correct, spam, ham, total, accuracy: ", correct, correct_spam, correct_ham, len(test_words), float(correct)/float(len(test_words))

writer = csv.writer(open("out.csv", 'w'))
for row in csv_output:
    writer.writerow(row)
