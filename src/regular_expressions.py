import re
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

filtered_training_corpus = 'NLP_dataset/filtered_training_set'


def main():
    """
    Main function of the program
    """

    corpus_dir = 'NLP_dataset/training_set'  # Directory of corpus.
    new_corpus = PlaintextCorpusReader(corpus_dir, '.*')
    for file_id in new_corpus.fileids():
        file_to_read = open(corpus_dir+"/"+file_id, "r")

        # reading each file to get matched sentences
        matched_sen = match_regular_expressions(file_to_read)

        # writing the matched sentences to files
        write_to_files(matched_sen, file_id)

def match_regular_expressions(file_to_read):
    """
    Matching the sentences in the job description with regular expressions
    :param file_to_read: the file to read sentences from.
    :return: matched_sentences
    """

    matched_sentences = []
    for line in file_to_read:
        a = re.match("((.*)degree(.*))", line, re.I)
        b = re.match("((.*)languages(.*))", line, re.I)
        c = re.match("((.*)technologies(.*))", line, re.I)
        d = re.match("((.*)experience(.*))", line, re.I)
        e = re.match("((.*)understanding(.*))", line, re.I)
        f = re.match("((.*)knowledge(.*))", line, re.I)
        g = re.match("((.*)comfortable(.*))", line, re.I)
        h = re.match("((.*)proficiency(.*))", line, re.I)
        i = re.match("((.*)platforms(.*))", line, re.I)

        if a or b or c or d or e or f or g or h or i:
            matched_sentences.append(line)

    return matched_sentences


def write_to_files(matched_sen, file_id):
    """
    To write matched sentences to resultant file.
    :param matched_sen: list of matched sentences
    :param file_id: file to write to
    :return:
    """
    file_to_write = open(filtered_training_corpus+'/'+file_id, "w")
    for line in matched_sen:
        file_to_write.write(line)

# main function
main()

