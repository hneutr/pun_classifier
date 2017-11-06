import xml.etree.ElementTree as ET
import argparse
import pickle, gzip

def parse_id(raw):
    """
    breaks down a string into its component identifying parts

    note:
        input elements are 1 indexed; output will be 0 indexed

    returns array of (type, sentence_id, word_id)
    """
    split = raw.split("_")

    # sentence id
    split[1] = int(split[1]) - 1

    # word id
    if len(split) == 3:
        split[2] = int(split[2]) - 1
    
    return split

def decode_task_1_labels(content):
    """
    format:
        het_1\t0
        sentence_id\tpun/nonpun

    return:
        array of booleans
    """
    return [int(line.split("\t")[1]) for line in content]

def decode_task_2_labels(content):
    """
    format: 
        het_1\thet_1_2
        sentence_id\tword_position

    return:
        array of integer indexes
    """
    return [parse_id(line.split("\t")[1])[2] for line in content]


def decode_task_3_labels(content):
    """
    format:
        het_1_14\tallege%2:32:00::\tledge%1:17:00::
        word_id_sentence_id\tsense1\tsense2

    return:
        array of tuples (word_id, sense1, sense2)
    """
    decoded = []

    for line in content:
        id, first_sense, second_sense = line.split("\t")
        _, sentence_id, word_id = parse_id(id)

        decoded.append((word_id, first_sense, second_sense))

    return decoded

def decode_sentences(path):
    """
    given a file path to an xml file, returns an array of sentences
    """
    xml_tree = ET.parse(path)
    xml_root = xml_tree.getroot()

    return [list(w.text for w in sent) for sent in xml_root]

def decode_labels(path, task):
    """
    given a file path to a tsv file of labels, returns an array of labels for that task
    """
    with open(path, 'r') as f:
        content = f.readlines()

    # strip newlines
    content = [ l.strip() for l in content]

    # retrive the labels by calling the appropriate decoder function
    return eval("decode_task_%s_labels" % task)(content)

def decode(test_or_trial, task, graphic):
    """
    writes pickled output for a given combination of task #, heterographic/homographic, and test/trial
    """
    base_name = "data/%s/subtask%s-%s-%s" % (test_or_trial, task, graphic, test_or_trial)

    x_file = "%s.xml" % base_name
    y_file = "%s.gold" % base_name

    sentences = decode_sentences(x_file)
    labels = decode_labels(y_file, task)

    out_path = "data/pickles/%s-%d-%s.pkl.gz" % (test_or_trial, task, graphic)
    print("outputting to: %s" % out_path)

    with open(out_path, 'wb') as f:
        pickle.dump([sentences, labels], f)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='xml pickler')
    parser.add_argument('--test_or_trial', type=str, default='test', help="load from test or trial data")
    parser.add_argument('--task', type=int, default=1, help="which subtask to convert")
    parser.add_argument('--all', action="store_true", help="generate all datasets")
    parser.add_argument('--graphic', type=str, default='homographic', help="which type of pun ['homographic', 'heterographic']. Default: homographic")

    args = parser.parse_args()

    if args.all:
        for test_or_trial in ['test', 'trial']:
            for graphic in ['heterographic', 'homographic']:
                for task in [1,2,3]:
                    decode(test_or_trial, task, graphic)
    else:
        decode(args.test_or_trial, args.task, args.graphic)

    # how to read:
    # with open(out_path, 'rb') as f:
    #     sentences, labels = pickle.load(f)
