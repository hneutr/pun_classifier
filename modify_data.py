import pickle, gzip
import argparse
from pun_data import Data

def add_nonpuns(to_add, non_puns):
    """
    to_add is a class of type Data.
    non_puns are an array of sentences
    """
    for x in non_puns:
        to_add.x_set.append(x)
        to_add.y_set.append(0)

    return to_add

def get_new_name(type):
    return "data/pickles/test-1-%s-even.pkl.gz" % type

def write_new_data(data, type):
    with open(get_new_name(type), 'wb') as f:
        pickle.dump([data.x_set, data.y_set], f)

def verify(new_set, old_set, neg_set):
    """
    given three sets:
        - old_set (unmodified)
        - neg_set (set the negatives were pulled from)
        - new_set (set with old_set and negatives from neg_set)

    verifies that
    - all items in old_set are in the new set with the correct labels
    - all items in new_set not in old set have correct labels
    """

    for i in range(len(new_set.x_set)):
        if i < len(old_set.x_set):
            assert(new_set.x_set[i] == old_set.x_set[i])
            assert(new_set.y_set[i] == old_set.y_set[i])
        else:
            assert(new_set.y_set[i] == 0)

            for j, x in enumerate(neg_set.x_set):
                if new_set.x_set[j] == neg_set.x_set[j]:
                    assert(neg_set.y_set[i] == 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='change data')
    parser.add_argument('--examine', action="store_true", help="check for duplicates")
    parser.add_argument('--write', default=True, action="store_false", help="write new data")
    parser.add_argument('--verify', action="store_true", help="verify the data is properly set")

    args = parser.parse_args()

    hom = Data("homographic")
    het = Data("heterographic")

    if args.examine:
        neg_hom = [ x for i, x in enumerate(hom.x_set) if not hom.y_set[i]]
        neg_het = [ x for i, x in enumerate(het.x_set) if not het.y_set[i]]

        count_dup = 0
        for x in neg_hom:
            duplicates = [i for i in neg_het if i == x]
            if len(duplicates):
                count_dup += 1

        print("duplicates: %d" % count_dup)

    if args.write:
        # must do this first
        neg_hom = [ x for i, x in enumerate(hom.x_set) if not hom.y_set[i]]
        neg_het = [ x for i, x in enumerate(het.x_set) if not het.y_set[i]]

        write_new_data(add_nonpuns(hom, neg_het), 'homographic')
        write_new_data(add_nonpuns(het, neg_hom), 'heterographic')

    if args.verify:
        hom_new = Data("homographic", True)
        het_new = Data("heterographic", True)

        verify(hom_new, hom, het)
        verify(het_new, het, hom)
