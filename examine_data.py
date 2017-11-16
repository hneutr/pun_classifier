from pun_data import HeterographicData, HomographicData

if __name__ == "__main__":
    hom_data = HomographicData()
    het_data = HeterographicData()

    neg_hom = [ x for i, x in enumerate(hom_data.x_set) if not hom_data.y_set[i]]
    neg_het = [ x for i, x in enumerate(het_data.x_set) if not het_data.y_set[i]]

    print(len(neg_hom))
    print(len(neg_het))

    count_dup = 0
    for x in neg_hom:
        duplicates = [i for i in neg_het if i == x]
        if len(duplicates):
            count_dup += 1

    print("duplicates: %d" % count_dup)
