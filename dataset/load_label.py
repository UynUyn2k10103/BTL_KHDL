def load_label():
    dic = {}

    with open("dataset/_UIT-VSFC/index2label.txt", mode='r', encoding='utf-8-sig') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '']
        for idx in range(0, len(lines), 2):
            index_label = lines[idx]
            label = lines[idx + 1]
            
            dic.update({int(index_label):label})

    return dic

if __name__ == '__main__':
    list_labels = load_label()
    print(list_labels)