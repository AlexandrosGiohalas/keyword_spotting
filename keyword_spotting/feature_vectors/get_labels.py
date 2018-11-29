ground_truth = open('words.txt','r')
ground_truth_list = list(ground_truth)
labels = open('labels.txt','w+')
for i in range(len(ground_truth_list)):
    if ground_truth_list[i].startswith('a02') or ground_truth_list[i].startswith('a03') or ground_truth_list[i].startswith('a04'):
        tokens = ground_truth_list[i].split(' ')
        labels.write(tokens[8])
    elif ground_truth_list[i].startswith('a05'):
        break


