def space_separate(sent):
    # Separate phonemes in a given sentence
    phonemes = []

    i = 0
    while i < len(sent):
        if i < len(sent)-1:
            if sent[i+1] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
                if i < len(sent) - 2 and sent[i+2] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
                    phonemes.append(sent[i]+sent[i+1]+sent[i+2])
                    i += 3
                else:
                    phonemes.append(sent[i]+sent[i+1])
                    i += 2
                continue
            elif sent[i+1] in ['͡']:
                if i < len(sent) - 3 and sent[i+3] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
                    phonemes.append(sent[i]+sent[i+1]+sent[i+2]+sent[i+3])
                    i += 4
                else:
                    phonemes.append(sent[i]+sent[i+1]+sent[i+2])
                    i += 3
                continue
        phonemes.append(sent[i])
        if sent[i] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
            print('\t',sent)
        i += 1
    return phonemes

