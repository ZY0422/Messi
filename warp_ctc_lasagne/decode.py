# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:32:50 2015

@author: richi-ubuntu
"""
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

def state_decoder(state_seq, state_maps, output_phoneme):
    try:
        frobj_state = open(state_seq ,'r')
        frobj = open(state_maps, 'r')
        fwobj = open(output_phoneme, 'w')
    except IOError:
        print "failed to open ", state_seq, " for reading"
        print "failed to open ", state_maps, " for reading"
        print "failed to open ", output_phoneme, " for writing"
    else:
        info = frobj.readlines()
        start_end_list = []
        start_dict = OrderedDict()
        end_dict = OrderedDict()
        output_phoneme_dict = OrderedDict()
        for eachline in info:
            content = eachline.strip().split()
            l = []
            for item in content[1:]:
                l.append(int(item))
            start_dict[l[0]] = content[0]
            end_dict[l[-1]] = content[0]
            start_end_list.extend([l[0],l[-1]])
        state_info = frobj_state.readlines()
        state_dict = OrderedDict()
        for eachline in state_info:
            content = eachline.strip().split()
            l = []
            reduced = []
            for item in content[1:]:
                l.append(int(item))
            for index in xrange(len(l)):
                if index == 0:
                    reduced.append(l[index])
                elif l[index-1] != l[index]:
                    reduced.append(l[index])
            state_dict[content[0]] = reduced # get reduced state
        for key in state_dict.keys():
            hyp = state_dict[key]
            hyp_start_end = []
            hyp_symbol = []
            for item in hyp:
                if item in start_end_list:
                    hyp_start_end.append(item)
            index_pos = []
            for item in hyp_start_end:
                if start_dict.has_key(item):
                    hyp_symbol.append(start_dict[item])
                    index_pos.append(0)
                elif end_dict.has_key(item):
                    hyp_symbol.append(end_dict[item])
                    index_pos.append(1)
            real_output = []
            for index in xrange(len(index_pos)-1):
                if index_pos[index] == 0 and index_pos[index+1] == 1:
                    # indicate begin and end
                    if hyp_symbol[index] == hyp_symbol[index+1]:
                        real_output.append(hyp_symbol[index])
            output_phoneme_dict[key] = real_output
        for key in output_phoneme_dict.keys():
            fwobj.write(key)
            fwobj.write(' ')
            for item in output_phoneme_dict[key]:
                fwobj.write(item)
                fwobj.write(' ')
            fwobj.write('\n')

        frobj_state.close()
        frobj.close()
        fwobj.close()
        return output_phoneme_dict

def state_sil_decoder(state_seq, state_maps, output_phoneme, eps):
    try:
        frobj_state = open(state_seq ,'r')
        frobj = open(state_maps, 'r')
        fwobj = open(output_phoneme, 'w')
    except IOError:
        print "failed to open ", state_seq, " for reading"
        print "failed to open ", state_maps, " for reading"
        print "failed to open ", output_phoneme, " for writing"
    else:
        info = frobj.readlines()
        start_end_list = []
        start_dict = OrderedDict()
        end_dict = OrderedDict()
        output_phoneme_dict = OrderedDict()
        for eachline in info:
            content = eachline.strip().split()
            l = []
            for item in content[1:]:
                l.append(int(item))
            start_dict[l[0]] = content[0]
            end_dict[l[-1]] = content[0]
            start_end_list.extend([l[0],l[-1]])
        state_info = frobj_state.readlines()
        state_dict = OrderedDict()
        for eachline in state_info:
            content = eachline.strip().split()
            l = []
            reduced = []
            for item in content[1:]:
                l.append(int(item))
            for index in xrange(len(l)):
                if index == 0 and l[index] not in eps:
                    reduced.append(l[index])
                elif l[index-1] != l[index] and l[index] not in eps:
                    reduced.append(l[index])
            state_dict[content[0]] = reduced # get reduced state
        for key in state_dict.keys():
            hyp = state_dict[key]
            hyp_start_end = []
            hyp_symbol = []
            for item in hyp:
                if item in start_end_list:
                    hyp_start_end.append(item)
            index_pos = []
            for item in hyp_start_end:
                if start_dict.has_key(item):
                    hyp_symbol.append(start_dict[item])
                    index_pos.append(0)
                elif end_dict.has_key(item):
                    hyp_symbol.append(end_dict[item])
                    index_pos.append(1)
            real_output = []
            for index in xrange(len(index_pos)-1):
                if index_pos[index] == 0 and index_pos[index+1] == 1:
                    # indicate begin and end
                    if hyp_symbol[index] == hyp_symbol[index+1]:
                        real_output.append(hyp_symbol[index])
            output_phoneme_dict[key] = real_output
        for key in output_phoneme_dict.keys():
            fwobj.write(key)
            fwobj.write(' ')
            for item in output_phoneme_dict[key]:
                fwobj.write(item)
                fwobj.write(' ')
            fwobj.write('\n')

        frobj_state.close()
        frobj.close()
        fwobj.close()
        return output_phoneme_dict
                
def edit_distance(ref,hyp):
    """
    Edit distance between two sequences reference (ref) and hypothesis (hyp).
    Returns edit distance, number of insertions, deletions and substitutions to
    transform hyp to ref, and number of correct matches.
    """
    n = len(ref)
    m = len(hyp)

    ins = dels = subs = corr = 0
    
    D = np.zeros((n+1,m+1))

    D[:,0] = np.arange(n+1)
    D[0,:] = np.arange(m+1)

    for i in xrange(1,n+1):
        for j in xrange(1,m+1):
            if ref[i-1] == hyp[j-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j],D[i,j-1],D[i-1,j-1])+1

    i=n
    j=m
    while i>0 and j>0:
        if ref[i-1] == hyp[j-1]:
            corr += 1
        elif D[i-1,j] == D[i,j]-1:
            ins += 1
            j += 1
        elif D[i,j-1] == D[i,j]-1:
            dels += 1
            i += 1
        elif D[i-1,j-1] == D[i,j]-1:
            subs += 1
        i -= 1
        j -= 1

    ins += i
    dels += j

    return D[-1,-1]

def decodeSequence(sequence_probdist, blanksymbol):
    """
    This function decodes the output sequence from the network, which has the same length
    as the input sequence. This one takes just the most probable label and removes blanks and same phonemes
    """
    # just for testing, brute-force take output with highest prob and then eleminate repeated labels+blanks
    mostProbSeq = sequence_probdist.argmax(axis=1)
    reduced = mostProbSeq[mostProbSeq!=blanksymbol]
    reduced_end = np.array([seq for index, seq in enumerate(reduced) \
        if (reduced[index] != reduced[index-1] or index == 0)])
    return reduced_end, reduced, mostProbSeq

def decodeStateSequence(mostProbSeq, blanksymbol):
    """
    This function decodes the output sequence from the network, which has the same length
    as the input sequence. This one takes just the most probable label and removes blanks and same phonemes
    """
    # just for testing, brute-force take output with highest prob and then eleminate repeated labels+blanks
    reduced = mostProbSeq[mostProbSeq!=blanksymbol]
    reduced_end = np.array([seq for index, seq in enumerate(reduced) \
        if (reduced[index] != reduced[index-1] or index == 0)])
    return reduced_end, reduced
    

def decodeSequenceNoCTC(sequence_probdist, mask):
    """
    This function decodes each timestep by outputting the label with the highest probability,
    given an output distribution
    :parameters:
        - sequence_probdist: numpy array of output distribution num_seq x output_dim
        - mask: mask for marking which elements in sequence_probdist are valid
    """
    return np.array(sequence_probdist[mask==1].argmax(axis=1))
    

#    
#def beamsearch(sequence):
#    T = len(sequence)
#    #Initalise: B = {∅}; Pr(∅) = 1
#    B = [[]] #list of lists --> list of sequences! take indices of sequences to access Pr(...)
#    Pr = [1]
#    for t in range (1, T):
#        A = B
#        B = [[]]
#        for y_index, y in enumerate(A):
#            Pr(y_index) += Pyˆ∈pref(y)∩A Pr(yˆ) Pr(y|yˆ, t)
#    
#        while B contains less than W elements moreprobable than the most probable in A:
#            y∗ = most probable in A
#            Remove y∗from A
#            Pr(y∗) = Pr(y∗) Pr(∅|y, t)
#            Add y∗to B
#            for k ∈ Y:
#                Pr(y∗ + k) = Pr(y∗) Pr(k|y∗, t)
#                Add y∗ + k to A
#        Remove all but the W most probable from B
#        
#    return y with highest log Pr(y)/|y| in B 

def beamsearch(cost, extra, initial, B, E):
	"""A breadth-first beam search.
	B = max number of options to keep,
	E = max cost difference between best and worst threads in beam.
	initial = [ starting positions ]
	extra = arbitrary information for cost function.
	cost = fn(state, extra) -> (total_cost, [next states], output_if_goal)
    
    THIS FUNCTION IS HERE JUST FOR COMPARISON; WILL NEED OWN IMPLEMENTATION OF BEAMSEARCH
	"""

	o = []
	B = max(B, len(initial))
	hlist = [ (0.0, tmp) for tmp in initial ]
	while len(hlist)>0:
		# print "Len(hlist)=", len(hlist), "len(o)=", len(o)
		hlist.sort()
		if len(hlist) > B:
			hlist = hlist[:B]
		# print "E=", hlist[0][0], " to ", hlist[0][0]+E
		hlist = filter(lambda q, e0=hlist[0][0], e=E: q[0]-e0<=e, hlist)
		# print "		after: Len(hlist)=", len(hlist)
		nlist = []
		while len(hlist) > 0:
			c, point = hlist.pop(0)
			newcost, nextsteps, is_goal = cost(point, extra)
			if is_goal:
				o.append((newcost, is_goal))
			for t in nextsteps:
				nlist.append((newcost, t))
		hlist = nlist
	o.sort()
	return o   

def calcPER(tar, out):
    """
    This function calculates the Phoneme Error Rate (PER) of the decoded networks output
    sequence (out) and a target sequence (tar) with Levenshtein distance and dynamic programming.
    This is the same algorithm as commonly used for calculating the word error rate (WER)
        :parameters:
        - tar: target output
        - out: network output (decoded)
    :returns:
        - phoneme error rate
    """
    # initialize dynammic programming matrix
    D = np.zeros((len(tar)+1)*(len(out)+1), dtype=np.uint16)
    D = D.reshape((len(tar)+1, len(out)+1))
    # fill border entries, horizontals with timesteps of decoded networks output
    # and vertical with timesteps of target sequence.
    for t in range(len(tar)+1):
        for o in range(len(out)+1):
            if t == 0:
                D[0][o] = o
            elif o == 0:
                D[t][0] = t
                
    # compute the distance by calculating each entry successively. 
    # 
    for t in range(1, len(tar)+1):
        for o in range(1, len(out)+1):
            if tar[t-1] == out[o-1]:
                D[t][o] = D[t-1][o-1]
            else:
                # part-distances are 1 for all 3 possible paths (diag,hor,vert). 
                # Each elem of distance matrix D represents the accumulated part-distances
                # to reach this location in the matrix. Thus the distance at location (t,o)
                # can be calculated from the already calculated distance one of the possible 
                # previous locations(t-1,o), (t-1,o-1) or (t,o-1) plus the distance to the
                # desired new location (t,o). Since we are interested only in the shortes path,
                # take the shortes (min)
                substitution = D[t-1][o-1] + 1 # diag path
                insertion    = D[t][o-1] + 1 # hor path
                deletion     = D[t-1][o] + 1 # vert path
                D[t][o] = min(substitution, insertion, deletion)
    # best distance is bottom right entry of Distance-Matrix D.
    return float(D[len(tar)][len(out)])/len(tar)
    
def calcPERNoCTC(tar,out):
    """
    This function calculates the phoneme-error-rate, when not using CTC, but having a network output
    for every input. just compares target output (tar) and actual output (out)
    :parameters:
        - tar: target output
        - out: network output (decoded)
    :returns:
        - phoneme error rate
    """
    return (tar!=out).mean()       
     
def getPhonemeMapForScoring():
    '''
    This function maps the 61 phones from timit to the 39 phonemes that are commonly used.
    glottal stops, silence phones, etc. get each mapped to one class (or deleted (q))
    '''
    
    phonemes39 = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah',
                'oy', 'ow', 'uh', 'uw', 'er',
                'jh', 'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's',
                'sh', 'z', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng',
                'l', 'r', 'w', 'y', 'hh', 'sil']
               
    # construct dictionary with the 39 phonemes and with with the 61 phonemes for mapping
    dic39 = dict(zip(phonemes39, range(0,39)))           
    dic61 = featureExtraction.getPhonemeDictionary()
    phonemes61 = dic61.keys()
    
    strMap = dict(zip(phonemes61, phonemes61))
    strMap['ao'] = 'aa'
    strMap['ax'] = 'ah'
    strMap['ax-h'] = 'ah'
    strMap['axr'] = 'er'
    strMap['hv'] = 'hh'
    strMap['ix'] = 'ih'
    strMap['el'] = 'l'
    strMap['em'] = 'm'
    strMap['en'] = 'n'
    strMap['nx'] = 'n'
    strMap['eng'] = 'ng'
    strMap['zh'] = 'sh'
    strMap['ux'] = 'uw'
    
    strMap['pcl'] = 'sil'
    strMap['tcl'] = 'sil'
    strMap['kcl'] = 'sil'
    strMap['bcl'] = 'sil'
    strMap['dcl'] = 'sil'
    strMap['gcl'] = 'sil'
    strMap['h#'] = 'sil'
    strMap['pau'] = 'sil'
    strMap['epi'] = 'sil'
    del(strMap['q'])



    # Now we have a dict for 61 phonemes and a dict for the 39 phonemes (+blank)
    # map integers from 61-phn dict to integers from 39-phn dict (+ add blank)
    intMap = {}
    for str61 in strMap:
        str39 = strMap[str61]
        int61 = dic61[str61]
        int39 = dic39[str39]        
        intMap[int61] = int39    
    return intMap
    
    
def mapNetOutputs(netOutputs, scoreMap):
    '''
    This function maps 61 phonemes + blank to 39 phonemes + blank
    Adds up network outputs that are mapped to the same output for scoring
    '''    
    mappedOutputs = np.zeros((netOutputs.shape[0],40))
    for key in scoreMap.keys():
        mappedOutputs[:, scoreMap[key]] += netOutputs[:,key]
    # blank again last element
    mappedOutputs[:,-1] = netOutputs[:,-1]
    return mappedOutputs
    
def mapTargets(target, scoreMap):
    '''
    This function maps 61 phonemes + blank to 39 phonemes + blank 
    '''
    # make sure 'q' (=57) is not in the sequence, it is not included for scoring...
    # TODO: maybe handle that differently, don't like explicit 57 here
    target = target[target != 57.0]
    mappedSeq = [scoreMap[t] for t in target]
    return np.array([mappedSeq[i] for i in range(len(mappedSeq)) if mappedSeq[i-1]!=mappedSeq[i] or i == 0])

def read_phomenelabel(label):
    try:
        frobj = open(label,'r')
    except IOError:
        print "failed to read ", label
    else:
        label_seq = frobj.readlines()
        label_dict = OrderedDict()
        for eachline in label_seq:
            content = eachline.strip().split()
            utt = content[0]
            label = content[1:]
            label_dict[utt] = label
        frobj.close()
        return label_dict

def read_output(result, phoneme2int, output_phoneme):
    try:
        frobj = open(result,'r')
        frobj_phoneme = open(phoneme2int, 'r')
        fwobj_phoneme = open(output_phoneme, 'w')
    except IOError:
        print "failed to read ", result
        print "failed to read ", phoneme2int
    else:
        result = frobj.readlines()
        result_dict = OrderedDict()
        cnt = 0
        for eachline in result:
            content = eachline.strip().split()
            utt = content[0]
            label = content[1:]
            label_list = []
            reduced = []
            for item in label:
                label_list.append(int(item))
            for index in xrange(len(label_list)):
                if index == 0:
                    if label_list[index] != 0:
                        reduced.append(label_list[index])
                else:
                    if label_list[index] != 0 and label_list[index-1] != label_list[index]:
                        reduced.append(label_list[index])
            result_dict[utt] = reduced

        phoneme = frobj_phoneme.readlines()
        phoneme_dict = OrderedDict()
        for eachline in phoneme:
            content = eachline.strip().split()
            phoneme_name = content[0]
            phoneme_dict[int(content[1])] = phoneme_name

        utt_phoneme_dict = OrderedDict()
        for key in result_dict.keys():
            fwobj_phoneme.write(key)
            fwobj_phoneme.write(' ')
            seq = result_dict[key]
            phoneme_list = []
            for item in seq:
                phoneme_list.append(phoneme_dict[item])
                fwobj_phoneme.write(phoneme_dict[item])
                fwobj_phoneme.write(' ')
            fwobj_phoneme.write('\n')
            utt_phoneme_dict[key] = phoneme_list
        frobj.close()
        frobj_phoneme.close()
        fwobj_phoneme.close()
        return result_dict, phoneme_dict, utt_phoneme_dict

def phoneme_transform(phoneme_maps, utt_phoneme_dict, transform_type, output_transform_phoneme):
    try:
        frobj = open(phoneme_maps,'r')
        fwobj_transform_phoneme = open(output_transform_phoneme, 'w') 
    except IOError:
        print "failed to read ", phoneme_maps
    else:
        maps = frobj.readlines()
        maps_dict = OrderedDict()
        non_map = []
        for eachline in maps:
            content = eachline.strip().split('\t')
            if len(content) == 1:
                non_map.append(content[0])
                continue
            if transform_type == '61-to-39':
                maps_dict[content[0]] = content[2]
            elif transform_type == '61-to-48':
                maps_dict[content[0]] = content[1]
        transform_dict = OrderedDict()
        for key in utt_phoneme_dict.keys():
            transform_label = []
            label = utt_phoneme_dict[key]
            fwobj_transform_phoneme.write(key)
            fwobj_transform_phoneme.write(' ')
            for item in label:
                if item in non_map:
                    continue
                else:
                    transform_label.append(maps_dict[item])
                    fwobj_transform_phoneme.write(maps_dict[item])
                    fwobj_transform_phoneme.write(' ')
            fwobj_transform_phoneme.write('\n')
            transform_dict[key] = transform_label
        fwobj_transform_phoneme.close()
        frobj.close()
        return transform_dict

def read_state2int(state2int):
    try:
        frobj = open(state2int,'r') 
    except IOError:
        print "failed to read ", state2int
    else:
        info = frobj.readlines()
        state_num = 0
        for eachline in info:
            content = eachline.strip().split()
            state_num += (len(content[1:]))
        frobj.close()
        return state_num

def read_state2int_sil(state2int, blank_symbol):
    try:
        frobj = open(state2int,'r') 
    except IOError:
        print "failed to read ", state2int
    else:
        info = frobj.readlines()
        state_num = 0
        blank_l = []
        for eachline in info:
            content = eachline.strip().split()
            if content[0] == blank_symbol:
                for item in content[1:]:
                    blank_l.append(int(item))
            state_num += (len(content[1:]))
        frobj.close()
        return state_num, blank_l

if __name__ == '__main__':
    result = '/mnt/workspace/xuht/TIMIT/test_61_61.txt'
    phoneme2int = '/mnt/workspace/xuht/TIMIT/lang_phn/units.txt'
    output_phoneme = '/mnt/workspace/xuht/TIMIT/dev_phoneme_61_61.txt'
    fwobj_transform_phoneme = '/mnt/workspace/xuht/TIMIT/test_phoneme_39_39.txt'
    phoneme_maps = '/mnt/workspace/xuht/TIMIT/phones.60-48-39.map'
    label = '/mnt/workspace/xuht/TIMIT/test/text_39'

    result_dict, phoneme_dict, utt_phoneme_dict = read_output(result, phoneme2int, output_phoneme)
    transform_dict = phoneme_transform(phoneme_maps, utt_phoneme_dict, '61-to-39', fwobj_transform_phoneme)

    print result_dict
    
    [dev_truth_result_dict, 
    phoneme_dict, 
    dev_truth_utt_phoneme_dict] = read_output('/mnt/workspace/xuht/TIMIT/dev/labels.cv', phoneme2int, output_phoneme)
    dev_truth_transform_dict = phoneme_transform(phoneme_maps, dev_truth_utt_phoneme_dict, 
                                                        '61-to-39', fwobj_transform_phoneme)
    for key in phoneme_dict.keys():
        print key, phoneme_dict[key]
    
    ground_label = read_phomenelabel(label)
    score = []
    cnt = 0
    k = 0
    for key in transform_dict.keys():
        if k == 2:
            print ground_label[key]
            print transform_dict[key]
        tmp_score = edit_distance(ground_label[key], transform_dict[key])
        score.append(tmp_score)
        cnt += len(ground_label[key])
        k += 1
    print cnt
    score = np.asarray(score)
    print score
    print np.sum(score) / cnt

    state_seq = '/mnt/workspace/xuht/TIMIT/dev_22_61_61.txt'
    state_maps = '/mnt/workspace/xuht/TIMIT/lang_phn/units2state_sil_2.txt'
    output_state2phoneme = '/mnt/workspace/xuht/TIMIT/text_state2phoneme'
    output_phoneme_dict = state_sil_decoder(state_seq, state_maps, output_state2phoneme,[0])
    dev_transform_dict = phoneme_transform(phoneme_maps, output_phoneme_dict, 
                                                     '61-to-39', fwobj_transform_phoneme)

    #print dev_transform_dict
    # label = '/mnt/workspace/xuht/TIMIT/dev/text_39'
    # ground_label = read_phomenelabel(label)
    # score = []
    # cnt = 0
    # k = 0
    # for key in dev_transform_dict.keys():
    #     if k == 2:
    #         print ground_label[key]
    #         print dev_transform_dict[key]
    #     tmp_score = edit_distance(ground_label[key], dev_transform_dict[key])
    #     score.append(tmp_score)
    #     cnt += len(ground_label[key])
    #     k += 1
    # print cnt
    # score = np.asarray(score)
    # print score
    # print np.sum(score) / cnt

            
