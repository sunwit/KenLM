import operator 
import numpy as np
import scipy.spatial
import subprocess
import string 
from  collections import OrderedDict
import sys
import kenlm
import en

contentWord = ['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNPS', 'NNS', 'NP', 'RB', 'RBR', 'RBS', 'RP', 'VA', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','VP', 'WRB']
def semantic(origin, candidate):
    if origin in vectorsWords.keys() and candidate in vectorsWords.keys():
       similar = 1 - scipy.spatial.distance.cosine(vectorsWords[candidate], vectorsWords[origin])
    else:
       similar = 1
    return similar
def context(origin, candidate):
    sum1 = 0
    iNum = int(wordNum[origin])
    origin_sentence = wordSentence[origin]
    sentence = ' '.join(origin_sentence)
    tt = len(origin_sentence)
    idx_1 = iNum - 1
    idx1 = iNum + 1
    n = 0
    listTuples = en.sentence.tag(sentence)
    while (idx_1 >= iNum-3 and idx_1 < iNum and idx_1>=0): 
        word_1 = listTuples[idx_1][0]
        wordPos = listTuples[idx_1][1]
        if ((word_1 in vectorsWords.keys()) and (wordPos in contentWord)):
           sum1 = sum1 +  1-scipy.spatial.distance.cosine(vectorsWords[candidate], vectorsWords[word_1]) 
           n = n + 1
        idx_1 = idx_1 - 1
    while (idx1<= iNum+3 and idx1>iNum and idx1<tt): 
        word1 = listTuples[idx1][0]
        wordPos = listTuples[idx1][1]
        if ((word1 in vectorsWords.keys()) and (wordPos in contentWord)):
           sum1 = sum1 + 1-scipy.spatial.distance.cosine(vectorsWords[candidate], vectorsWords[word1]) 
           n = n + 1
        idx1 = idx1 + 1
    if n != 0:
       sum1 = sum1 / (n * 1.0)
    else:
       sum1 = 1
    return sum1 


def lm_bigram_pre(origin, candidate):    
    iddNum = int(wordNum[origin])
    languageSentence = wordSentence[origin]
    listSentence = []
    if iddNum >0:
       listSentence.append(languageSentence[iddNum-1])   
       listSentence.append(candidate)
       likeliSentence = ' '.join(listSentence)
       model = kenlm.Model('lm-merged.kenlm')
       languageFeatures = model.score(likeliSentence, bos = True, eos = True)
    else:
       languageFeatures = 0
    return languageFeatures
def lm_bigram_post(origin, candidate):
    iddNum = int(wordNum[origin])
    languageSentence = wordSentence[origin]
    t = len(languageSentence)
    listSentence = []
    if iddNum < t-1:
       listSentence.append(candidate)
       listSentence.append(languageSentence[iddNum+1])   
       likeliSentence = ' '.join(listSentence)
       model = kenlm.Model('lm-merged.kenlm')
       languageFeatures = model.score(likeliSentence, bos = True, eos = True)
    else:
       languageFeatures = 0
    return languageFeatures
def lm_trigram_pre(origin, candidate):    
    iddNum = int(wordNum[origin])
    languageSentence = wordSentence[origin]
    listSentence = []
    if iddNum >1:
       listSentence.append(languageSentence[iddNum-2])   
       listSentence.append(languageSentence[iddNum-1])
       listSentence.append(candidate)
       likeliSentence = ' '.join(listSentence)
       model = kenlm.Model('lm-merged.kenlm')
       languageFeatures = model.score(likeliSentence, bos = True, eos = True)
    else:
       languageFeatures = 0
    return languageFeatures
def lm_trigram_post(origin, candidate):
    iddNum = int(wordNum[origin])
    languageSentence = wordSentence[origin]
    t = len(languageSentence)
    listSentence = []
    if iddNum < t-2:
       listSentence.append(candidate)
       listSentence.append(languageSentence[iddNum+1])   
       listSentence.append(languageSentence[iddNum+2])   
       likeliSentence = ' '.join(listSentence)
       model = kenlm.Model('lm-merged.kenlm')
       languageFeatures = model.score(likeliSentence, bos = True, eos = True)
    else:
       languageFeatures = 0
    return languageFeatures


def rankName(x):    
    lists = []
    dictions = {}
    rank = 0
    for candi in x.keys():
        if x[candi]  in lists:
            dictions[candi] = rank
        else:
            rank = rank + 1
            dictions[candi] = rank 
            lists.append(x[candi])
    return dictions

def Simplify():
    bestWords = {}
    for keyy in sorted(wordIdx.iterkeys()):
        scoresContext = {}
        scoresic = {}
        scores_bigram_pre = {}
        scores_bigram_post = {}
        scores_trigram_pre = {}
        scores_trigram_post = {}
        scoresSemantic = {}
        for sc in wordCandidate[int(keyy)]:
            scc = sc.strip().split(' ')
            length = len(scc)
            if length == 1:
               scoresSemantic[sc] = semantic(keyy, sc)
               if sc in icFreq.keys():
                  scoresic[sc] = icFreq[sc]
               else:
                  scoresic[sc] = 0
               if sc in vectorsWords.keys():
                  scoresContext[sc] = context(keyy, sc)
               else:
                  scoresContext[sc] = 1
               scores_bigram_pre[sc] = lm_bigram_pre(keyy,sc)
               scores_bigram_post[sc] = lm_bigram_post(keyy,sc)
               scores_trigram_pre[sc] = lm_trigram_pre(keyy,sc)
               scores_trigram_post[sc] = lm_trigram_post(keyy,sc)
            else:
               listTuplesScc = en.sentence.tag(sc)
               for k in range(length):
                  word = listTuplesScc[k][0]
                  pos = listTuplesScc[k][1]
                  if pos in contentWord:
                      scoresSemantic[sc] = semantic(keyy, word)
                      if word in vectorsWords.keys():
                         scoresContext[sc] = context(keyy, word)
                      else:
                         scoresContext[sc] = 1
                      if word in icFreq.keys():
                         scoresic[sc] = icFreq[word]
                      else:
                         scoresic[sc] = 0
                      scores_bigram_pre[sc] = lm_bigram_pre(keyy,word)
                      scores_bigram_post[sc] = lm_bigram_post(keyy,word)
                      scores_trigram_pre[sc] = lm_trigram_pre(keyy,word)
                      scores_trigram_post[sc] = lm_trigram_post(keyy,word)
                      break
                  if k == (length-1):
                      scoresic[sc] = 0
                      scores_bigram_pre[sc] = 0 
                      scores_bigram_post[sc] = 0 
                      scores_trigram_pre[sc] = 0 
                      scores_trigram_post[sc] = 0 
                      scoresSemantic[sc] = 1

        maxSemantic = max(scoresSemantic.iteritems(), key=operator.itemgetter(1))[1]
        minSemantic = min(scoresSemantic.iteritems(), key=operator.itemgetter(1))[1]

        max_bigram_pre = max(scores_bigram_pre.iteritems(), key=operator.itemgetter(1))[1]
        min_bigram_pre = min(scores_bigram_pre.iteritems(), key=operator.itemgetter(1))[1]

        max_bigram_post = max(scores_bigram_post.iteritems(), key=operator.itemgetter(1))[1]
        min_bigram_post = min(scores_bigram_post.iteritems(), key=operator.itemgetter(1))[1]

        max_trigram_pre = max(scores_trigram_pre.iteritems(), key=operator.itemgetter(1))[1]
        min_trigram_pre = min(scores_trigram_pre.iteritems(), key=operator.itemgetter(1))[1]

        max_trigram_post = max(scores_trigram_post.iteritems(), key=operator.itemgetter(1))[1]
        min_trigram_post = min(scores_trigram_post.iteritems(), key=operator.itemgetter(1))[1]

        
        maxic = max(scoresic.iteritems(), key=operator.itemgetter(1))[1]
        minic = min(scoresic.iteritems(), key=operator.itemgetter(1))[1]
       
        
        averageScore = {}
        for sc in wordCandidate[int(keyy)]:    
            if maxSemantic != minSemantic:
               scoresSemantic[sc] = (scoresSemantic[sc] - minSemantic)/(maxSemantic - minSemantic)
            else:
               scoresSemantic[sc] = 0.5
            if max_bigram_pre != min_bigram_pre:
               scores_bigram_pre[sc] = (scores_bigram_pre[sc] - min_bigram_pre)/(max_bigram_pre - min_bigram_pre)
            else:
               scores_bigram_pre[sc] = 0.5
            if max_bigram_post != min_bigram_post:
               scores_bigram_post[sc] = (scores_bigram_post[sc] - min_bigram_post)/(max_bigram_post - min_bigram_post)
            else:
               scores_bigram_post[sc] = 0.5

            if max_trigram_pre != min_trigram_pre:
               scores_trigram_pre[sc] = (scores_trigram_pre[sc] - min_trigram_pre)/(max_trigram_pre - min_trigram_pre)
            else:
               scores_trigram_pre[sc] = 0.5
            if max_trigram_post != min_trigram_post:
               scores_trigram_post[sc] = (scores_trigram_post[sc] - min_trigram_post)/(max_trigram_post - min_trigram_post)
            else:
               scores_trigram_post[sc] = 0.5


            if maxic != minic:
               scoresic[sc] = (scoresic[sc] - minic)/(maxic-minic)
            else:
               scoresic[sc] = 0.5
            averageScore[sc] = (scores_bigram_pre[sc] + scores_bigram_post[sc] + scores_trigram_pre[sc] + scores_trigram_post[sc] + scoresic[sc] + scoresSemantic[sc])/6.0
   
 
  

        rAverageScore = sorted(averageScore.items(),key=operator.itemgetter(1), reverse = True)
        rankWord = rankName(OrderedDict(rAverageScore))
               
        
        middleDic = {}
        ss = []
        for key5 in rankWord.keys():
            rankScore = rankWord[key5]
            if rankScore in ss:
                middlelist = []
                if type(middleDic[rankScore]) is list:
                    for dd, vue in enumerate(middleDic[rankScore]):
                        middlelist.append(vue)
                    middlelist.append(key5)
                    middleDic[rankScore] = middlelist
                else:
                    middlelist.append(middleDic[rankScore])
                    middlelist.append(key5)
                    middleDic[rankScore] = middlelist
            else:
                middleDic[rankScore] = key5
                ss.append(rankScore)

        listAll = []
        middle = sorted(middleDic.items(), key=operator.itemgetter(0))
        orderdict = OrderedDict(middle)
        for key5 in orderdict.keys(): 
            listAll.append(orderdict[key5])
        bestWords[int(keyy)] = listAll
    return bestWords    
        


if __name__ == "__main__":
    vectorsWords = {}
    wordNum = {}
    wordIdx = {}
    wordSentence = {}
    wordCandidate = {}
    frelist = []
    repeatlist = []
    icFreq = {}

    with open("glove.6B.200d.txt", "r") as f:
        for x in f.readlines():
            key = x.rstrip().split()[0]
            listVector  = x.rstrip().split()[1:]
            vectorsWords[key] = np.asarray(listVector, dtype = float)

    with open("lst", "r") as f:
        for x in f.readlines():
            strWord = x.rstrip().split('\t')[0]
            key = strWord.split('.')[0]
            idx = x.rstrip().split('\t')[1]
            num = x.rstrip().split('\t')[2]
            sentence = x.rstrip().split('\t')[3]
            wordIdx[idx] = key
            wordNum[idx] = num
            sentences = sentence.split(' ')
            wordSentence[idx] = sentences

    with open("sub", "r") as f:
        lineNum = 300
        for i,x in enumerate(f.readlines()):
           lineNum = lineNum + 1
           wordCandidate[lineNum] = x.rstrip().split(';')[0:-1]
    with open("fuck" , "r") as f:       
        for x in f.readlines():
            word = x.rstrip().split("\t")[0]
            freq = x.rstrip().split("\t")[1]
            if word not in icFreq.keys():
                  icFreq[word] = float(freq)
               
           
    rankKeyWords = Simplify()

    with open("baba", "a") as f:
        for idxx in sorted(rankKeyWords.iterkeys()):
            f.write("Sentence " + str(idxx) + " rankings: ") 
            for idd, value in enumerate(rankKeyWords[idxx]):
                if type(value) is list:
                    listss = value
                    f.write("{")
                    tt = len(listss)
                    for idxxx, valueee in enumerate(listss):
                        if idxxx <  (tt-1):
                            f.write(str(valueee) + ", ")
                        if idxxx == (tt-1):
                            if idd != len(rankKeyWords[idxx])-1:
                                f.write(str(valueee) + "} ")
                            else:
                                f.write(str(valueee) + "}")
                else:
                    if idd != len(rankKeyWords[idxx])-1:
                        f.write("{" + str(value) + "} ")
                    else:
                        f.write("{" + str(value) + "}")
            f.write("\n")
