#!/usr/bin/python
# coding: utf-8
import time
import cPickle
import numpy as np
import scipy.io
from collections import OrderedDict, defaultdict
import datetime
import json

if __name__ == '__main__':
    if_test = True
    k_near = 10
    
    if if_test:
        print 'loading coco data...'
        x = None
        with open("./coco/data.p", "rb") as fd:
            x = cPickle.load(fd)
        train, val, test, wordtoix, ixtoword = x
        train = list(train)
        val = list(val)
        test = list(test)
        img_data = [train[0] + val[0] + test[0], train[1] + val[1] + test[1], train[2] + val[2] + test[2]]
        img_data = tuple(img_data)
        del x, train, val, test
        
        print 'create dict...'
        img_dict = {}
        for i in range(len(img_data[2])):
            if img_data[1][i] in img_dict:
                if img_dict[img_data[1][i]]['name'] == img_data[2][i]:
                    if img_dict[img_data[1][i]]['name'] == 'COCO_val2014_000000507065.jpg':
                        test_index = [img_data[1][i]]
                    img_dict[img_data[1][i]]['sens'].append(img_data[0][i])
                else:
                    print "wrong!"
            else:
                img_dict[img_data[1][i]] = {'name':img_data[2][i], 'sens': [img_data[0][i]]}
        print len(img_dict)
        print 'loading resnet features near info...'
        data = cPickle.load(open("./resnet_feats_10near.p","rb"))
        img_feats_list = data['img_feats_list'][:k_near,:].T
        img_feats_score = data['img_feats_score'][:k_near,:].T


        for t in test_index:
            print img_dict[t]['name']
            for (near_id, score) in zip(img_feats_list[t], img_feats_score[t]):
                print '*****************'
                print 'like_score: ' + str(score)
                print 'img name: ' + img_dict[near_id]['name']




        

