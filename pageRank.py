import numpy as np
import json
import wiki
import os

def loadData(opts):
    adjMat=np.load(opts.adjMat_file)
    with open(opts.s2i_file, "r") as of:
        for line in of.readlines():
            dict_s2i=json.loads(line)
    for key, val in dict_s2i.items():
        dict_s2i.pop(key)
        dict_s2i[key]=int(val)
    with open(opts.i2s_file, "r") as of:
        for line in of.readlines():
            dict_i2s=json.loads(line)
    for key, val in dict_i2s.items():
        dict_i2s.pop(key)
        dict_i2s[int(key)]=val
    return adjMat, dict_s2i, dict_i2s

def HITS(adjMat):
    lenth=np.shape(adjMat)[0]
    auth=np.ones(lenth)
    hub=np.ones(lenth)
    #update the auth and hub
    threshold=0.01
    cnt=0
    while True:
        cnt+=1
        new_auth=np.zeros(lenth)
        new_hub=np.zeros(lenth)
        for i in range(lenth):
            new_auth[i]=np.matmul(adjMat[:,i], hub)
            new_hub[i]=np.matmul(adjMat[i], auth)
        new_auth=new_auth*lenth/np.sum(new_auth)
        new_hub=new_hub*lenth/np.sum(new_hub)
        #if norm_2(delta(auth))+norm_2(delta(hub)) < threshold, stop the loop
        delta=np.linalg.norm(auth-new_auth)+np.linalg.norm(hub-new_hub)
        auth, hub=new_auth, new_hub
        if cnt%10 == 0:
            print("CURRENT HITS LOOP: %d, DLETA:%f" % (cnt, delta))
        if delta<=0.01:
            break
    return auth, hub

def PageRank(adjMat):
    lenth=np.shape(adjMat)[0]
    values=np.ones(lenth)
    #update the pageranks
    threshold=0.01
    s=0.85
    cnt=0
    while True:
        cnt+=1
        new_values=np.zeros(lenth)
        for i in range(lenth):
            outs=adjMat[i]
            #if no output_edge, keep the value of itself
            if np.sum(outs)==0:
                new_values[i]+=values[i]
            #otherwise, divide the values to its neibors
            else:
                eachone=values[i]/np.sum(outs)
                dlt=np.ones(lenth)*eachone*outs
                assert (np.sum(dlt)-values[i])<0.00001
                new_values+=dlt
        #shrink and compensate.
        new_values*=s
        new_values+=(1-s)
        #check the delta(values), if < threshold then break the loop.
        delta=np.linalg.norm(new_values-values)
        if cnt%10==0:
            print("CURRENT PAGERANK LOOP: %d, DELTA:%f" % (cnt, delta))
        values=new_values
        if delta<=threshold:
            break
    return values

def getResMat(adjMat, dict_s2i, resFile, method):
    resMat=adjMat
    with open(resFile, "r") as file:
        for line in file.readlines():
            line=json.loads(line)
            #add the top-3 candidates
            resor=line[method][0:3]
            tgt=dict_s2i[line["Target"]]
            resorId=[dict_s2i[item] for item in resor]
            for item in resorId:
                resMat[item, tgt]=1
    return resMat

if __name__=="__main__":
    opts = wiki.Opt()
    adjMat, dict_s2i, dict_i2s = loadData(opts)
    auth_before, hub_before=HITS(adjMat)
    pageranks_before=PageRank(adjMat)

    #type: "MW", "SVD", "Frequency"
    resMat_mw=getResMat(adjMat, dict_s2i, opts.outfile_path, "MW")
    resMat_svd=getResMat(adjMat, dict_s2i, opts.outfile_path, "SVD")
    resMat_freq=getResMat(adjMat, dict_s2i, opts.outfile_path, "Frequency")

    auth_after_mw, hub_after_mw=HITS(resMat_mw)
    auth_after_svd, hub_after_svd=HITS(resMat_svd)
    auth_after_freq, hub_after_freq=HITS(resMat_freq)

    pageranks_after_mw=PageRank(resMat_mw)
    pageranks_after_svd=PageRank(resMat_svd)
    pageranks_after_freq=PageRank(resMat_freq)
    #save the resMat
    np.save("saved/resMat_mw", resMat_mw)
    np.save("saved/resMat_svd", resMat_svd)
    np.save("saved/resMat_freq", resMat_freq)
    #save the auth and hub
    np.save("pr_res/auth_before", auth_before)
    np.save("pr_res/hub_before", hub_before)
    np.save("pr_res/auth_after_mw", auth_after_mw)
    np.save("pr_res/auth_after_svd", auth_after_svd)
    np.save("pr_res/auth_after_freq", auth_after_freq)
    np.save("pr_res/hub_after_mw", hub_after_mw)
    np.save("pr_res/hub_after_svd", hub_after_svd)
    np.save("pr_res/hub_after_freq", hub_after_freq)
    #save the pagerank-results
    np.save("pr_res/pageranks_before", pageranks_before)
    np.save("pr_res/pageranks_after_mw", pageranks_after_mw)
    np.save("pr_res/pageranks_after_svd", pageranks_after_svd)
    np.save("pr_res/pageranks_after_freq", pageranks_after_freq)
