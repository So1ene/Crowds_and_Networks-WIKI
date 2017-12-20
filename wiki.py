import numpy as np
import urllib
import json
import os

class Opt():
    def __init__(self):
        self.article_path="paths_and_graph/articles.tsv"
        self.paths_file="paths_and_graph/paths_finished.tsv"
        self.links_file="paths_and_graph/links.tsv"
        self.outfile_path="saved/results.json"
        self.dataInfo_path="saved/statistics"
        self.adjMat_file="saved/Adjacent_Mat.npy"
        self.s2i_file="saved/dict_s2i.json"
        self.i2s_file="saved/dict_i2s.json"
        self.K=256
        self.SkipPart=0.5

    #check whether the graph_info and the pathes match well.
    def checkPath(self, adjMat, path_dict):
        cnt=0
        for key, val in path_dict.items():
            for path in val:
                for i in range(len(path)-1):
                    if(adjMat[path[i], path[i+1]]!=1):
                        val.remove(path)
                        cnt+=1
                        break
        return cnt

    #load the dataset
    def loadData(self):
        #get the nodes from the article_file.
        def getNode(article_path):
            with open(article_path, "r") as of:
                node_s2i=dict()
                node_i2s=dict()
                skipForeword=False
                for line in of.readlines():
                    if not skipForeword:
                        if line[0]=="#":
                            continue
                        skipForeword=True
                    else:
                        tempNode=urllib.unquote(line.split()[0])
                        if tempNode not in node_s2i.keys():
                            node_s2i[tempNode]=len(node_s2i)
                            node_i2s[len(node_i2s)]=tempNode
            assert(len(node_s2i)==len(node_i2s))
            return node_s2i, node_i2s

        #get users' pathes from the path_file.
        def getPath(paths_file):
            path_dict=dict()
            with open(paths_file, "r") as of:
                skipForeword=False
                for line in of.readlines():
                    if not skipForeword:
                        if line[0]=="#":
                            continue
                        skipForeword=True
                    else:
                        tpath=line.split()[3].split(";")
                        #Remove the back_click and map str to idx
                        #keyError=False
                        ipath=[]
                        for item in tpath:
                            if item=='<' and len(ipath)!=0:
                                ipath.pop()
                                continue
                            idx=node_s2i[urllib.unquote(item)]
                            ipath+=[idx]
                        target=ipath[-1]
                        if target not in path_dict.keys():
                            path_dict[target]=[]
                        path_dict[target].append(ipath)
            return path_dict

        #get the links among all the nodes.
        def getLinks(links_file):
            # adjacent_matrix with form (source_idx, target_idx)
            adjMat=np.zeros((len(node_s2i), len(node_s2i)), dtype=int)
            with open(links_file, "r") as of:
                skipForeword=False
                for line in of.readlines():
                    if not skipForeword:
                        if line[0]=="#":
                            continue
                        skipForeword=True
                    else:
                        line=line.split()
                        assert len(line)==2
                        src_idx, tgt_idx = ( node_s2i[urllib.unquote(line[0])]
                                    ,node_s2i[urllib.unquote(line[1])] )
                        adjMat[src_idx, tgt_idx]=1
            return adjMat

        node_s2i, node_i2s = getNode(self.article_path)
        path_dict = getPath(self.paths_file)
        adjMat = getLinks(self.links_file)

        removed=self.checkPath(adjMat, path_dict)
        print("%d Paths not Comparaible with the Links are removed" % (removed))
        #Output Statistics Infor.
        self.print_Statistics(node_s2i, path_dict, adjMat)

        #If have not saved, save the adjacent matrix and the dicts
        if not os.path.isfile(self.adjMat_file):
            np.save(self.adjMat_file[:-5], adjMat)
        if not os.path.isfile(self.s2i_file):
            with open(self.s2i_file,"w") as of:
                of.write(json.dumps(node_s2i))
        if not os.path.isfile(self.i2s_file):
            with open(self.i2s_file,"w") as of:
                of.write(json.dumps(node_i2s))
        return adjMat, path_dict, node_s2i, node_i2s

    #output the statistics of the dataset.
    def print_Statistics(self, node_s2i, path_dict, adjMat):
        with open(self.dataInfo_path, "w") as of:
            #Statistics
            of.write("==============Statistics of the Dataset================\n")
            shape=np.shape(adjMat)
            assert(shape[0]==shape[1])
            print("Total Number of nodes: %d" % (shape[0]))
            print("Total Number of links: %d" % (np.sum(adjMat)))
            #About the path data
            path_cnt, max_cnt, min_cnt=0, 0, 0x3f3f3f3f
            lenth_cnt, max_len, min_len=0, 0, 0x3f3f3f3f
            for key, val in path_dict.items():
                path_cnt+=len(val)
                max_cnt=max(max_cnt, len(val))
                min_cnt=min(min_cnt, len(val))
                for item in val:
                    lenth_cnt+=len(item)
                    max_len=max(max_len, len(item))
                    min_len=min(min_len, len(item))
            of.write("Number of distinct targets: %d\n" % len(path_dict))
            of.write("Average Number of Paths per Target: %.2f\n" % (path_cnt*1.0/len(path_dict)))
            of.write("Maximum Number of Paths per Target: %.2f\n" % (max_cnt))
            #of.write("Minimum Number of Paths per Target: %.2f\n" % (min_cnt))
            of.write("Average length of Paths: %.2f\n" % (lenth_cnt*1.0/path_cnt))
            of.write("Maximum length of Paths: %.2f\n" % (max_len))
            #of.write("Minimum length of Paths: %.2f\n" % (min_len))
            #About the graph
            link_cnt=0
            in_max, out_max=0, 0
            in_min, out_min=0x3f3f3f3f, 0x3f3f3f3f
            for idx in range(np.shape(adjMat)[0]):
                out_cnt=np.sum(adjMat[idx])
                in_cnt=np.sum(adjMat[:,idx])
                link_cnt+=out_cnt
                out_max=max(out_max, out_cnt)
                in_max=max(in_max, in_cnt)
                out_min=min(out_min, out_cnt)
                in_min=min(in_min, out_cnt)
            of.write("Average in_degree = out_degree = %.2f\n" % (link_cnt*1.0/np.shape(adjMat)[0]))
            of.write("Maximum in_degree: %d\n" % (in_max))
            of.write("Maximum out_degree: %d\n" % (out_max))
            #of.write("Minimum in_degree: %d\n" % (in_min))
            #of.write("Minimum out_degree: %d\n" % (out_min))
            of.write("=====================End of Output======================\n")

class Wiki():
    def __init__(self, opt):
        self.opt = opt

    #extract and rerank the possible candidate for certain target and paths_set
    def candidate(self, adjMat, target, paths, ranking, appro_Mat=None):
        assert (ranking in ["mw", "svd", "frequency"])

        candidator=[]
        for pt in paths:
            #discard the forehalf of the path
            for i in range(int(len(pt)*self.opt.SkipPart), len(pt)):
                #Not consider the source and target node.
                if i==0 or i==(len(pt)-1):
                    continue
                if pt[i] not in candidator:
                    candidator+=[pt[i]]
        #Filter the candidator
        idx=0
        for i in range(len(candidator)):
            #if the edge already exists.
            if adjMat[candidator[idx], target]==1:
                candidator.pop(idx)
            else:
                idx+=1
        assert idx==len(candidator)

        def mw(s):
            in_s=adjMat[:,s]
            in_t=adjMat[:,target]
            upper=(np.log(max(np.sum(in_s), np.sum(in_t))) - np.log(max(np.matmul(in_s,in_t), 1)))*1.0
            assert min(np.sum(in_s), np.sum(in_t))>=1
            lower=(np.log(np.shape(adjMat)[0])-np.log(min(np.sum(in_s), np.sum(in_t))))*1.0
            distance=1-upper/lower
            return distance

        if ranking=="mw":
            #rank the candidators according to the MW distance
            candidator=sorted(candidator, key=mw, reverse=True)
        elif ranking=="svd":
            candidator=sorted(candidator, key=(lambda s:appro_Mat[s,target]), reverse=True)
        elif ranking=="frequency":
            count=dict()
            for node in candidator:
                count[node]=0
            #calculate the frequency
            for pt in paths:
                for node in pt[1:-1]:
                    if node in candidator:
                        count[node]+=1
            candidator=sorted(candidator, key=(lambda s:count[s]), reverse=True)
        return candidator

    #use SVD to get the rank-k approximation of the original adjacent matrix.
    def getApprox(self, k, adjMat):
        # calculate the rank-k approximation Ak of the Input_Matrix
        shape=np.shape(adjMat)
        u,s,v=np.linalg.svd(adjMat)
        res=np.zeros(shape, dtype=float)
        k=min(shape[0], shape[1], k)
        for i in range(k):
            #shape[0]
            u_v=np.reshape(u[:,i], (shape[0], 1))
            v_v=np.reshape(v[i,], (1, shape[1]))
            res+=np.matmul(u_v, s[i]*v_v)
        return res

    #calculate the candidates
    def doCalculate(self, adjMat, path_dict, dict_s2i):
        #get the rank-k approximation matrix of the adjacent matrix
        if os.path.isfile('saved/approx_mat_K'+str(self.opt.K)+'.npy'):
            print("======Loading the approximation of adjacent matrix======")
            appro_Mat=np.load('saved/approx_mat_K'+str(self.opt.K)+'.npy')
        else:
            print("======calculating the approximation of adjacent matrix======")
            appro_Mat=getApprox(K, adjMat)
            np.save("saved/approx_mat_K"+str(K), appro_Mat)

        print("======calculating the candidators=======")
        #process all the data.
        outfile=open(self.opt.outfile_path, "w")
        for key, val in path_dict.items():
            res=dict()
            #Get the candidator returned by the algorithm proposed in the paper.
            candidator_mw=self.candidate(adjMat, target=key, paths=val, ranking="mw")
            candidator_svd=self.candidate(adjMat, target=key, paths=val, ranking="svd", appro_Mat=appro_Mat)
            candidator_freq=self.candidate(adjMat, target=key, paths=val, ranking="frequency")
            paths=[]
            for path in val:
                pt=[node_i2s[node] for node in path]
                paths.append(pt)
            res_mw=[node_i2s[node] for node in candidator_mw]
            res_svd=[node_i2s[node] for node in candidator_svd]
            res_freq=[node_i2s[node] for node in candidator_freq]
            res["Target"]=node_i2s[key]
            res["Paths"]=paths
            res["MW"]=res_mw
            res["Frequency"]=res_freq
            res["SVD"]=res_svd
            outfile.write(json.dumps(res)+"\n")
        outfile.close()

if __name__=="__main__":
    opts=Opt()
    adjMat, path_dict, node_s2i, node_i2s=opts.loadData()
    wiki=Wiki(opts)
    wiki.doCalculate(adjMat, path_dict, node_s2i)
