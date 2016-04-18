//  !/usr/bin/env python
//   -*- coding: utf-8 -*-

//   Latent Dirichlet Allocation + collapsed Gibbs sampling
//   This code is available under the MIT License.
//   (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
//   Updated by Jey Han Lau, 2012
//   Updates:
//   -Parallelisation of the Gibbs Sampler
//   -Online Processing of Documents
//   Related Paper:
//   -On-line Trend Analysis with Topic Models: //  twitter trends detection topic model online
//     (COLING 2012)

import numpy
import operator
import time
import os
import pickle
from multiprocessing import Pool
import threading

auto  parallelInference( i, st, ed, o_docs, o_ZMN , o_n_m_z, o_n_z_t, o_n_z)
{

//      print "i =", i, "st =", st, "ed =", ed, "docs =", o_docs
//      print "BEFORE:"
//      print "\tZMN  =", o_ZMN 
//      print "\tn_m_z =", o_n_m_z
//      print "\tn_z_t =", o_n_z_t
//      print "\tn_z =", o_n_z

    foreach( m, doc; enumerate(o_docs))
    {
        z_n = o_ZMN [m];
        n_m_z = o_n_m_z[m];
        foreach( n, t; enumerate(doc))
        {
            //   discount foreach( n-th word t with topic z
            z = z_n[n];
            --n_m_z[z];
            --o_n_z_t[z, t];
            --o_n_z[z];


            //   sampling topic new_z foreach( t
            //   nanjunxiao ps: p(z|doc,-t) sim= p(t|z)*p(z|doc),then 轮盘赌,必须采样，不能简单使用最大后验概率
            //   p(t|z)*p(z|doc)=(nd,z + alpha/sum(nd,z+alpha) ) / (nz,t+beta/sum(nz,t+beta) )
            //   and sum(nd,z+alpha)与当前词的主题分配无关，文档长度常量.
            p_z = o_n_z_t[:, t] * n_m_z / o_n_z;
            new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax();

            //   set z the new topic and increment counters
            z_n[n] = new_z;
            ++n_m_z[new_z];
            ++o_n_z_t[new_z, t];
            ++o_n_z[new_z];
        }
    }


//      print "AFTER:"
//      print "\tZMN  =", o_ZMN 
//      print "\tn_m_z =", o_n_m_z
//      print "\tn_z_t =", o_n_z_t
//      print "\tn_z =", o_n_z

    return (i, st, ed, o_ZMN , o_n_m_z, o_n_z_t, o_n_z);
}

struct LDA
{
    int numTopics; // K
    double alpha;
    double beta;
    double cont;
    int numProcessors;


    auto  __init__(this , K, alpha, beta, cont, docs, docs_nt, V, docsTimes , outputDir , prevLDA , nproc, smartinit=True)
    {
        this.K = K;
        this.alpha = alpha; //   parameter of topics prior
        this.beta = beta;   //   parameter of words prior
        this.cont = cont; //   contribution proportion of history topics
        this.docs = docs;
        this.docs_nt = docs_nt;
        this.docsTimes  = docsTimes;
        this.V = V;
        this.outputDir  = outputDir;
        this.nproc = nproc;
        this.tlock = threading.Lock();

        this.ZMN  = []; //   topics of words of documents
        this.n_m_z = numpy.zeros((this.docs.length, K)) + alpha;
        this.n_z_t = numpy.zeros((K, V)) + beta; //   word count of each topic and vocabulary
        this.n_z_t_new = numpy.zeros((K, V)); //   new word count of each topic and vocabulary
        this.n_z = numpy.zeros(K) + V * beta;    //   word count of each topic

        if (prevLDA  != None)
        {
            //  convert the old model's topic-word matrix counts to proportion
            sum_n_z_t = 0;
            foreach( z_t; prevLDA.n_z_t)
                sum_n_z_t += sum(z_t);
            foreach( (z, z_t);enumerate(prevLDA .n_z_t))
            {
                foreach( (t, val); enumerate(z_t)
                {
                    this .n_z_t[z, t] = ((prevLDA.n_z_t[z,t].to!double / sum_n_z_t)* this.V * this.K * this.beta* this.cont)
                        + (this .beta*(1.0-this .cont));
                }
            }
            foreach( (z, val);enumerate(this.n_z))
                this .n_z[z] = sum(this.n_z_t[z]);

            foreach( (row_id, row);enumerate(prevLDA.n_m_z)
            {
                foreach( (col_id, col);enumerate(row))
                    this.n_m_z[row_id][col_id] = col;
            }
        }  
        this .N = 0

        foreach( m, doc in enumerate(docs):
            this .N += len(doc)
            z_n = []
            foreach( t in doc:
                if smartinit:
                    p_z = this .n_z_t[:, t] * this .n_m_z[m] / this .n_z
                    z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                else:
                    z = numpy.random.randint(0, K)
                z_n.append(z)
                this .n_m_z[m, z] += 1
                this .n_z_t[z, t] += 1
                this .n_z[z] += 1
            this .ZMN .append(numpy.array(z_n))

            //  update the document timestamp
            prev_time = docsTimes [m]

    auto  parallelInferenceComplete (this , result):
        (i, st, ed, ZMN , n_m_z, n_z_t, n_z) = result

        this .tlock.acquire()
        //  update ZMN  and n_m_z
        this .ZMN [st:ed] = ZMN 
        this .n_m_z[st:ed] = n_m_z

        //  update n_z_t (reduce-scatter operation)
        this .n_z_t_new = this .n_z_t_new + (n_z_t - this .n_z_t)

        this .tlock.release()
        

    auto  inference(this ):
//          print "ORIGINAL:"
//          print "\tdocs =", this .docs
//          print "\tZMN  =", this .ZMN 
//          print "\tn_m_z =", this .n_m_z
//          print "\tn_z_t =", this .n_z_t
//          print "\tn_z =", this .n_z
        //  refesh the n_z_t array used foreach( storing new counts
        this .n_z_t_new = numpy.zeros((this .K, this .V))
        //  Spawn a number of threads to do the inference
        po = Pool()
        num_doc_per_proc = float(len(this .docs))/this .nproc
        foreach( i in range(0, this .nproc):
            st = int(round(float(i)*num_doc_per_proc))
            ed = int(round(float(i+1)*num_doc_per_proc))
            po.apply_async(parallel_inference, \
                (i, st, ed, this .docs[st:ed], this .ZMN [st:ed], this .n_m_z[st:ed], \
                this .n_z_t, this .n_z), callback=this .parallelInferenceComplete )
    
        po.close()
        po.join()

        //  update n_z_t
        this .n_z_t = this .n_z_t + this .n_z_t_new
        //  update n_z
        this .n_z = numpy.sum(this .n_z_t, 1)

//          print "MERGED:"
//          print "\tZMN  =", this .ZMN 
//          print "\tn_m_z =", this .n_m_z
//          print "\tn_z_t =", this .n_z_t
//          print "\tn_z =", this .n_z

    auto  worddist(this ):
        """get topic-word distribution"""
        return this .n_z_t / this .n_z[:, numpy.newaxis]

    auto  perplexity(this , docs=None):
        if docs == None: docs = this .docs
        phi = this .worddist()
        log_per = 0
        N = 0
        Kalpha = this .K * this .alpha
        foreach( m, doc in enumerate(docs):
            theta = this .n_m_z[m] / (len(this .docs[m]) + Kalpha)
            foreach( w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)

//  added by nanjunxiao ,theta & phi
auto computeTheta (this )
{
    thetaFile  = File(this.outputDir  + "/theta.txt", "w");
    Kalpha = this .K * this .alpha;
    foreach( m, doc; enumerate(this .docs))
    {
        theta = this.n_m_z[m] / (this.docs[m].length + Kalpha);
        foreach( k; 0.. this.numTopics)
            thetaFile.write(theta[k].to!string ~" ");
        thetaFile.writefln("")
    }
}
 

auto dropOneDayData (voca, lda, alpha)
{
    n_firstday = lda.docs_nt[0];
    //  decrement the counts of topic-word matrix foreach( the documents to be removed
    foreach( (m, doc) in enumerate(lda.docs[:n_firstday])
    {
        foreach( (n, t) in enumerate(doc)
        {
            z = lda.ZMN [m][n];
            --lda.n_z_t[z, t];
            --lda.n_z[z];
            --voca.wordfreq[t];
        }
    }

    lda.docs = lda.docs[n_firstday..$];
    lda.docs_nt = lda.docs_nt[1..$];
    lda.docsTimes  = lda.docsTimes [n_firstday..$];
    lda.ZMN  = lda.ZMN [n_firstday..$]
    lda.n_m_z = lda.n_m_z[n_firstday..$]

    //  convert the n_m_z counts to priors
    total_n_m_z = 0.0;
    foreach( n_m; lda.n_m_z)
        total_n_m_z += sum(n_m);
    foreach( (m, n_m); enumerate(lda.n_m_z))
    {
        foreach( (z, count); enumerate(n_m))
        {
            new_prior = (float(count)/(total_n_m_z))*len(lda.n_m_z)*lda.K*alpha;
            lda.n_m_z[m][z] = new_prior;
        }
    }

    return tuple(voca, lda);
}

auto  ldaLearning (lda, iteration, voca):
    pre_perp = lda.perplexity()
    print "initial perplexity=%f" % pre_perp
    foreach( i in range(iteration):
        start = time.time()
        lda.inference()
        print "(%.1fs) iter=%d" % (time.time()-start, i + 1),
        if ( (i+1)%50 == 0):
            perp = lda.perplexity()
            print "p=%f" % (perp)
        else:
            print
    outputWordTopicDist (lda, voca)
    //  added by nanjunxiao
    lda.computeTheta ()


auto outputWordTopicDist(lda, voca)
{
    auto phi = lda.worddist();
    topicsFile  = File(lda.outputDir  ~ "/topics.txt", "w");
    foreach( k; 0.. lda.NumTopics)
    {
        //  writefln("\n-- topic: %s",
        foreach( w in numpy.argsort(-phi[k])[:15]
        {
            //  print "%s: %f" % (voca[w], phi[k,w])
            topicsFile.write(voca[w] + " ");
        }
        topicsFile.writefln("");
    }
}

struct Options
{
    string filename;
    string timeFile;
    string outputDir;
    ModelName model;
    double alpha = 0.001;
    double beta = 0.01,
    double contributionProportion = 0.5;
    int numTopics = 50; // K
    int iterationCount = 500;
    bool smartInit = true;
    bool excludeStopWords = true;
    int seed = -1;
    int wordFreq=1;
    int numProcessors=4;
}
int main(string[] args)
{
    import std.getopt;
    auto helpInformation = getopt(
        "filename|f", "corpus filename", &options.filename,
        "timeFile|t", "timestamp of documents", &options.timeFile,
        "outputDir|o", "output directory", &options.outputDir,
        "model|m", "previously trained model", &options.model,
        "alpha", "parameter alpha", &options.alpha,
        "beta", "parameter beta", &options.beta,
        "cont|p", "parameter contribution proportion", &options.contributionProportion,
        "numTopics|k", "number of topics", &options.numTopics;
        "i", "iteration count", &iterationCount,
        "smartinit", "smart initialization of parameters", &options.smartInit,
        "stopwords", "exclude stopwords", &options.excludeStopWords,
        "seed", "random seed", &options.seed,
        "wordfreq|wf", "threshold of word frequency to cut words", &options.wordFreq,
        "numproc", "number of processors", &options.numProcessors
    );

    if (helpInformation.wanted)
    {
        defaultGetOptPrinter("LDA information", helpInformation.options);
        return -1;
    }
    enforce(options.filename.length && options.timeFile.length && options.outputDir.length, 
        "need corpus filename -f and document timestamp file -t and output directory -o");

    auto corpus = vocabulary.loadFile(options.filename)
    if (options.seed >-1)
        numpy.random.seed(options.seed);

    if (!exists(options.outputDir)
        mkdirRecurse(options.outputDir);

    auto voca = Vocabulary.cocabulary(options.stopwords, options.wordFreq);

    if (options.model.length>0)
    {
        (prevVoca , prevLDA ) = pickle.load(open(options.model));
        //  drop one day worth's of data to accommodate the new day's data
        prevVoca , prevLDA  = dropOneDayData (prevVoca , prevLDA , options.alpha);
        options.numTopics = prevLDA.numTopics;
    }
    else
    {
        prevLDA  = None;
        prevVoca  = None;
    }

    //  generate the vocabularies foreach( voca
    voca.genVocabs (corpus, prevVoca , prevLDA )

    docs = [voca.doc_to_ids(doc) foreach( doc in corpus]
    //  calculate the number of elements foreach( each timestamp group in docs
    docs_nt = []
    docsTimes  = [ item.strip() foreach( item in open(options.timeFile ).readlines() ]
    tmpNT  = {}
    foreach( time in set(docsTimes ):
        tmpNT [time] = docsTimes .count(time)
    foreach( (time, count) in sorted(tmpNT .items()):
        docs_nt.append(count)
    tmpNT .clear()

    if options.model
    {
        docs = prevLDA.docs ~ docs;
        docsTimes  = prevLDA.docsTimes ~ docsTimes;
        docsNT = prevLDA.docsNT ~ docsNT;
    }
    //  if options.wf > 0: docs = voca.cutLowFreq (docs, options.wf)

    //  initialise lda
    lda = LDA(options.K, options.alpha, options.beta, options.cont, docs, docs_nt, voca.size(),
        docsTimes , options.outputDir , prevLDA , options.nproc, options.smartinit);

    //  print word frequency
    freqword = {}
    freqWordFile  = File(lda.outputDir  ~ "/freqwords.txt", "w");
    foreach(entry; voca.wordfreq)
        freqword[voca.vocas[entry.vocab_id]] = entry.freq;

    foreach( entry;sorted(freqword.items(), key=operator.itemgetter(1), reverse=True)
        freqWordFile.writefln(entry.vocab ~ " " ~ entry.freq.to!string);

    freqWordFile.flush();

    writefln("corpus=%s, words=%s, K=%s, a=%s, b=%s, nproc=%s",corpus.length,voca.vocas.length,
        options.numTopics, options.alpha, options.beta, options.numProcessors);

    //  cProfile.runctx('ldaLearning (lda, options.iteration, voca)', globals(), locals(), 'lda.profile');
    ldaLearning(lda, options.iteration, voca);

    //  save the model foreach( potential re-use later
    lda.tlock = None;
    pickle.dump((voca, lda), File(options.outputDir  ~ "/model.dat", "w"));
}
