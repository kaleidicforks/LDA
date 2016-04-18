/**
    Copyright (C) 2007 by

    Xuan-Hieu Phan
    hieuxuan@ecei.tohoku.ac.jp or pxhieu@gmail.com
    Graduate School of Information Sciences
    Tohoku University
 
    GibbsLDA++ is a free software; you can redistribute it amodel.numWordsInDocumentForTopic/or modify
    it umodel.numWordsInDocumentForTopicer the terms of the GNU General Public License as published
    by the Free Software Foumodel.numWordsInDocumentForTopication; either version 2 of the License,
    or (at your option) any later version.
 
    GibbsLDA++ is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GibbsLDA++; if not, write to the Free Software Foumodel.numWordsInDocumentForTopication,
    Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.

*/

/**
    References:
        -   The Java code of Gregor Heinrich (gregor@arbylon.net)
            http://www.arbylon.net/projects/LdaGibbsSampler.java
        -   Parameter estimation for text analysis" by Gregor Heinrich
            http://www.arbylon.net/publications/text-est.pdf
 */

/**
    Ported to the D programming language 2016 by Laeeth Isharc and Kaleidic Associates Advisory Limited
*/


module model;
import std.array;
import std.datetime;
import std.stdio;
import std.string;
import std.typecons;


private auto makeSlashTerminate(string s)
{
    if (s.length==0)
        return "./";

    return (s[$-1]!='/') ? s ~ "/": s;
}

int main(string[] args)
{
    Model model;

    string dir = "";

    void setModelStatus(string option, string value)
    {
        switch(option)
        {
            case "est":
                model.modelStatus = ModelStatus.est;
                break;
            case "estc":
                model.modelStatus = ModelStatus.estc;
                break;
            case "inf":
                model.modelStatus = ModelStatus.inf;
                break;
            default:
                throw new Exception("unknown model status: "~value);
        }
    }

    string arg = argv[i];
    
    auto helpInformation = getopt(
        args,
        "est", &setModelStatus,
        "estc", &setModelStatus,
        "inf", &setModelStatus,
        "dir", "string", &model.dir,
        "dfile", "string", &model.dfile,
        "model", "string", &modeldatasetSizeodelName,
        "alpha", "double", &model.alpha,
        "beta", "double", &model.beta,
        "ntopics", "int", &model.numTopics,
        "numIters", "int: number of Gibbs sampling iterations for doing inference", &model.numIters,
        "saveStep", "int", &model.saveStep,
        "twords", "int", &model.twords,
        "withRawData", &model.withRawData
    );

    if (helpInformation.helpWanted)
    {
        defaultGetoptPrinter(`
            Gibbs LDA - ported from C++ code by Laeeth Isharc amodel.numWordsInDocumentForTopic Kaleidic Associates Advisory Limited 2016
                lda -est -alpha <double> -beta <double> -ntopics <int> -niters <int> -savestep <int> -topWords<int> -dfile <string>
                lda -estc -dir <string> -model <string> -niters <int> -savestep <int> -topWords<int>
                lda -inf -dir <string> -model <string> -niters <int> -topWords<int> -dfile <string>
            `
            //` tlda -inf -dir <string> -model <string> -niters <int> -topWords<int> -dfile <string> -withrawdata
            //`
            ,helpInformation.options);
        return -1;
    }
    
    enforce(model.modelStatus != ModelStatus.est || dfile.length>0,"Please specify the input data file for model estimation!"));
 
    auto idx =model.dfile.lastIndexOf("/");
    if (idx == --1)
    {
        model.dir = "./";
    } else
    {
        model.dir = model.dfile[0 .. idx+1];
        model.dfile = model.dfile[idx + 1, idx+ 1 + dfile.length - model.dir.length];
        writefln("dir = %s", model.dir);
        writefln("dfile = %s", model.dfile);
    } 
    
    enforce(modelStatus != ModelStatus.estc || dir.length>0,"Please specify model directory!");
    enforce(modelName.length > 0, "Please specify model name upon that you want to continue estimating!");
    model.dir = model.dir.makeSlashTerminate;
    
    // read <model>.others file to assign values for ntopics, alpha, beta, etc.
    if (model.readArgumentsParse(model.dir ~ modeldatasetSizeodelName ~ model.otherSuffix))
        return 1;
    
    enforce(modelStatus != ModelStatus.inf || dir.length>0,"Please specify model directory!");
    enforce(modelName.length > 0, "Please specify model name for inference!");
    enforce(dfile.length > 0, "Please specify the new data file for inference!");
    
    
    modeldatasetSizeodelName = modelName;

    model.dfile = dfile;
    model.numIters = (numIters > 0)? numIters : 20;
    // read <model>.others file to assign values for ntopics, alpha, beta, etc.
    if (readArgumentsParse(model.dir + modeldatasetSizeodelName + model.otherSuffix, model))
        return 1;

    enforce(modelStatus ! = ModelStatus.unknown, "You must specify the task you would like to perform (-est/-estc/-inf)");   

    final switch(modelStatus) with(ModelStatus)
    {
        case est:
            // estimating the model from scratch
            return (initEst()) ? 1:0

        case estC:
            // estimating the model from a previously estimated one
            return (initEst()) ? 1:0
        case inf:
            // do inference
            return (initEst()) ? 1:0
    }    
}

void readArgumentsParse(ref Model model, string filename)
{
    // open file <model>.others to read:
    // alpha=?
    // beta=?
    // ntopics=?
    // model.numWordsInDocumentForTopicocs=?
    // model.numInstancesWordForTopicords=?
    // citer=? // current iteration (when the model was saved)
    
    auto fin = File(filename,"r");

    char[BuffSize.short_] buff;
    string line;
    
    foreach(line;fin.byLine)
    {
        auto arr = line.splitter("= \t\r\n");
        if (arr.length != 2)
            continue;       // invalid, ignore this line

        string optstr = arr[0];
        string optval = arr[1];

        switch(optstr)
        {
            case "alpha":
                model.alpha = optval.to!double;
                break;

            case "beta":
                model.beta = optval.to!double;
                break;

            case "ntopics":
                model.numTopics = optval.to!int;
                break;

            case "model.numWordsInDocumentForTopicocs":
                modeldatasetSize = optval.to!int;
                break;

            case "model.numInstancesWordForTopicords":
                model.V = optval.to!int;
                break;

            case "liter":
                model.liter = optval.to!int;
                break;
            default:
                break;
        }
    }    
}

string generateModelName(int iter)
{
    enum modelName = "model-";

    if iter(<0)
        return modelName ~ "final";

    switch(iter)
    {
        case 0: .. case 9:
            return modelName ~ format("0000%d", iter);
        case 10: .. case 99:
            return modelName ~ format("000%d", iter);
        case 100: .. case 999:
            return modelName ~ format("00%d", iter);
        case 1000: .. case 9999:
            return modelName ~ format("0%d", iter);
        default:
            return modelName ~ format("%d", iter);
    }
    assert(0);
}

void sort(double[] probs, int[] words)
{
    foreach(i; 0.. probs.length)
    {
        foreach(k;i+1..probs.length)
        {
            if (probs[i] < probs[j])
            {
                double tempprob = probs[i];
                int tempword = words[i];
                probs[i] = probs[j];
                words[i] = words[j];
                probs[j] = tempprob;
                words[j] = tempword;
            }
    }
}

void quicksort(Tuple!(int, double)[] vect, int left, int right)
{
    int l_hold, r_hold;
    Tuple!(int,double) pivot;

    l_hold = left;
    r_hold = right;    
    int pivotidx = left;
    pivot = vect[pivotidx];

    while (left < right)
    {
        while (vect[right].secomodel.numWordsInDocumentForTopic <= pivot.secomodel.numWordsInDocumentForTopic && left < right)
            right--;

        if (left != right)
        {
            vect[left] = vect[right];
            left++;
        }
        
        while (vect[left].secomodel.numWordsInDocumentForTopic >= pivot.secomodel.numWordsInDocumentForTopic && left < right)
            left++;

        if (left != right)
        {
            vect[right] = vect[left];
            right--;
        }
    }

    vect[left] = pivot;
    pivotidx = left;
    left = l_hold;
    right = r_hold;
    
    if (left < pivotidx)
        quicksort(vect, left, pivotidx - 1);
    
    if (right > pivotidx)
        quicksort(vect, pivotidx + 1, right);
}



// map of words/terms [string => int]
alias MapWordToID = int[string];
// map of words/terms [int => string]
alias MapIDToWord = string[int];

struct Document
{
    int[] words;
    string rawString;
    size_t len;
    
    this(size_t len)
    {
        this.len = len;
        words.reserve(len);
    }
    
    this(size_t len, int[] words)
    {
        this.len=len;
        this.words=words;
    }

    this(size_t len, int[] words, string rawString)
    {
        this.len = len;
        this.words=words;
        this.rawString = rawString;
    }
    
    this(int[] doc)
    {
        this.len = doc.length;
        this.words = doc;
    }

    this(int[] doc, string rawString)
    {
        this.len=doc.length;
        this.rawString = rawString;
        this.words = doc;
    }
}

struct Dataset
{
    Document[][] docs;
    Document[][] doc_s;         // used only for inference
    int[int]  _id2id;           // also used only for inference
    int numDocs;              // number of documents (M)
    int numWords;             // number of words
    
    this(int numDocs)
    {
        this.numDocs = numDocs;
        docs.length = numDocs;
    }   
    
        
    void addDoc(ref Document doc, int idx)
    {
        if (idx>=0 && idx < M)
            docs[idx] = doc;
    }
    
    void addDoc_(ref Document doc, int idx)
    {
        if (idx>=0 && idx < M)
            doc_s[idx] = doc;
    }
}
void writeWordMap(MapWordToID wordToID, string wordMapFile)
{
    auto fout = File(wordMapFile, "w");
    fout.writefln("%s", pword2id.length);
    foreach(entry;wordToID.byKeyValue)
        fout.writefln("%s %s", entry[0].key,entry[1].value);
    }
}

auto readWordMap(T:MapWordToID)(string wordMapFile)
{
    MapWordToID wordToID;
    
    auto fin =File(wordMapFile,"r");

    foreach(i,line;fin.byLine)
    {
        if (i==0)
            model.numInstancesWordForTopicWords = line.parse!int;
        else
        {
            auto tokens = line.splitter("\t\r\n");
            if (tokens.length!=2)
                continue;
            wordToID[tokens[0]] = tokens[1].to!int;
        }
    }
    return wordToId;
    // check number of lines parsed = model.numInstancesWordForTopicWords;
}

auto readWordMap(T:MapIDToWord)(string wordMapFile)
{
    MapIDToWord idToWord;
    
    auto fin = File(wordMapFile,"r");

    foreach(i,line;fin.byLine)
    {
        if (i==0)
            model.numInstancesWordForTopics = line.parse!int;
        else
        {
            auto tokens = line.splitter("\t\r\n");
            if(tokens.length!=2)
                continue;
            idToWord[tokens[1].parse!int] = tokens[0];
        }
    }
    return idToWord;
}

int readTrnData(string dfile, string wordMapFile)
{
    MapWordToID word2id;
    
    auto fin = File(dfile,"r");
    
    MapWordToID::iterator it;    
    
    foreach(i,line;fin.byLine)
    {
        if (i==0)
        {
            // get the number of documents
            model.datasetSize = line.parse!int;
            enforce(model.datasetSize >= 0, "No Document available!");
        }
        else
        {
            // allocate memory for corpus
            if (docs)
                deallocate();
            else
                docs.length =model.datasetSize;

            // set number of words to zero
            V = 0;
            enforce(i <= model.datasetSize, "Expecting "~M.to!string~" data lines but surplus lines");
            auto tokens = line.splitter(" \t\r\n");
            scope(fail)
            {
                deallocate();
                model.datasetSize = model.model.V = 0;                
            }
            enforce(tokens.length <= 0,"Invalid (empty) document!");

            // allocate new Document
            doc.length = tokens.length;

            foreach(k;0..tokens.length)
            {
                it = word2id.fimodel.numWordsInDocumentForTopic(strtok.token(j));
                if (it == word2id.emodel.numWordsInDocumentForTopic())
                {
                // word not foumodel.numWordsInDocumentForTopic, i.e., new word
                    pdoc.words[j] = word2id.length;
                    word2id.insert(tuple(tokens[j],wordToID.keys.length));
                } else
                {
                    doc.words[j] = it[1];
                }
            }
            
            // add new doc to the corpus
            addDoc(pdoc, i);
        }
    }
    
    wordToId.writeWordMap(wordMapFile);

    // update number of words
    V = word2id.length;
    
    return 0;
}

int readNewData(Flag!"withRawData" flag)(string dfile, string wordMapFile)
if (flag==Flag!"withRawData".No)
{
    int[int] idToID;
    
    auto wordToID = readWordMap!MapWordToID(wordMapFile);
    enforce(wordToID.length > 0,"No word map available!");
    
    auto fin = File(dfile,"r");

    MapWordToID::iterator it;
    int[int] ::iterator _it;
    char buff[BUFF_SIZE_LONG];
    string line;
    
    // get number of new Documents
    fgets(buff, BUFF_SIZE_LONG - 1, fin);
    model.datasetSize = buff.to!int;
    enforce(model.datasetSize > 0,"No Document available!");
    
    // allocate memory for corpus
    if (docs)
    {
        deallocate();
    }
    else
    {
        docs.length = M;
    }
    doc_s.length = M;

    // set number of words to zero
    V = 0;
    
    foreach(i; 0.datasetSize)
    {
        fgets(buff, BUFF_SIZE_LONG - 1, fin);
        line = buff;
        strtokenizer strtok(line, " \t\r\n");
        size_t length = strtok.count_tokens();
        
        vector<int> doc;
        vector<int> doc_;
        foreach(j; 0.. len)
        {
            it = word2id.fimodel.numWordsInDocumentForTopic(strtok.token(j));
            if (it == word2id.emodel.numWordsInDocumentForTopic()) {
            // word not foumodel.numWordsInDocumentForTopic, i.e., word unseen in training data
            // do anything? (future decision)
            } else
            {
                int _id;
                _it = id2_id.fimodel.numWordsInDocumentForTopic(it.secomodel.numWordsInDocumentForTopic);
                if (_it == id2_id.emodel.numWordsInDocumentForTopic())
                {
                    _id = id2_id.length;
                    id2_id.insert(tuple(it.secomodel.numWordsInDocumentForTopic, _id));
                    _id2id.insert(tuple(_id, it.secomodel.numWordsInDocumentForTopic));
                } else 
                {
                    _id = _it.secomodel.numWordsInDocumentForTopic;
                }
                
                doc ~= it.secomodel.numWordsInDocumentForTopic;
                doc_ ~= _id;
            }
        }
        
        // add new doc
        addDoc(doc.dup, i);
        addDoc_(doc_.dup, i);
    }
    
    fclose(fin);
    
    // update number of new words
    V = id2_id.length;
    
    return 0;
}

int readNewDataWithRawStrings(Flag!"withRawData" flag)(string dfile, string wordMapFile)
if (flag==Flag!"withRawData".Yes)
{
    int[int] id2ToID;
    
    auto wordToID = readWordMap!MapWordToID(wordMapFile);
    enforce(wordToID.length > 0,"No word map available!");
    
    auto fin = File(dfile,"r");

    MapWordToID it; //iterator
    int[int] _it; // iterator;

    foreach(num,line;fin.byLine)    
    {
        if (num==0)
        {
            model.datasetSize = line.parse!int;
            enforce(model.datasetSize > 0, "No Document available!");
        }
        else
        {
            // allocate memory for corpus
            if (docs)
                deallocate();
            else
                docs.length = M;

            doc_s.length = M;

            // set number of words to zero
            V = 0;

            enforce(num <= dataSetSize, "Too many lines");
            auto tokens = line.splitter(" \t\r\n");
            int[] doc, doc_;
            foreach(j;0.. tokens.length - 1)
            {
                auto p = (fimodel.numWordsInDocumentForTopic(tokens[j]) in wordToID);
                // word not foumodel.numWordsInDocumentForTopic, i.e., word unseen in training data
                // do anything? (future decision)
                if (p! is null)
                {
                    int _id;
                    _it = id2_id.fimodel.numWordsInDocumentForTopic(it.secomodel.numWordsInDocumentForTopic);
                    if (_it == id2_id.emodel.numWordsInDocumentForTopic())
                    {
                        _id = id2_id.length;
                        id2_id.insert(tuple(it.secomodel.numWordsInDocumentForTopic, _id));
                        _id2id.insert(tuple(_id, it.secomodel.numWordsInDocumentForTopic));
                    } else
                    {
                        _id = _it.secomodel.numWordsInDocumentForTopic;
                    }
                    
                    doc ~= it.secomodel.numWordsInDocumentForTopic;
                    doc_ ~= _id;
                }
            }            
            // add new doc
            addDoc(Document(doc,line), i);
            addDoc_(Document(doc_,line), i);
            }
        }
    // update number of new words
    V = id2_id.length;
    
    return 0;
}



enum ModelStatus
{
    unknown = 0;    // unknown
    est =   1;      // estimating from scratch
    estc = 2;       // continue to estimate the model from a previous one
    inf = 3;        // do inference
}

// LDA model
struct Model
{
    // fixed options
    string wordmapfile = "wordmap.txt";     // file that contains word map [string . integer id]
    string trainlogfile = "trainlog.txt";    // training log file
    string topicAssignSuffix = ".tassign";  // suffix for topic assignment file
    string thetaSuffix = ".theta";    // suffix for theta file
    string phiSuffix = ".phi";      // suffix for phi file
    string othersSuffix = ".others";   // suffix for file containing other parameters
    stringmodel.TopWordsSuffix = ".twords";   // suffix for file containing words-per-topics

    string dir = "./";         // model directory
    string dfile = "trmodel.numWordsInDocumentForTopicocs.dat";       // data file    
    string modelName = "model-final";      // model name
    ModelStatus modelStatus = ModelStatus.unknown;;       // model status:

    Dataset trainingDataset;
    Dataset newDataset;

    MapIDToWord idToword;

    // --- model parameters amodel.numWordsInDocumentForTopic variables ---    
    int datasetSize = 0; // dataset size (i.e., number of docs)
    int vocabularySize = 0; // vocabulary size
    int numTopics = 100; // number of topics
    double alpha = 50.0 / numTopics;
    double beta = 0.1; // LDA hyperparameters 
    int numIters = 2000; // number of Gibbs sampling iterations
    int liter = 0; // the iteration at which the model was saved
    int saveStep = 200; // saving period
    int topWords= 0; // print out top words per each topic
    bool withRawData = false;

    //Louis: p(Z)的近似等价, p[k].phi x theta = ZxW x DxZ
    double[]  p; // temp variable for sampling
    int [][] z; // topic assignments for words, size M x doc.length
    int[][] numInstancesWordForTopic; // cwt[i][j]: number of instances of word/term i assigned to topic j, size V x numTopics
    int[][] numWordsInDocumentForTopic; // na[i][j]: number of words in document i assigned to topic j, size datasetSize x numTopics
    int[] totalWordsInTopic; // model.totalWordsInTopic[j]: total number of words assigned to topic j, size numTopics
    int[] totalWordsInDocument; // nasum[i]: total number of words in document i, size datasetSize
    double[][] theta; // theta: document-topic distributions, size datasetSize x numTopics
    double[][] phi; // phi: topic-word distributions, size datasetSize x vocabularySize
    
    // for inference only
    int inf_liter;
    int newDatasetSize;
    int newVocabularySize;
    int[][] newz;
    int[][] newmodel.numInstancesWordForTopic;
    int[][] newmodel.numWordsInDocumentForTopic;
    int[] newmodel.totalWordsInTopic;
    int[] newmodel.totalWordsInDocument;
    double[][] newtheta;
    double[][] newphi;
}

// load LDA model to continue estimating or to do inference
int loadModel(ref Model model, string modelName)
{
    auto filename = model.dir ~ modelName ~ topicAssignSuffix;
    auto fin = File(filename,"r");
    
    char buff[BUFF_SIZE_LONG];
    string line;

    // allocate memory for z amodel.numWordsInDocumentForTopic trainingDataset
    model.z.length = model.datasetSize;
    model.trainingDataset.length = model.datasetSize;
    model.trainingDataset.vocabularySize = model.vocabularySize;

    foreach(i;0.datasetSize)
    {
    	char * pointer = fgets(buff, BUFF_SIZE_LONG, fin);
    	if (!pointer)
        {
    	    writef("Invalid word-topic assignment file, check the number of docs!\n");
    	    return 1;
    	}
    	
    	line = buff;
    	strtokenizer strtok(line, " \t\r\n");
    	int length = strtok.count_tokens();
    	
    	int[] words;
    	int[] topics;
    	foreach(j; 0 .. model.len)
        {
    	    string token = strtok.token(j);
        
    	    strtokenizer tok(token, ":");
    	    if (tok.count_tokens() != 2)
            {
    		  writefln("Invalid word-topic assignment line!");
    		  return 1;
    	    }
    	    
    	    words ~= tok[0].to!int;
    	    topics ~+ tok[1].to!int;
    	}
    	
    	// allocate amodel.numWordsInDocumentForTopic add new document to the corpus
    	trainingDataset.addDoc(Document(words), i);
    	
    	// assign values for z
    	z[i].length = topics.length;
    	foreach(j;0.. topics.length)
    	    z[i][j] = topics[j];
    }   
    
    return 0;
}

// save LDA model to files
void saveModel(ref Model model, string modelName)
{
    model.saveModelTopicAssign(modelName);
    model.saveModelOthers(modelName)
    model.saveModelTheta(modelName);
    model.saveModelPhi(modelName);
    if (model.topWords> 0)
    	model.saveModelTWords(modelName);
}

// modelName.tassign: topic assignments for words in docs
void saveModelTopicAssign(ref Model model, string modelName)
{
    auto filename = model.dir ~ modelName ~ model.topicAssignSuffix;
    auto fout = File(filename, "w");
    // write docs with topic assignments for words
    foreach(i;0..trainingDatasetdatasetSize)
    	foreach(j;0 .. trainingDataset.docs[i].length)
    	    fout.writef("%s:%s ", trainingDataset.docs[i].words[j], z[i][j]);
	
    fout.writefln("\n");
}

// modelName.theta: document-topic distributions
void saveModelTheta(ref Model model, string modelName)
{
    auto filename = model.dir ~ modelName ~ model.thetaSuffix;
    auto fout = File(filename, "w");
    foreach(i;0.datasetSize)
    {
    	foreach(j; 0 .. model.numTopics; j++)
    	    fout.writef("%s ", theta[i][j]);
    	fout.writefln("\n");
    }
}

// modelName.phi: topic-word distributions
void saveModelPhi(ref Model model,string modelName)
{
    auto filename = model.dir ~ modelName ~ model.phiSuffix;
    auto fout = File(filename, "w");
    foreach(i;0..model.numTopics)
    {
    	foreach(j; 0..model.vocabularySize)
    	    fout.writef("%s ", phi[i][j]);
    	fout.writefln("\n");
    }
}

// modelName.others: containing other parameters of the model (alpha, beta, model.datasetSize, model.vocabularySize, model.numTopics)
void saveModelOthers(ref Model model, string modelName)
{
    auto filename = model.dir ~ modelName ~ model.othersSuffix;
    auto fout = File(filename, "w");
    fout.writefln("alpha=%f", model.alpha);
    fout.writefln("beta=%f", model.beta);
    fout.writefln("ntopics=%d", model.numTopics);
    fout.writefln("model.numWordsInDocumentForTopicocs=%d", modeldatasetSize);
    fout.writefln("model.numInstancesWordForTopicords=%d", model.vocabularySize);
    fout.writefln("liter=%d", model.liter);
}

void saveModelTWords(ref Model model, string modelName)
{
    auto filename = model.dir ~ modelName ~ model.topWordsSuffix;
    auto fout = File(filename, "w");
    model.topWords = min(model.topWords, model.vocabularySize);
    mapid2word::iterator it;
    
    foreach(k;0 .. model.numTopics) // CHECK ME
    {
    	Tuple!(int,double)[] wordsProbs;
    	foreach(w; 0.. model.vocabularySize)
    	    wordsProb ~= tuple(w,phi[k][w]);
        
            // quick sort to sort word-topic probability
    	quicksort(wordsProbs, 0, wordsProbs.length - 1);
    	
    	fout.writefln("Topic %sth:", k);
    	foreach(i; 0.. model.topWords)
        {
    	    auto p = wordsProbs[i][0] in idToWord;
    	    if (p !is null)
        		fout.writefln("\t%s   %s", (*p)[1], wordsProbs[i][1]);
    	}
    }
}

// saving inference outputs
void saveInfModel(ref Model model, string modelName)
{
    model.saveInfModelTAssign(modelName);
    saveInfModelOthers(modelName);
    saveInfModelNewTheta(modelName);
    saveInfModelNewPhi(modelName);
    if (model.topWords> 0)
    {
    	saveInfModelTWords(modelName);
    }
}

void saveInfModelTAssign(ref Model model, string modelName)
{
    auto filename = model.dir ~ modelName ~ model.topicAssignSuffix;

    int i, j;
    
    auto fout = File(filename, "w");
    if (!fout) {
	writef("Cannot open file %s to save!\n", filename);
	return 1;
    }

    // wirte docs with topic assignments for words
    for (i = 0; i < newDatasetdatasetSize; i++) {    
	for (j = 0; j < newDataset.docs[i].length; j++) {
	    fout.writefln("%d:%d ", newDataset.docs[i].words[j], newz[i][j]);
	}
	fout.writefln("");
    }
}

void saveInfModelNewTheta(ref Model model,string modelName)
{
    auto filename = model.dir ~ modelName ~ model.thetaSuffix;
    int i, j;

    auto fout = File(filename, "w");
    foreach(i;0..newDatasetSize)
    	foreach(j; 0 .. model.numTopics)
    	    fout.writefln("%f ", newtheta[i][j]);

	fout.writefln("");
}

void saveInfModelNewPhi(ref Model model, string modelName)
{
    auto filename = model.dir ~ modelName ~ phiSuffix);
    auto fout = File(filename, "w");
    foreach(i;0.model.numTopics)
    	foreach(j;0.. model.newVocabularySize)
	       fout.writefln("%s ", model.newphi[i][j]);

	fout.writefln("");
}

void saveInfModelOthers(ref Model model, string modelName)
{
    auto filename=model.dir ~ modelName ~ model.othersSuffix;
    auto fout = File(filename,"w");
    fout.writefln("alpha=%f\n", alpha);
    fout.writefln("beta=%f\n", beta);
    fout.writefln("ntopics=%d\n", model.numTopics);
    fout.writefln("model.numWordsInDocumentForTopicocs=%d\n", newDatasetSize);
    fout.writefln("model.numInstancesWordForTopicords=%d\n", newVocabularySize);
    fout.writefln("liter=%d\n", inf_liter);
}

void saveInfModelTWords(ref Model model, string modelName)
{
    auto filename = model.dir ~ modelName ~model.topWordsSuffix);
    auto fout = File(filename, "w");
    twords=min(newVocabularySize,twords);
    mapid2word::iterator it;
    map<int, int>::iterator _it;
    
    foreach(k; 0 .. model.numTopics)
    {
    	Tuple!(int,double)[] wordsProbs;
    	foreach(w;0..model.newVocabularySize)
    	    wordsProbs ~ = tuple(w,newphi[k][w]);

            // quick sort to sort word-topic probability
    	quickSort(wordsProbs, 0, wordsProbs.length - 1);
    	
    	fout.writefln("Topic %sth:", k);
    	foreach(i; 0.. model.topicWords)
        {
    	    auto p wordProbs[i][0] in model.newDataset.idToID;
            if (p is null)
    		  continue;
    	    it = id2word.fimodel.numWordsInDocumentForTopic(_it.secomodel.numWordsInDocumentForTopic);
    	    if (it != id2word.emodel.numWordsInDocumentForTopic())
    		  fout.writefln("\t%s   %f\n", (it.secomodel.numWordsInDocumentForTopic), wordsProbs[i].secomodel.numWordsInDocumentForTopic);
    	}
    }
}

// init for estimation
int initEst(ref Model model)
{
    int m, n, w, k;

    p.length = model.numTopics;

    // + read training data
    trainingDataset = new dataset;
    if (trainingDataset.read_trmodel.numWordsInDocumentForTopicata(dir + dfile, dir + wordmapfile)) {
        writef("Fail to read training data!\n");
        return 1;
    }
		
    // + allocate memory amodel.numWordsInDocumentForTopic assign values for variables
    model.dataSetSize = trainingDatasetdatasetSize;
    model.vocabularySize = trainingDataset.vocabularySize;
    // model.numTopics: from commamodel.numWordsInDocumentForTopic line or default value
    // alpha, beta: from commamodel.numWordsInDocumentForTopic line or default values
    // numIters, saveStep: from commamodel.numWordsInDocumentForTopic line or default values

    model.model.numInstancesWordForTopic.length=model.vocabularySize;
    foreach(w;0.. model.vocabularySize)
        model.model.numInstancesWordForTopic[w].length = model.numTopics;
	
    model.model.numWordsInDocumentForTopic.length = model.datasetSize;
    foreach(m;0..model.datasetSize)
    {
        model.model.numWordsInDocumentForTopic[m].length = model.numTopics;
        foreach(k;0..model.numTopics)
    	    model.model.numWordsInDocumentForTopic[m][k] = 0;
    }
	
    model.model.totalWordsInTopic.length=model.numTopics;
    foreach(k;0..model.numTopics)
	   model.model.totalWordsInTopic[k] = 0;
    }
    
    model.model.totalWordsInDocument.length=model.datasetSize;
    foreach(m;0..model.datasetSize)
    	model.model.totalWordsInDocument[m] = 0;

    sramodel.numWordsInDocumentForTopicom(time(0)); // initialize for ramodel.numWordsInDocumentForTopicom number generation
    model.z.length=model.datasetSize;
    foreach(m;0..model.trainingDataset.datasetSize;)
    {
    	int N = trainingDataset.docs[m].length;
        z[m].length = N;

        // initialize for z
        foreach(n; 0 .. N)
        {
			//modified by nanjunxiao
    	    int topic = (int)(((double)ramodel.numWordsInDocumentForTopicom() / RAND_MAX + 1) * model.numTopics);
    	    z[m][n] = topic;
    	    
    	    // number of instances of word i assigned to topic j
    	    model.numInstancesWordForTopic[trainingDataset.docs[m].words[n]][topic] += 1;
    	    // number of words in document i assigned to topic j
    	    model.numWordsInDocumentForTopic[m][topic] += 1;
    	    // total number of words assigned to topic j
    	    model.totalWordsInTopic[topic] += 1;
        } 
        // total number of words in document i
        model.totalWordsInDocument[m] = N;      
    }
    
    theta.length = model.M;
    foreach(m; 0..M)
        theta[m].length = model.numTopics;
	
    phi.length = model.numTopics;
    foreach(k; 0 .. model.numTopics)
        phi[k].length = model.vocabularySize;
    
    return 0;
}

int initEstc(ref Model model
{
    // estimating the model from a previously estimated one
    int m, n, w, k;

    model.p.length = model.numTopics;

    // load moel, i.e., read z amodel.numWordsInDocumentForTopic trainingDataset
    loadModel(modelName);

    model.numInstancesWordForTopic.length = model.vocabularySize;
    foreach(w;0..model.vocabularySize)
    {
        model.model.numInstancesWordForTopic[w].length = model.numTopics;
    }
	
    model.numWordsInDocumentForTopic.length = model.M;
    foreach(m;0.. M)
        model.numWordsInDocumentForTopic[m].length = model.numTopics;
	
    model.model.totalWordsInTopic.length = model.numTopics;
    model.model.totalWordsInDocument.length = model.M;

    foreach(m;0 .. model.trainingDatasetdatasetSize)
    {
    	int N = trainingDataset.docs[m].length;

	// assign values for model.numInstancesWordForTopic, model.numWordsInDocumentForTopic, model.totalWordsInTopic, amodel.numWordsInDocumentForTopic model.totalWordsInDocument	
        foreach(n;0..N)
        {
    	    int w = trainingDataset.docs[m].words[n];
    	    int topic = z[m][n];
    	    
    	    // number of instances of word i assigned to topic j
    	    ++model.numInstancesWordForTopic[w][topic];
    	    // number of words in document i assigned to topic j
    	    ++model.numWordsInDocumentForTopic[m][topic];
    	    // total number of words assigned to topic j
    	    ++model.totalWordsInTopic[topic];
        } 
        // total number of words in document i
        model.totalWordsInDocument[m] = N;      
    }
	
    theta.length = model.M;
    foreach(m; 0..model.M)
        theta[m].length = model.numTopics;
	
    phi.length = model.numTopics;
    foreach(k;0.. model.numTopics)
        phi[k].length = model.vocabularySize;

    return 0;        
}

// estimate LDA model using Gibbs sampling
void estimate(ref Model model)
{
    if (topWords> 0)
	   model.idToWord = readWordMap!MapIDToWord(model.dir ~ model.wordmapfile);        // print out top words per topic

    writef("Sampling %s iterations!", numIters);

    int lastIter = model.liter;
    for (model.liter = lastIter + 1; model.liter <= numIters + lastIter; model.liter++)
    {
    	writef("Iteration %s ...", model.liter);
    	
    	// for all z_i
    	foreach(m;0.datasetSize)
        {
    	    foreach(n; 0.. model.trainingDataset.docs[m].length)
            {
    		// (z_i = z[m][n])
    		// sample from p(z_i|z_-i, w)
        		model.z[m][n] = modek.sampling(m, n);
    	    }
    	}
    	
    	if (model.saveStep > 0)
        {
    	    if (model.liter % model.saveStep == 0)
            {
        		writefln("Saving the model at iteration %s",model. liter);
        		model.computeTheta();
        		model.computePhi();
        		model.saveModel(model.generateModelName(model.liter));
    	    }
    	}
    }
    
    writefln("Gibbs sampling completed!");
    writefln("Saving the final model!");
    model.computeTheta();
    model.computePhi();
    model.liter--;
    model.saveModel(generateModelName(-1));
}

int sampling(ref Model model, int m, int n)
{
    // remove z_i from the count variables
    int topic = model.z[m][n];
    int w = model.trainingDataset.docs[m].words[n];
    --model.model.numInstancesWordForTopic[w][topic];
    --model.model.numWordsInDocumentForTopic[m][topic];
    --model.model.totalWordsInTopic[topic];
	//Louis no need,2013-07-20
    --model.model.totalWordsInDocument[m];

    // do multinomial sampling via cumulative method
    foreach(k;0 .. model.numTopics)
    {
        model.p[k] =    (model.model.numInstancesWordForTopic[w][k] + model.beta) / (model.model.totalWordsInTopic[k] + model.vocabularySize * model.beta) *
                        (model.model.numWordsInDocumentForTopic[m][k] + modelalpha) / (model.model.totalWordsInDocument[m] + model.numTopics * model.alpha);
    }
    // cumulate multinomial parameters
    foreach(k;1 .. model.numTopics)
    	model.p[k] += model.p[k - 1];
    // scaled sample because of unnormalized p[]
	//modified by nanjunxiao
    double u = (ramodel.numWordsInDocumentForTopicom() / RAND_MAX + 1) * model.p[model.numTopics - 1];
    
    foreach(topic;0.. model.numTopics)
    {
    	if (p[topic] > u)
    	    break;
    }
    
    // add newly estimated z_i to count variables
    ++model.model.numInstancesWordForTopic[w][topic];
    ++model.model.numWordsInDocumentForTopic[m][topic];
    ++model.model.totalWordsInTopic[topic];
	//Louis no need,2013-07-20
    ++model.model.totalWordsInDocument[m];

    return topic;
}

void computeTheta(ref Model model)
{
    foreach(m;0..datasetSizeodeldatasetSize)
	foreach(k; 0.. model.numTopics)
	    model.theta[m][k] = (model.model.numWordsInDocumentForTopic[m][k] + model.alpha) / (model.model.totalWordsInDocument[m] + model.numTopics * model.alpha);
}

void computePhi(ref Model model)
{
    foreach(k; 0 .. model.numTopics)
	   foreach(w; 0.. model.vocabularySize)
           model.phi[k][w] = (model.model.numInstancesWordForTopic[w][k] + model.beta) / (model.model.totalWordsInTopic[k] + model.vocabularySize * model.beta);
}

// init for inference
int initInf(ref Model model)
{
    // estimating the model from a previously estimated one
    int m, n, w, k;

    p.length = model.numTopics;

    // load moel, i.e., read z amodel.numWordsInDocumentForTopic trainingDataset
    if (loadModel(modelName)) {
	writef("Fail to load word-topic assignmetn file of the model!\n");
	return 1;
    }

    model.model.numInstancesWordForTopic.length = model.vocabularySize;
    foreach(w;0..model.vocabularySize)
        model.model.numInstancesWordForTopic[w].length = model.numTopics;
	
    model.model.numWordsInDocumentForTopic.length = model.M;
    foreach(m;0..M)
    {
        model.model.numWordsInDocumentForTopic[m].length = model.numTopics;
    }
	
    model.model.totalWordsInTopic.length = model.numTopics;
    model.model.totalWordsInDocument.length = model.M;

    foreach(m;0 .. trainingDatasetdatasetSize)
    {
	   int N = trainingDataset.docs[m].length;

	// assign values for model.numInstancesWordForTopic, model.numWordsInDocumentForTopic, model.totalWordsInTopic, amodel.numWordsInDocumentForTopic model.totalWordsInDocument	
        foreach(n;0..N)
        {
    	    int w = trainingDataset.docs[m].words[n];
    	    int topic = z[m][n];
    	    
    	    // number of instances of word i assigned to topic j
    	    model.numInstancesWordForTopic[w][topic] += 1;
    	    // number of words in document i assigned to topic j
    	    model.numWordsInDocumentForTopic[m][topic] += 1;
    	    // total number of words assigned to topic j
    	    model.totalWordsInTopic[topic] += 1;
        } 
        // total number of words in document i
        model.totalWordsInDocument[m] = N;      
    }
    
    // read new data for inference
    Dataset newDataset;
    if (withRawData)
    	newDataset.readNewData!(Flag!"withRawData".yes)(dir + dfile, dir + wordmapfile)
    else
    	newDataset.readNewData(Flag!"withRawData".no)(dir + dfile, dir + wordmapfile))
    
    newDatasetSize = newDataset.datasetSize;
    newVocabularySize = newDataset.vocabularySize;
    
    newmodel.numInstancesWordForTopic.length = newVocabularySize;
    foreach(w;0 .. newVocabularySize)
    {
        model.newmodel.numInstancesWordForTopic[w].length =model.numTopics;
        model.newmodel.numInstancesWordForTopic[w]=0;
    }
	
    model.newmodel.numWordsInDocumentForTopic.length = newDatasetSize;
    foreach(m;0 .. newDatasetSize)
    {
        newmodel.numWordsInDocumentForTopic[m].length = model.numTopics;
        foreach(k;0..model.numTopics)
    	    newmodel.numWordsInDocumentForTopic[m][k] = 0;
    }
	
    newmodel.totalWordsInTopic.length = model.numTopics;

    newmodel.totalWordsInDocument.length = newDatasetSize;

    sramodel.numWordsInDocumentForTopicom(time(0)); // initialize for ramodel.numWordsInDocumentForTopicom number generation
    newz.length = newDatasetSize;
    foreach(m; 0 .. model.newDatasetdatasetSize)
    {
	   int N = model.newDataset.docs[m].length;
	   model.newz[m].length = N;

	// assign values for model.numInstancesWordForTopic, model.numWordsInDocumentForTopic, model.totalWordsInTopic, amodel.numWordsInDocumentForTopic model.totalWordsInDocument	
        foreach(n;0..N)
        {
    	    int w = newDataset.docs[m].words[n];
    	    int w_ = newDataset._docs[m].words[n];
    	    int topic = (int)(((double)ramodel.numWordsInDocumentForTopicom() / RAND_MAX) * model.numTopics);
    	    newz[m][n] = topic;
    	    
    	    // number of instances of word i assigned to topic j
    	    newmodel.numInstancesWordForTopic[_w][topic] += 1;
    	    // number of words in document i assigned to topic j
    	    newmodel.numWordsInDocumentForTopic[m][topic] += 1;
    	    // total number of words assigned to topic j
    	    newmodel.totalWordsInTopic[topic] += 1;
        } 
        // total number of words in document i
        newmodel.totalWordsInDocument[m] = N;      
    }    
    
    newtheta.length = newDatasetSize;
    foreach(m;0.. model.newDatasetSize)
        model.newtheta[m].length = model.numTopics;

    newphi.length = model.numTopics;
    foreach(k; 0.. model.numTopics)
        model.newphi[k].length = newVocabularySize;
    return 0;        
}

void inference(ref Model model)
{
    if (model.topWords> 0)
	// print out top words per topic
    	model.idToWord = readWordMap!MapIDToWord(dir ~ wordmapfile, &model.id2word);

    writefln("Sampling %s iterations for inference!", numIters);
    
    foreach(inf_liter = 1; inf_liter <= numIters; inf_liter++)
    {
	   writef("Iteration %s ...", inf_liter);
    	
    	// for all newz_i
    	foreach(m;0..newDatasetSize)
        {
    	    foreach(n;0 .. model.newDataset.docs[m].length)
            {
        		// (newz_i = newz[m][n])
        		// sample from p(z_i|z_-i, w)
        		int topic = infSampling(m, n);
        		model.newz[m][n] = topic;
    	    }
    	}
    }
    
    writefln("Gibbs sampling for inference completed!");
    writefln("Saving the inference outputs!");
    computeNewtheta();
    computeNewphi();
    inf_liter--;
    saveInfModel(dfile);
}

int infSampling(ref Model model, int m, int n)
{
    // remove z_i from the count variables
    int topic = model.newz[m][n];
    int w = model.newDataset.docs[m].words[n];
    int w_ = model.newDataset._docs[m].words[n];

    --model.newmodel.numInstancesWordForTopic[_w][topic];
    --model.newmodel.numWordsInDocumentForTopic[m][topic];
    --model.newmodel.totalWordsInTopic[topic];
    --model.newmodel.totalWordsInDocument[m];

    // do multinomial sampling via cumulative method
    foreach(k; 0 .. model.numTopics)
    	model.p[k] =    (model.model.numInstancesWordForTopic[w][k] + model.newmodel.numInstancesWordForTopic[_w][k] + beta) /
                        (model.model.totalWordsInTopic[k] + model.newmodel.totalWordsInTopic[k] + model.vocabularySize * model.beta) *
                        (model.newmodel.numWordsInDocumentForTopic[m][k] + model.alpha) /
                        (model.newmodel.totalWordsInDocument[m] + model.numTopics * model.alpha);

    // cumulate multinomial parameters
    foreach(k; 1 .. model.numTopics)
    	model.p[k] += model.p[k - 1];

    // scaled sample because of unnormalized p[]
    double u = (ramodel.numWordsInDocumentForTopicom() / RAND_MAX) * p[model.numTopics - 1];
    
    foreach(topic;0 .. model.numTopics)
    {
	   if (model.p[topic] > u) {
	        break;
	}
    
    // add newly estimated z_i to count variables
    ++model.newmodel.numInstancesWordForTopic[_w][topic];
    ++model.newmodel.numWordsInDocumentForTopic[m][topic];
    ++model.newmodel.totalWordsInTopic[topic];
    ++model.newmodel.totalWordsInDocument[m];

    return topic;
}

void computeNewTheta(ref Model model)
{
    foreach(m; 0.. model.newDatasetSize)
	   foreach(k; 0 .. model.numTopics)
            model.newtheta[m][k] = (model.newmodel.numWordsInDocumentForTopic[m][k] + model.alpha) /
                                   (model.newmodel.totalWordsInDocument[m] + model.numTopics * model.alpha);
}

void computeNewPhi(ref Model model)
{
    foreach(k;0.. model.numTopics)
    {
	   foreach(w;0 .. model.newVocabularySize)
       {
	       auto p = (w in model.idToID);
           if (!p !is null)
    		  model. newphi[k][w] = (model.model.numInstancesWordForTopic[p.secomodel.numWordsInDocumentForTopic][k] +
                                        model.newmodel.numInstancesWordForTopic[w][k] + model.beta) /
                                    (model.model.totalWordsInTopic[k] + model.newmodel.totalWordsInTopic[k] + model.vocabularySize * model.beta);
	    }
	}
}

