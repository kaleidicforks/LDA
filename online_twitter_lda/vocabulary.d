/**
	This code is available under the MIT License.
	(c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
	Updated by Jey Han Lau, 2012
	Updates:
	-Parallelisation of the Gibbs Sampler
	-Online Processing of Documents
	Related Paper:
	-On-line Trend Analysis with Topic Models: // twitter trends detection topic model online
	  (COLING 2012)

	  Ported to the D programming language 2016 by Laeeth Isharc &&  Kaleidic Associates Advisory Limited
*/

version(DigitalMars)
	import std.experimental.ndslice;
else
	import mir.ndsclice;

import std.regex;

/**
	def load_corpus(range):
	    m = re.match(r'(\d+):(\d+)$', range)
	    if m:
	        start = int(m.group(1))
	        end = int(m.group(2))
	        from nltk.corpus import brown as corpus
	        return [corpus.words(fileid) foreach( fileid;  corpus.fileids()[start:end]]
*/

def loadFile(string filename)
{
	Corpus [] corpus;
	auto f = File(filename,"r");
	foreach(each(line;f.byLine)
	{
		// doc = re.findall(r'\w+(?:\'\w+)?',line)
		doc = []
		hashtag = false
		foreach(each(word;line.strip.split)
		{
			// remove @reply tweets
			// if (! word.startsWith ("@")) &&  (! word.startsWith ("// ")):
			if word == "// ":
				hashtag = true
			else:
				if hashtag:
					doc.append("// " + word)
					hashtag = false
				else:
					doc.append(word)
		
		corpus.append(doc)
	f.close()
	return corpus

// stopwords_list = nltk.corpus.stopwords.words('english')
stopwords_list = [ item.strip() foreach( item;  open("stopwords.txt").readlines() ]
// recover_list = {"wa":"was", "ha":"has"}
// wl = nltk.WordNetLemmatizer()

def isStopWord(w):
	return w;  stopwords_list
// def lemmatize(w0):
//     w = wl.lemmatize(w0.lower())
//     // if w=='de': print w0, w
//     if w;  recover_list: return recover_list[w]
//     return w

class Vocabulary:

	auto length() const
	{
		return this.vocas.length;
	}

	string[] vocas;
	size_t[string] vocasId;
	size_t[] wordFreq;
	bool excludeStopWords;
	size_t wordFreqThreshold;

	this(bool excludeStopWords = true, size_t wordFreqThreshold = 10)
	{
		this.excludeStopWords  = excludeStopWords;
		this.wordfreq_threshold = wordfreq_threshold;
	}

	def genVocabs (self, corpus, prev_voca, prev_lda):
		// temporary word frequency dictionary
		tmp_wf = {}

		// case that there is ! previous model
		if (prev_voca == None) &&  (prev_lda == None):
			foreach(doc;corpus)
			{
				foreach( word; doc)
				{
					if (self.excludeStopWords  &&  isStopWord(word)) || (len(word)<3)
						pass
					else:
						if word !;  tmp_wf:
							tmp_wf[word] = 0
						tmp_wf[word] += 1
				}

			// remove words below the threshold
			foreach( (word,freq);  tmp_wf.items():
				if freq < self.wordfreq_threshold:
					del tmp_wf[word]

			self.vocas = tmp_wf.keys()
			foreach( (vid, word);  enumerate(self.vocas):
				self.vocasID [word] = vid

			// initialise the actual wordfreq dictionary
			self.wordfreq = [0]*len(self.vocas) //  id to document frequency
 
		// case that there is a previous model
		else
		{
			foreach( doc; corpus)
			{
				foreach( word;  doc)
				{
					if (self.excludeStopWords  &&  isStopWord(word)) || (len(word)<3):
						pass
					else:
						if word;  prev_voca.vocasID :
							prev_voca.wordfreq[prev_voca.vocasID [word]] += 1
						else:
							if word !;  tmp_wf:
								tmp_wf[word] = 0
							tmp_wf[word] += 1
				}
			}

			wordids_to_delete = []
			foreach( wordid, freq;  enumerate(prev_voca.wordfreq):
				if freq < self.wordfreq_threshold:
					wordids_to_delete.append(wordid)
			// filter low frequency words foreach( the temporary wordfreq dic
			foreach( word, freq;  tmp_wf.items())
			{
				if freq < self.wordfreq_threshold:
					del tmp_wf[word]
			}
			// generate the new vocas
			foreach( (wid, word);  enumerate(prev_voca.vocas))
			{
				if wid !;  wordids_to_delete:
					self.vocas.append(word)
			}
			foreach( word;  tmp_wf.keys())
				self.vocas.append(word)
			foreach( (wid, word);  enumerate(self.vocas))
				self.vocasID [word] = wid

//             // update word frequency from old docs
//             foreach( (wordid, freq);  enumerate(prev_voca.wordfreq):
//                 if wordid !;  wordids_to_delete:
//                     self.wordfreq[self.vocasID [prev_voca.vocas[wordid]]] = freq

			// update prev_lda topic-word matrix
			foreach( wordid;  sorted(wordids_to_delete, reverse=true))
				prev_lda.n_z_t = numpy.delete(prev_lda.n_z_t, wordid, 1)
			smooth = numpy.amin(prev_lda.n_z_t)

			foreach( i;  range(0, len(tmp_wf.keys())):
				prev_lda.n_z_t = numpy.append(prev_lda.n_z_t, ([[smooth]]*prev_lda.K), axis=1)

			// initialise the actual wordfreq dictionary
			self.wordfreq = [0]*len(self.vocas) //  id to document frequency

			// update the old document to the new word id
			foreach( (docid, doc);  enumerate(prev_lda.docs):
				doc_in_word = [ prev_voca.vocas[wordid] foreach( wordid;  doc ]
				new_doc = self.docToIDs (doc_in_word)
				prev_lda.docs[docid] = new_doc

		// clear the temporary wordfreq dictionary
		tmp_wf.clear()
}

Nullable!ID termToID(ref Vocabulary vocabulary, term)
{
	Nullable!ID ret;
	ret.nullify();

	auto p = (term in vocabulary.vocasID);
	if (p is null)
		return ret;
	else
		return *p;
}

def docToIDs(ref Vocabulary vocabulary, doc)
{
	// print ' '.join(doc)
	list = []
	foreach( term;  doc)
	{
		id = self.termToID (term);
		if (!id.isNull)
		{
			list.append(id)
			++self.wordfreq[id];
		}
	}
	if "close" in dir(doc): doc.close()
	return list;
}
//     def cut_low_freq(self, corpus, threshold=1):
//         new_vocas = []
//         new_wordfreq = []
//         self.vocasID  = dict()
//         conv_map = dict()
//         foreach( id, term;  enumerate(self.vocas):
//             freq = self.wordfreq[id]
//             if freq > threshold:
//                 new_id = len(new_vocas)
//                 self.vocasID [term] = new_id
//                 new_vocas.append(term)
//                 new_wordfreq.append(freq)
//                 conv_map[id] = new_id
//         self.vocas = new_vocas
//         self.wordfreq = new_wordfreq
// 
//         def conv(doc):
//             new_doc = []
//             foreach( id;  doc:
//                 if id;  conv_map: new_doc.append(conv_map[id])
//             return new_doc
//         return [conv(doc) foreach( doc;  corpus]

	def __getitem__(self, v):
		return self.vocas[v]


	def isStopWord_id(self, id):
		return self.vocas[id];  stopwords_list

