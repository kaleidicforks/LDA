// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _OPENSOURCE_GLDA_ACCUMULATIVE_MODEL_H__
#define _OPENSOURCE_GLDA_ACCUMULATIVE_MODEL_H__

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include "common.hh"
#include "model.hh"
using std::pair;

namespace learning_lda {

// LDAAccumulativeModel is used by LDASampler together with LDAModel.
// Each Gibbs sampling iteration after the burn-in period should (1)
// update the LDAModel object model_, and (3) accumulate model_ into
// the LDAAccumulativeModel object accumulated_model_.  After the last
// iteration of Gibbs sampling, we should average accumulative_model_
// by the number of iterations after the burn-in period.
class LDAAccumulativeModel {
 public:
  LDAAccumulativeModel(int num_topics, int vocab_size);
  ~LDAAccumulativeModel() {}

  // Accumulate a model into accumulative_topic_distributions_ and
  // accumulative_global_distributions_.
  void AccumulateModel(const LDAModel& model);

  // Divide accumulative_topic_distributions_ and
  // accumulative_global_distributions_ by num_estiamte_iterations.
  void AverageModel(int num_estiamte_iterations);

  // Returns the topic distribution for word.
  const TopicProbDistribution& GetWordTopicDistribution(
      int word) const;

  // Returns the global topic distribution.
  const TopicProbDistribution& GetGlobalTopicDistribution() const;

  // Returns the number of topics in the model.
  int num_topics() const { return global_distribution_.size(); }

  // Returns the number of words in the model (not including the global word).
  int num_words() const { return topic_distributions_.size(); }

  // Output accumulative_topic_distributions_ in human-readable
  // format.
  void AppendAsString(const map<string, int>& word_index_map,
                      std::ostream& out) const;

  //added by nanjunxiao
  void SaveModelTWords(const map<string, int>& word_index_map, const string& twords_filename);

 private:
  // Increments the topic count for a particular word (or decrements, for
  // negative values of count).  Creates the word distribution if it doesn't
  // exist, even if the count is 0.
  void IncrementTopic(int word,
                      int topic,
                      int64 count);

  // If users query a word for its topic distribution via
  // GetWordTopicDistribution, but this word does not appear in the
  // training corpus, GetWordTopicDistribution returns
  // zero_distribution_.
  TopicProbDistribution zero_distribution_;

  // The summation of P(word|topic) matrices and P(topic) vectors
  // estimated by Gibbs sampling iterations after the burn-in period.
  vector<TopicProbDistribution> topic_distributions_;
  TopicProbDistribution global_distribution_;
  
  //added by nanjunxiao
  bool static cmp_func(const pair<double,int>& one, const pair<double,int>& two)
  {
  return one.first > two.first; 
  }
};

  //added by nanjunxiao
  const int twordsnum = 15;

}  // namespace learning_lda

#endif  // _OPENSOURCE_GLDA_ACCUMULATIVE_MODEL_H__
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "accumulative_model.hh"

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <string>

#include <cstdio>

namespace learning_lda {

  LDAAccumulativeModel::LDAAccumulativeModel(int num_topics, int vocab_size) {
    CHECK_LT(1, num_topics);
    CHECK_LT(1, vocab_size);
    global_distribution_.resize(num_topics, 0);
    zero_distribution_.resize(num_topics, 0);
    topic_distributions_.resize(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
      topic_distributions_[i].resize(num_topics, 0);
    }
  }

  // Accumulate a model into accumulative_topic_distributions_ and
  // accumulative_global_distributions_.
  void LDAAccumulativeModel::AccumulateModel(const LDAModel& source_model) {
    CHECK_EQ(num_topics(), source_model.num_topics());
    for (LDAModel::Iterator iter(&source_model); !iter.Done(); iter.Next()) {
      const TopicCountDistribution& source_dist = iter.Distribution();
      TopicProbDistribution* dest_dist = &(topic_distributions_[iter.Word()]);
      CHECK_EQ(num_topics(), source_dist.size());
      for (int k = 0; k < num_topics(); ++k) {
        (*dest_dist)[k] += static_cast<double>(source_dist[k]);
      }
    }

    for (int k = 0; k < num_topics(); ++k) {
      global_distribution_[k] +=
        static_cast<double>(source_model.GetGlobalTopicDistribution()[k]);
    }
  }

  void LDAAccumulativeModel::AverageModel(int num_accumulations) {
    for (vector<TopicProbDistribution>::iterator iter =
           topic_distributions_.begin();
         iter != topic_distributions_.end();
         ++iter) {
      TopicProbDistribution& dist = *iter;
      for (int k = 0; k < num_topics(); ++k) {
        dist[k] /= num_accumulations;
      }
    }
    for (int k = 0; k < num_topics(); ++k) {
      global_distribution_[k] /= num_accumulations;
    }
  }

  const TopicProbDistribution&
  LDAAccumulativeModel::GetWordTopicDistribution(int word) const {
    return topic_distributions_[word];
  }

  const TopicProbDistribution&
  LDAAccumulativeModel::GetGlobalTopicDistribution() const {
    return global_distribution_;
  }

  void
  LDAAccumulativeModel::AppendAsString(const map<string, int>& word_index_map,
                                       std::ostream& out) const {
    vector<string> index_word_map(word_index_map.size());
    for (map<string, int>::const_iterator iter = word_index_map.begin();
         iter != word_index_map.end(); ++iter) {
      index_word_map[iter->second] = iter->first;
    }
    for (int i = 0; i < topic_distributions_.size(); ++i) {
      out << index_word_map[i] << "\t";
      for (int topic = 0; topic < num_topics(); ++topic) {
        out << topic_distributions_[i][topic]
            << ((topic < num_topics() - 1) ? " " : "\n");
      }
    }
  }

  //added by nanjunxiao
  void LDAAccumulativeModel::SaveModelTWords(const map<string,int>& word_index_map, const string& twords_filename)
  {
 	FILE* fout = fopen(twords_filename.c_str(),"w"); 
	if(!fout)
	{
		printf("Cannot open file %s to save!\n", twords_filename.c_str() );	
		return;
	}
    	vector<string> index_word_map(word_index_map.size());
    	for (map<string, int>::const_iterator iter = word_index_map.begin();
         iter != word_index_map.end(); ++iter) {
      	index_word_map[iter->second] = iter->first;
    }

	for(int topic=0; topic<num_topics(); ++topic)
	{
		vector<pair<double,int> > v_pair_num2index;
		for(int w=0; w<topic_distributions_.size(); ++w)
		{
			v_pair_num2index.push_back(std::make_pair(topic_distributions_[w][topic], w) );	
		}
		//sort
		std::sort(v_pair_num2index.begin(), v_pair_num2index.end(), cmp_func);
		fprintf(fout, "Topic %dth:\n", topic);
		int printwordsnum = twordsnum <= topic_distributions_.size() ? twordsnum : topic_distributions_.size() ;
		for(int i=0; i<printwordsnum; ++i)
		{
			fprintf(fout, "\t%s    %f\n", index_word_map[v_pair_num2index[i].second].c_str(), v_pair_num2index[i].first);	
		}
	}

	fclose(fout);
  }

}  // namespace learning_lda
