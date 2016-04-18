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

#ifndef _OPENSOURCE_GLDA_MODEL_H__
#define _OPENSOURCE_GLDA_MODEL_H__

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common.hh"

namespace learning_lda {

// The LDAModel class stores topic-word co-occurrence count vectors as
// well as a vector of global topic occurrence counts.  The global vector is
// the sum of the other vectors.  These vectors are precisely the components of
// an LDA model (as document-topic associations are not true model parameters).
//
// This class supports common operations on this sort of model, primarily in
// the form of assigning new topic occurrences to words, and in reassigning
// word occurrences from one topic to another.
//
// This class is not thread-safe.  Do not share an object of this
// class by multiple threads.
class LDAModel {
 public:
  // An iterator over a LDAModel.  Returns distributions in an arbitrary
  // order.  Altering the parent LDAModel in any way invalidates the
  // iterator, although this is not currently enforced.
  class Iterator {
   public:
    // Initializes the iterator for the model specified by parent.  parent must
    // exist and must not be modified for the lifetime of the iterator.
    explicit Iterator(const LDAModel* parent);

    ~Iterator();

    // Advances to the next word.
    void Next();

    // Returns true if we have finished iterating over the model.
    bool Done() const;

    // Returns the current word.
    int Word() const;

    // Returns the current word's distribution.
    const TopicCountDistribution& Distribution() const;

   private:
    const LDAModel* parent_;
    int iterator_;
  };
  friend class Iterator;

  LDAModel(int num_topic, const map<string, int>& word_index_map);

  // Read word topic distribution and global distribution from iframe.
  // Return a map from word string to index. Intenally we use int to represent
  // each word.
  LDAModel(std::istream& in, map<string, int>* word_index_map);

  ~LDAModel() {}

  // Returns the topic distribution for word.
  const TopicCountDistribution& GetWordTopicDistribution(
      int word) const;

  // Returns the global topic distribution.
  const TopicCountDistribution& GetGlobalTopicDistribution() const;

  // Increments the topic count for a particular word (or decrements, for
  // negative values of count).  Creates the word distribution if it doesn't
  // exist, even if the count is 0.
  void IncrementTopic(int word,
                      int topic,
                      int64 count);

  // Reassigns count occurrences of a word from old_topic to new_topic.
  void ReassignTopic(int word,
                     int old_topic,
                     int new_topic);

  // Change topic distributions and the global distribution to zero vectors.
  void Unset();

  // Returns the number of topics in the model.
  int num_topics() const { return global_distribution_.size(); }

  // Returns the number of words in the model (not including the global word).
  int num_words() const { return topic_distributions_.size(); }

  // Output topic_distributions_ into human readable format.
  void AppendAsString(std::ostream& out) const;


 protected:
  // The dataset which keep all the model memory.
  vector<int64> memory_alloc_;

 private:
  // If users query a word for its topic distribution via
  // GetWordTopicDistribution, but this word does not appear in the
  // training corpus, GetWordTopicDistribution returns
  // zero_distribution_.
  vector<int64> zero_distribution_;


  // topic_distributions_["word"][k] counts the number of times that
  // word "word" and assigned topic k by a Gibbs sampling iteration.
  vector<TopicCountDistribution> topic_distributions_;

  // global_distribution_[k] is the number of words in the training
  // corpus that are assigned by topic k.
  TopicCountDistribution global_distribution_;

  map<string, int> word_index_map_;
};

}  // namespace learning_lda

#endif  // _OPENSOURCE_GLDA_MODEL_H__
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

#include "model.hh"

#include <map>
#include <sstream>
#include <string>

namespace learning_lda {

  // Start by pointing to the beginning of the parent model's topic
  // distribution map.
  LDAModel::Iterator::Iterator(const LDAModel* parent)
    : parent_(parent),
      iterator_(0) { }

  LDAModel::Iterator::~Iterator() { }

  void LDAModel::Iterator::Next() {
    CHECK(!Done());
    ++iterator_;
  }

  bool LDAModel::Iterator::Done() const {
    return iterator_ == parent_->topic_distributions_.size();
  }

  int LDAModel::Iterator::Word() const {
    CHECK(!Done());
    return iterator_;
  }

  // Returns the current word's distribution.
  const TopicCountDistribution& LDAModel::Iterator::Distribution() const {
    CHECK(!Done());
    return parent_->GetWordTopicDistribution(iterator_);
  }

  LDAModel::LDAModel(int num_topics, const map<string, int>& word_index_map) {
    int vocab_size = word_index_map.size();
    memory_alloc_.resize(((int64)(num_topics)) * ((int64) vocab_size + 1), 0);
    // topic_distribution and global_distribution are just accessor pointers
    // and are not responsible for allocating/deleting memory.
    topic_distributions_.resize(vocab_size);
    global_distribution_.Reset(&memory_alloc_[0] +
                               (int64)vocab_size * num_topics,
                               num_topics);
    for (int i = 0; i < vocab_size; ++i) {
      topic_distributions_[i] =
        TopicCountDistribution(&memory_alloc_[0] + num_topics * i,
                               num_topics);
    }
    word_index_map_ = word_index_map;
  }

  const TopicCountDistribution&
  LDAModel::GetWordTopicDistribution(int word) const {
    return topic_distributions_[word];
  }

  const TopicCountDistribution&
  LDAModel::GetGlobalTopicDistribution() const {
    return global_distribution_;
  }

  void LDAModel::IncrementTopic(int word,
                                int topic,
                                int64 count) {
    CHECK_GT(num_topics(), topic);
    CHECK_GT(num_words(), word);

    topic_distributions_[word][topic] += count;
    global_distribution_[topic] += count;

    // We do the following checks only in serial computing version,
    // because thread simultaneous adding may cause count dropping and
    // simultaneous substracting may cause count increasing.  This
    // problem is fixed by periodically update the model given training
    // documents and latent topics.
#ifndef _OPENMP
    CHECK_LE(0, topic_distributions_[word][topic]);
    CHECK_LE(0, global_distribution_[topic]);
#endif // _OPENMP
  }

  void LDAModel::ReassignTopic(int word,
                               int old_topic,
                               int new_topic) {
    IncrementTopic(word, old_topic, -1);
    IncrementTopic(word, new_topic, 1);
  }

  void LDAModel::AppendAsString(std::ostream& out) const {
    vector<string> index_word_map(word_index_map_.size());
    for (map<string, int>::const_iterator iter = word_index_map_.begin();
         iter != word_index_map_.end(); ++iter) {
      index_word_map[iter->second] = iter->first;
    }
    for (LDAModel::Iterator iter(this); !iter.Done(); iter.Next()) {
      out << index_word_map[iter.Word()] << "\t";
      for (int topic = 0; topic < num_topics(); ++topic) {
        out << iter.Distribution()[topic]
            << ((topic < num_topics() - 1) ? " " : "\n");
      }
    }
  }

  LDAModel::LDAModel(std::istream& in, map<string, int>* word_index_map) {
    word_index_map_.clear();
    memory_alloc_.clear();
    string line;
    while (getline(in, line)) {  // Each line is a training document.
      if (line.size() > 0 &&      // Skip empty lines.
          line[0] != '\r' &&      // Skip empty lines.
          line[0] != '\n' &&      // Skip empty lines.
          line[0] != '#') {       // Skip comment lines.
        std::istringstream ss(line);
        string word;
        double count_float;
        CHECK(ss >> word);
        while (ss >> count_float) {
          memory_alloc_.push_back((int64)count_float);
        }
        int size = word_index_map_.size();
        word_index_map_[word] = size;
      }
    }
    int vocab_size = word_index_map_.size();
    int num_topics = memory_alloc_.size() / vocab_size;
    memory_alloc_.resize(((int64)(num_topics)) * ((int64) vocab_size + 1), 0);
    // topic_distribution and global_distribution are just accessor pointers
    // and are not responsible for allocating/deleting memory.
    topic_distributions_.resize(vocab_size);
    global_distribution_.Reset(&memory_alloc_[0] +
                               (int64)vocab_size * num_topics,
                               num_topics);
    for (int i = 0; i < vocab_size; ++i) {
      topic_distributions_[i] =
        TopicCountDistribution(&memory_alloc_[0] + num_topics * i,
                               num_topics);
    }
    for (int i = 0; i < vocab_size; ++i) {
      for (int j = 0; j < num_topics; ++j) {
        global_distribution_[j] += topic_distributions_[i][j];
      }
    }
    *word_index_map = word_index_map_;
  }

  void LDAModel::Unset() {
    for (size_t i = 0; i < memory_alloc_.size(); ++i) {
      memory_alloc_[i] = 0;
    }
  }

}  // namespace learning_lda
